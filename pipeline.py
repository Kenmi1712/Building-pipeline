#!/usr/bin/env python3
import os
import sys
import glob
import shutil
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
import torch
import torch.nn as nn
from config import *

# ---------------------------------------------------
# LIBRARY PATH (for vedas raster)
# ---------------------------------------------------
sys.path.append("/opt/vedas_env/lib/raster_data_system")
import vedas_raster_lib.data as data

# ---------------------------------------------------
# CITY BOUNDING BOXES
# ---------------------------------------------------
city_bbox = {
    "ahmedabad": (72.394107, 22.888827, 72.763000, 23.209796),
    "gandhinagar": (72.45, 23.05, 72.75, 23.35),
    "vadodara": (73.0591, 22.20, 73.35, 22.45),
    "bhuj": (69.60, 23.20, 69.80, 23.35),
}

# ---------------------------------------------------
# MSI CONFIG
# ---------------------------------------------------
args2_base = {
    "dataset_id": DATASET_ID,
    "merge_method": "max",
    "indexes": [2, 3, 4, 8, 15],  # Last is SCL band
    "nodata": "255"
}

# ---------------------------------------------------
# OUTPUT FOLDERS
# ---------------------------------------------------
OUTPUT_ROOT = "./output"                # intermediate city-wise
FINAL_OUTPUT = "./final_output"         # cloud-free final rasters
REJECT_FOLDER = os.path.join(OUTPUT_ROOT, "no_need")  # cloud + empty

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(FINAL_OUTPUT, exist_ok=True)
os.makedirs(REJECT_FOLDER, exist_ok=True)

# ---------------------------------------------------
# DATABASE FETCH
# ---------------------------------------------------
def fetch_city_dates():
    query = """
    SELECT r."timestamp", c.city
    FROM published_rasters r
    JOIN LATERAL (
        SELECT 'ahmedabad' city WHERE ST_Intersects(r.bounds, ST_MakeEnvelope(72.394107,22.888827,72.763000,23.209796,4326))
    ) c ON TRUE
    WHERE r.dataset_id=%s AND r."timestamp" BETWEEN %s AND %s
    """
    records = set()
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, (DATASET_ID, FROM_DATE, TO_DATE))
        for r in cur.fetchall():
            records.add((r["city"], r["timestamp"].strftime("%Y%m%d")))
    conn.close()
    return sorted(records)

# def fetch_city_dates():
#     query = """
#     SELECT r."timestamp", c.city
#     FROM published_rasters r
#     JOIN LATERAL (
#         SELECT 'ahmedabad' city WHERE ST_Intersects(r.bounds, ST_MakeEnvelope(72.394107,22.888827,72.763000,23.209796,4326))
#         UNION ALL SELECT 'gandhinagar' city WHERE ST_Intersects(r.bounds, ST_MakeEnvelope(72.45,23.05,72.75,23.35,4326))
#         UNION ALL SELECT 'vadodara' city WHERE ST_Intersects(r.bounds, ST_MakeEnvelope(73.0591,22.20,73.35,22.45,4326))
#         UNION ALL SELECT 'bhuj' city WHERE ST_Intersects(r.bounds, ST_MakeEnvelope(69.60,23.20,69.80,23.35,4326))
#     ) c ON TRUE
#     WHERE r.dataset_id=%s AND r."timestamp" BETWEEN %s AND %s
#     """
#     records = set()
#     conn = psycopg2.connect(
#         host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
#     )
#     with conn.cursor(cursor_factory=RealDictCursor) as cur:
#         cur.execute(query, (DATASET_ID, FROM_DATE, TO_DATE))
#         for r in cur.fetchall():
#             records.add((r["city"], r["timestamp"].strftime("%Y%m%d")))
#     conn.close()
#     return sorted(records)

# ---------------------------------------------------
# RASTER FETCH / SAVE
# ---------------------------------------------------
def get_raster(algo_args, bbox=None, width=None, height=None, projection=None):
    if 'nodata' in algo_args and algo_args['nodata'] is not None:
        algo_args['nodata'] = int(algo_args['nodata'])
    else:
        algo_args['nodata'] = None

    if 'indexes' not in algo_args:
        algo_args['indexes'] = [1]

    if 'merge_method' not in algo_args:
        algo_args['merge_method'] = 'last'

    if isinstance(algo_args['indexes'], str):
        algo_args['indexes'] = list(map(int, algo_args['indexes'].split(',')))

    raster_data = data.get_raster(
        algo_args['dataset_id'],
        algo_args['from_time'],
        algo_args['to_time'],
        bbox,
        projection,
        0,
        algo_args['merge_method'],
        width,
        height,
        indexes=algo_args['indexes'],
        nodata=algo_args['nodata']
    )
    return raster_data

def save_raster_tif(filename, array, bbox, projection='EPSG:4326'):
    if array.ndim == 2:
        height, width = array.shape
        array = array.reshape(1, height, width)
    else:
        _, height, width = array.shape

    transform = from_bounds(*bbox, width, height)
    with rasterio.open(
        filename, 'w', driver='GTiff',
        height=height, width=width, count=array.shape[0],
        dtype=array.dtype, crs=projection, transform=transform
    ) as dst:
        for i in range(array.shape[0]):
            dst.write(array[i], i+1)

def is_cloudy_tile(scl_band, threshold=0.80):
    cloud_classes = {8, 9, 10, 11}
    cloud_fraction = np.isin(scl_band, list(cloud_classes)).sum() / scl_band.size
    return cloud_fraction >= threshold

# ---------------------------------------------------
# CNN MODEL
# ---------------------------------------------------
class ImprovedCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(4,16,3,padding='same'); self.bn1=nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,3,padding='same'); self.bn2=nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,64,3,padding='same'); self.bn3=nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,64,5,padding='same'); self.bn4=nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64,16,1); self.bn5=nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16,1,1)
        self.relu = nn.ReLU(); self.dropout=nn.Dropout2d(dropout_rate); self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x))); x=self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x))); x=self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x))); x=self.dropout(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.sigmoid(self.conv6(x))
        return x

# ---------------------------------------------------
# UTILITY
# ---------------------------------------------------
def standardize_channels(msi_array, target_channels=4):
    c,h,w = msi_array.shape
    if c==target_channels: return msi_array
    elif c>target_channels: return msi_array[:target_channels]
    else:
        padded = np.zeros((target_channels,h,w),dtype=msi_array.dtype)
        padded[:c]=msi_array
        return padded

def apply_cloud_mask(msi_array, cloud_band_values=[4,5,7], fill_value=-1):
    last_band = msi_array[-1]
    mask = np.isin(last_band, cloud_band_values)
    msi_masked = msi_array.copy()
    for i in range(msi_array.shape[0]):
        msi_masked[i, ~mask]=fill_value
    return msi_masked, mask

# ---------------------------------------------------
# PROCESS + INFERENCE
# ---------------------------------------------------
def process_msi_file(model, msi_file, dst_dir, processed_set, device='cuda', threshold=0.5):
    basename = Path(msi_file).stem           # MSI_ahmedabad_20260101
    parts = basename.split('_')
    if len(parts) < 3:
        return
    city = parts[1]
    date_str = parts[2]

    year, month, day = date_str[:4], date_str[4:6], date_str[6:8]

    target_dir = Path(dst_dir)/year/month/day
    target_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # FIXED FILENAME: date at end
    # ----------------------------
    dst_file = target_dir / f"MSI_{city}_probability_{date_str}.tif"

    if str(dst_file) in processed_set or dst_file.exists():
        return

    with rasterio.open(msi_file) as src:
        arr = src.read()
        bbox = src.bounds
        crs = src.crs

    arr = standardize_channels(arr,4)
    if arr.shape[0]>4:
        arr,_ = apply_cloud_mask(arr,[4,5,7],-1)
        arr[arr==-1]=0

    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(arr.astype(np.float32)/10000.0).unsqueeze(0).to(device)
        prob = model(tensor).cpu().numpy()[0,0]
        pred = (prob>threshold).astype(np.uint8)
        prob_scaled = (prob*100).astype(np.uint16)

    save_raster_tif(dst_file, prob_scaled, bbox, crs)
    processed_set.add(str(dst_file))
    with open('processed.log','a') as f:
        f.write(str(dst_file)+'\n')
    print(f"✓ Processed and saved: {dst_file}")


# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------
def run_pipeline():
    records = fetch_city_dates()
    print(f"Fetched {len(records)} records from DB")

    seen = set()
    for city, date in records:
        if (city,date) in seen: continue
        seen.add((city,date))

        bbox = city_bbox.get(city)
        if bbox is None: continue

        print(f"\nProcessing CITY={city} DATE={date}")

        city_dir = os.path.join(OUTPUT_ROOT, city)
        os.makedirs(city_dir, exist_ok=True)

        args2 = args2_base.copy()
        args2['from_time']=date; args2['to_time']=date
        ras = get_raster(args2,bbox=bbox,projection='EPSG:4326')
        if ras is None: continue
        arr = ras.read(); scl_band=arr[4] if arr.shape[0]>4 else arr[-1]

        if not np.any(arr!=0):
            save_raster_tif(os.path.join(REJECT_FOLDER,f"{city}_{date}_empty.tif"),arr,bbox)
            continue

        if is_cloudy_tile(scl_band,0.8):
            save_raster_tif(os.path.join(REJECT_FOLDER,f"MSI_{city}_{date}_cloud_tile.tif"),arr,bbox)
        else:
            out_file = os.path.join(FINAL_OUTPUT,f"MSI_{city}_{date}.tif")
            save_raster_tif(out_file,arr,bbox)

    # ===== RUN CNN INFERENCE ON FINAL OUTPUT =====
    model = ImprovedCNN()
    model.load_state_dict(torch.load('best_model_seasonal.pth', map_location='cpu', weights_only=True))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    processed_set = set()
    if os.path.exists('processed.log'):
        with open('processed.log','r') as f:
            processed_set = set(line.strip() for line in f.readlines())

    DST = Path("/68_data/PROCESSED_DATA_THEMES/T6/T6S1/s2_ai_building_probability_T6S1P15")
    DST.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(os.path.join(FINAL_OUTPUT,'MSI_*.tif')))
    print(f"\nRunning CNN inference on {len(files)} files...\n")
    for i,f in enumerate(files,1):
        print(f"[{i}/{len(files)}] {Path(f).name}")
        try:
            process_msi_file(model,f,DST,processed_set,device=device,threshold=0.4)
        except Exception as e:
            print(f"✗ ERROR {f}: {e}")

if __name__=='__main__':
    run_pipeline()
    print("\n✓ Full pipeline completed successfully!")
