# nuScenes Dataset Download Guide

## Dataset Selection: v1.0 Trainval Split

**Why Trainval?**
- 850 scenes total (700 train + 150 val)
- Good balance for training AutoVLA model
- Includes all sensor modalities (camera, lidar, radar)
- ~300GB total size

---

## Download Instructions

### Step 1: Create Dataset Directory
```bash
mkdir -p /work/amd456/autovla/dataset/nuscenes
cd /work/amd456/autovla/dataset/nuscenes
```

### Step 2: Download Files

**From:** https://www.nuscenes.org/download

Download in this order:

#### Part A: Metadata (0.43 GB)
```
Full dataset (v1.0) → Trainval → Metadata
```
Location: `/work/amd456/autovla/dataset/nuscenes/v1.0-trainval_metadata.tar.gz`

#### Part B: Sensor File Blobs (292 GB total)
Download ALL 10 parts:
```
Full dataset (v1.0) → Trainval → File blobs part 1-10
```
- Part 1: 29.41 GB
- Part 2: 28.06 GB
- Part 3: 27.81 GB
- Part 4: 29.87 GB
- Part 5: 26.25 GB
- Part 6: 25.61 GB
- Part 7: 27.50 GB
- Part 8: 28.19 GB
- Part 9: 31.21 GB
- Part 10: 38.87 GB

Files: `/work/amd456/autovla/dataset/nuscenes/v1.0-trainval*.tar.gz`

---

## Step 3: Extract Files

```bash
cd /work/amd456/autovla/dataset/nuscenes

# Extract metadata
tar -xzf v1.0-trainval_metadata.tar.gz

# Extract all sensor blobs
for file in v1.0-trainval*.tar.gz; do
    tar -xzf "$file"
done
```

Expected folder structure after extraction:
```
/work/amd456/autovla/dataset/nuscenes/
├── v1.0-trainval/
│   ├── maps/
│   ├── samples/
│   ├── sweeps/
│   ├── v1.0-trainval_meta.json
│   └── ... (other metadata)
```

---

## Step 4: Verify Structure

```bash
ls -la /work/amd456/autovla/dataset/nuscenes/v1.0-trainval/
```

Should show:
- ✅ `maps/` folder
- ✅ `samples/` folder (camera images, lidar, radar)
- ✅ `sweeps/` folder (lidar sweeps)
- ✅ `v1.0-trainval_meta.json` (annotations)

---

## Storage Requirements

| Component | Size |
|-----------|------|
| Metadata | 0.43 GB |
| Sensor Blobs (10 parts) | ~292 GB |
| **Total** | **~300 GB** |

**Available at Arc:** `/work/amd456/` has 1.1 PB → ✅ No quota issues

---

## Timeline
- Download: 2-6 hours (depends on connection + server)
- Extract: 30-60 minutes
- Total: ~3-8 hours

---

## Next After Download

Once extracted, update config file:
```bash
# Edit training config
nano config/training/qwen2.5-vl-3B-mix-sft.yaml
```

Update paths:
```yaml
data:
  train:
    json_dataset_path: /work/amd456/autovla/dataset/nuscenes/v1.0-trainval
    sensor_data_path: /work/amd456/autovla/dataset/nuscenes/v1.0-trainval
```
