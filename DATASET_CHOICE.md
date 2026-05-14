# Dataset Comparison for AutoVLA

## Dataset Options

| Dataset | Size | Download | Process | Training Time | Eval Tool | Notes |
|---------|------|----------|---------|---------------|-----------|-------|
| **nuScenes** | ~400GB | ~30 min | ~2 hours | Fast ✅ | Good | **Best for solo training** |
| **Waymo E2E** | ~500GB | ~1 hour | ~3 hours | Fast | Good | Challenging scenarios |
| **nuPlan** | ~1.2TB | ~2 hours | ~6 hours | Slower | Best (NAVSIM) | Official benchmark |
| **CARLA** | ~50GB | ~5 min | ~30 min | Very Fast | Limited | Simulation only |

---

## ⭐ RECOMMENDATION: nuScenes (BEST FOR YOU)

### Why nuScenes?
✅ **Smallest real-world dataset** (400GB)  
✅ **Fastest setup** (download + process = 3-4 hours)  
✅ **Good for quick testing** before larger datasets  
✅ **Good evaluation metrics** included  
✅ **Popular benchmark** widely used  
⚠️ Simpler urban scenarios than nuPlan  

### Size Breakdown
- **Raw download**: ~380-400GB
- **Processed data**: ~300-400GB
- **Total space needed**: ~800GB
- **Arc /work/ space**: 1.1 PB (plenty!)

---

## Alternative: CARLA (Fast Testing)

If you just want to **test code quickly**:
- ⚡ Only 50GB download
- ⚡ Process in 30 minutes
- ⚡ Full training in 4-6 hours
- ❌ Simulation only (less realistic)
- ✅ Good for debugging

---

## If You Have Time: nuPlan (Official Benchmark)

- Largest dataset (1.2TB)
- Official evaluation framework (NAVSIM)
- Best evaluation metrics
- Takes longer to process
- Benchmark used in paper

---

## 📊 Recommended Plan for Arc

### Option 1: Quick Test (CARLA) ⚡
```
Time: 1 day total
1. Download CARLA (5 min)
2. Process (30 min)
3. Setup (1 hour)
4. SFT Training (4-6 hours)
5. Test model (1 hour)
```

### Option 2: Production (nuScenes) ⭐ RECOMMENDED
```
Time: 3-4 days total
1. Download nuScenes (30 min)
2. Process (2-3 hours)
3. Setup (1 hour)
4. SFT Training (24-30 hours)
5. RFT Training (18-24 hours)
6. Evaluate (2-3 hours)
```

### Option 3: Full Benchmark (nuPlan)
```
Time: 5-7 days total
1. Download nuPlan (2 hours)
2. Process (5-6 hours)
3. Setup (1 hour)
4. SFT Training (36-48 hours)
5. RFT Training (24-36 hours)
6. Evaluate on NAVSIM (3-4 hours)
```

---

## Storage Calculation

**Arc /work/ usage:**
- Base code: ~2GB
- Downloaded data: ~400-1200GB
- Processed data: ~300-800GB
- Models (Qwen): ~50GB
- Logs + checkpoints: ~50-100GB
- **Total: ~500GB-2.2TB** ✅ (Arc has 1.1PB!)

---

## 🎯 What I Recommend

**Start with nuScenes:**
1. Medium size (good middle ground)
2. Fast download (30 min)
3. Fast processing (3 hours)
4. Standard benchmark
5. Can test code quickly

**Then move to nuPlan if needed:**
- For official benchmark results
- For better evaluation metrics

---

## Download Links

### nuScenes
```
https://www.nuscenes.org/
→ Registration required
→ Download v1.0-trainval split
→ ~400GB
```

### Waymo E2E
```
https://waymo.com/open/download/
→ Sign up required
→ Select End-to-End Driving dataset
→ ~500GB
```

### nuPlan
```
https://github.com/autonomousvision/navsim
→ Follow setup instructions
→ ~1.2TB
```

### CARLA
```
Official download:
https://github.com/carla-simulator/carla/releases
Or from repository
~50GB
```

---

## ✅ Next Steps

**Which dataset do you want to start with?**

**Option A:** nuScenes (RECOMMENDED)
- Best balance of speed + quality
- Most practical for Arc

**Option B:** CARLA (Fast)
- Quick testing
- Small size

**Option C:** nuPlan (Official)
- Best evaluation
- More time needed

**Option D:** Start small
- Just setup environment first
- Decide dataset later

Let me know your choice! 👈
