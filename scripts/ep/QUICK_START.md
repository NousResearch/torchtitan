# ⚡ Quick Start - Fast Dashboard Loading

## TL;DR

**Problem:** Dashboard takes 60-90 seconds to load.

**Solution:** Pre-compute analysis once, dashboard loads in <5 seconds!

---

## 🚀 Quick Start

### Option 1: Automatic (Recommended)

```bash
./scripts/ep/START_DASHBOARD.sh
```

This script automatically:
1. Checks if cache exists
2. If not → runs pre-computation (~60 seconds, one-time)
3. Launches dashboard (<5 seconds)

### Option 2: Manual Control

```bash
# Step 1: Pre-compute analysis (one-time, ~60 seconds)
./scripts/ep/precompute_analysis.py

# Step 2: Launch dashboard (instant!)
./scripts/ep/START_DASHBOARD.sh
```

---

## 📊 What Gets Cached?

The pre-compute script processes:
- 2.4GB of raw trace files
- All EP=1 and EP=2 profiling data
- All ultra-deep analysis (communication, memory, modules, FLOPs)
- Saves to `scripts/ep/.analysis_cache.pkl` (~50-100 MB)

---

## ⏱️ Time Savings

| Action | Without Cache | With Cache |
|--------|---------------|------------|
| Dashboard Load | 60-90 seconds | <5 seconds |
| **Speedup** | **1x** | **12-18x faster!** |

---

## 🔄 When to Re-run Precompute

**Automatically handles:**
- If cache doesn't exist → runs automatically
- If cache is up-to-date → skips recomputation

**You should re-run after:**
- New profiling runs (traces changed)
- Want to force refresh

```bash
# Force refresh
rm scripts/ep/.analysis_cache.pkl
./scripts/ep/precompute_analysis.py
```

---

## 💡 Key Benefits

1. **Instant UI Access** - No more waiting for loading bars
2. **Full Functionality** - All 10 dashboard tabs work
3. **Smart Caching** - Auto-detects when to recompute
4. **Better Workflow** - Pre-compute once, use dashboard many times

---

## 📝 Complete Workflow Example

```bash
cd /home/phuc/workspace/moe/reference_repos/torchtitan-nous

# 1. Run profiling (if needed)
./scripts/ep/run_profiling.sh both

# 2. Just launch dashboard - it handles everything!
./scripts/ep/START_DASHBOARD.sh

# The script will:
# - Check for cache
# - Pre-compute if needed (first time only)
# - Launch dashboard instantly
# - Show you the public Gradio URL
```

---

## 📚 More Information

- **FAST_LOADING_GUIDE.md** - Complete technical details
- **DASHBOARD_MERGE_COMPLETE.md** - Ultra-deep analysis features
- **PROFILING_COMPLETE_README.md** - Profiling guide

---

**Status:** ✅ Fast loading system ready to use!
