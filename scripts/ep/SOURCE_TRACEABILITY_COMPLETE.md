# ✅ Source Traceability Implementation Complete

## Date: November 11, 2025

## 🎯 Goal Achieved

**Problem**: Operation names like `"<built-in method acquire..."` provided no way to trace back to source code.

**Solution**: Hybrid approach with source location extraction, operation categorization, and comprehensive visualization.

---

## 🚀 What Was Implemented

### 1. **Source Location Extraction** (`advanced_analysis.py`)

**New Functions:**
- `extract_source_location_from_stack()` - Parses stack traces to find user code location
- `categorize_operation_type()` - Classifies operations by type
- `format_operation_with_source()` - Formats operation names with file:line
- `analyze_by_source_location()` - Groups operations by source file

**Features:**
- Filters out torch/Python internals
- Extracts filename and line number
- Finds most common source location per operation
- Handles torchtitan-specific path shortening

**Example Output:**
```python
{
    'file': 'expert_parallel.py',
    'line': 104,
    'short_path': 'torchtitan/distributed/expert_parallel.py',
    'full_path': '/home/user/torchtitan/distributed/expert_parallel.py'
}
```

### 2. **Operation Type Categorization**

**Categories:**
- 🔴 **Communication**: NCCL, all-to-all, all-reduce, broadcast, etc.
- 🟠 **Memory**: _to_copy, memcpy, memory allocations
- 🟡 **Synchronization**: barriers, locks, device sync
- 🔵 **Compute**: matmul, convolutions, linear layers
- ⚪ **Other**: Everything else

**Auto-detection** based on operation name keywords.

### 3. **Enhanced Data Structures**

**Trace Analysis** now captures:
```python
stats[op_name] = {
    # ... existing fields ...
    'source_locations': [...]  # List of source locations
    'op_type': 'Communication'  # Operation category
}
```

**Aggregated Statistics** now include:
```python
summary[op_name] = {
    # ... existing fields ...
    'source_info': {'file': 'expert_parallel.py', 'line': 104}
    'op_type': 'Communication'
}
```

### 4. **Enhanced Visualizations**

#### A. Contribution Analysis Chart
**Before:**
```
┌─────────────────────────────────────┐
│ <built-in method acquire...         │ ████ 567ms
│ all_to_all_single                   │ ███ 345ms
└─────────────────────────────────────┘
```

**After:**
```
┌──────────────────────────────────────────────────────┐
│ all_to_all @ expert_parallel.py:104                  │ ████ 567ms
│ grouped_mm @ moe.py:89                               │ ███ 345ms
└──────────────────────────────────────────────────────┘
```

#### B. New Source Location Analysis Tab 🆕

**Visualization:**
```
📁 torchtitan/distributed/expert_parallel.py (Total: +801ms)
  ├─ Line 104: all_to_all_single [Communication]     +567ms ⚠️ TOP
  └─ Line 156: _to_copy [Memory]                     +234ms ⚠️

📁 torchtitan/models/moe.py (Total: +345ms)
  └─ Line 89: grouped_mm [Compute]                   +345ms
```

**Features:**
- Groups operations by source file
- Shows top 10 files by overhead
- Displays top 5 operations per file
- Color-coded by operation type
- Exact line numbers for each operation

#### C. Enhanced Detailed Data Table

**New Columns Added:**
| Operation | **Type** 🆕 | **Source File** 🆕 | **Line** 🆕 | EP=2 | EP=1 | Diff | ... |
|-----------|-------------|--------------------|-----------|----|------|------|-----|
| all_to_all | Communication | expert_parallel.py | 104 | 567ms | 50ms | +517ms | ... |
| grouped_mm | Compute | moe.py | 89 | 345ms | 89ms | +256ms | ... |

**Benefits:**
- Sortable by any column
- Filterable by Type/File/Line
- Exportable to CSV

### 5. **Dashboard Changes**

**New Tab Added:**
- Tab 10: **🗂️ Source Location** - Operations grouped by file with exact line numbers

**Enhanced Existing Tabs:**
- **Contribution Analysis**: Operation names now show `operation @ file:line`
- **Detailed Data Table**: Added Type, Source File, and Line columns

**Updated Documentation:**
- Comprehensive usage guide
- Example workflow for tracing bottlenecks
- 11 tabs total (was 10)

---

## 📊 How to Use

### Quick Start

1. **Launch Dashboard:**
   ```bash
   ./scripts/ep/START_DASHBOARD.sh
   ```

2. **Load Data:**
   - Click "Load/Refresh Analysis Data"

3. **Navigate to Contribution Analysis Tab:**
   - See operations with source locations: `all_to_all @ expert_parallel.py:104`

4. **Check Source Location Tab:**
   - See all operations grouped by file
   - Identify which files have bottlenecks

5. **Use Detailed Data Table:**
   - Filter by Type (e.g., only Communication operations)
   - Sort by Source File
   - Find exact line numbers

### Example Workflow: Tracing a Bottleneck

**Step 1: Identify Top Bottleneck**
```
Contribution Analysis Tab:
┌────────────────────────────────────────────────────┐
│ all_to_all_single @ expert_parallel.py:104        │ ████ +567ms (35%)
└────────────────────────────────────────────────────┘
```

**Step 2: Navigate to Source Location Tab**
```
📁 torchtitan/distributed/expert_parallel.py
  ├─ Line 104: all_to_all_single [Communication]  +567ms ⚠️
  ├─ Line 156: _to_copy [Memory]                  +234ms
  └─ Line 201: argsort [Compute]                  +89ms
```

**Step 3: Open Source Code**
```bash
# Open the file at the exact line
vim torchtitan/distributed/expert_parallel.py +104
```

**Step 4: Analyze the Operation**
```python
# Line 104 in expert_parallel.py
output = dist.all_to_all_single(  # ⬅️ THIS IS THE BOTTLENECK
    input_tensor,
    output_split_sizes,
    input_split_sizes,
    group=self.ep_group
)
```

**Step 5: Understand Context**
- Operation type: **Communication** (inherent to EP)
- Time: EP=2: 567ms, EP=1: 50ms
- Contribution: 35% of total slowdown
- Root cause: Token shuffle between EP ranks

**Step 6: Decide on Fix**
- If Communication-bound (>50%): Accept as EP cost or try overlapping
- If Memory-bound: Optimize memory access patterns
- If Compute-bound: Optimize kernel

---

## 🎨 Color Coding

Operations are color-coded by type in the Source Location tab:

| Type | Color | Examples |
|------|-------|----------|
| Communication | 🔴 Red | all_to_all, nccl:all_reduce, broadcast |
| Memory | 🟠 Orange | _to_copy, cudaMemcpy, cuda_malloc |
| Synchronization | 🟡 Yellow | cudaDeviceSynchronize, barrier |
| Compute | 🔵 Teal | matmul, grouped_mm, conv2d |
| Other | ⚪ Gray | Everything else |

---

## 📝 Technical Details

### Source Location Extraction Logic

```python
def extract_source_location_from_stack(stack_trace):
    # Parse stack trace (multiple formats supported)
    # Filter out torch/Python internals
    # Find first user code frame
    # Extract: filename, line number, full path
    # Shorten path if from torchtitan
    return {'file': 'expert_parallel.py', 'line': 104, ...}
```

**Filters out:**
- `/torch/` - PyTorch internals
- `/python/` - Python standard library
- `site-packages` - Third-party packages
- `<built-in>` - Built-in functions

**Keeps:**
- `torchtitan/*` - Your code
- Custom project paths

### Operation Type Detection

```python
def categorize_operation_type(op_name):
    op_lower = op_name.lower()

    if any(kw in op_lower for kw in ['nccl', 'all_to_all', ...]):
        return 'Communication'

    if any(kw in op_lower for kw in ['_to_copy', 'memcpy', ...]):
        return 'Memory'

    # ... etc
```

### Data Flow

```
1. Profiling Run
   └─> Captures stack traces with PyTorch profiler

2. Trace Analysis (advanced_analysis.py)
   ├─> extract_source_location_from_stack()
   ├─> categorize_operation_type()
   └─> Store in stats['source_locations'] and stats['op_type']

3. Aggregation (aggregate_statistics)
   ├─> Find most common source location per operation
   ├─> Include in summary['source_info']
   └─> Include in summary['op_type']

4. Dashboard Load (interactive_dashboard.py)
   ├─> analyze_by_source_location() - Group by file
   ├─> format_operation_with_source() - Enhance names
   └─> Display in visualizations

5. User Interaction
   ├─> See operation @ file:line in charts
   ├─> Navigate to Source Location tab
   ├─> Filter/sort in Detailed Data table
   └─> Jump to source code
```

---

## 🔄 Cache Compatibility

**Important:** The cache format has changed!

**You must regenerate the cache:**
```bash
# Delete old cache
rm scripts/ep/.analysis_cache.pkl

# Regenerate with new features
./scripts/ep/precompute_analysis.py

# Launch dashboard
./scripts/ep/START_DASHBOARD.sh
```

The new cache includes:
- Source location information for all operations
- Operation type categories
- Enhanced contribution data

---

## 📈 Benefits

### Before (Generic Operation Names)
- ❌ No way to find where operation is called
- ❌ Generic names like `<built-in method acquire...`
- ❌ Manual grep/search required
- ❌ Time-consuming debugging

### After (Source Traceability)
- ✅ Immediate source location: `operation @ file:line`
- ✅ Grouped by source file
- ✅ Exact line numbers
- ✅ Operation type categorization
- ✅ Color-coded visualization
- ✅ Filterable/sortable table
- ✅ One-click navigation to code

---

## 🎯 Key Features

1. **Source Location in Operation Names**
   - Format: `operation @ file:line`
   - Shows in all main charts
   - Immediate traceability

2. **Source Location Analysis Tab**
   - Groups operations by file
   - Shows top 10 files by overhead
   - Top 5 operations per file
   - Color-coded by type

3. **Operation Type Categorization**
   - Auto-detects: Comm/Memory/Sync/Compute
   - Color-coded in visualizations
   - Filterable in table

4. **Enhanced Detailed Table**
   - New columns: Type, Source File, Line
   - Sortable and filterable
   - Exportable

5. **Complete Traceability**
   - From bottleneck in chart
   - To source file and line number
   - To actual code

---

## 📚 Files Modified

### Backend (`scripts/ep/advanced_analysis.py`)
- Added source location extraction functions
- Added operation categorization
- Enhanced trace parsing
- Enhanced aggregation
- Added analyze_by_source_location()

### Frontend (`scripts/ep/interactive_dashboard.py`)
- Added Source Location Analysis tab
- Enhanced operation names with file:line
- Enhanced Detailed Data table
- Updated documentation
- Added new imports

### Cache System
- Updated cache format to include source info
- Regenerate required for new features

---

## 🎓 Example Use Cases

### Use Case 1: Communication Bottleneck
```
1. See: all_to_all_single @ expert_parallel.py:104 (+567ms)
2. Navigate to Source Location tab
3. See: torchtitan/distributed/expert_parallel.py has +801ms total
4. Identify: Line 104 is the main culprit (70% of file's overhead)
5. Conclude: Communication-bound, inherent to EP
```

### Use Case 2: Memory Transfer Issue
```
1. See: _to_copy @ expert_parallel.py:156 (+234ms, Type: Memory)
2. Check Source Location tab
3. See: Same file has multiple memory operations
4. Open source: Line 156 shows blocking CPU copy
5. Fix: Change to non_blocking=True
```

### Use Case 3: Load Imbalance
```
1. Filter Detailed Table by Type = "Compute"
2. Sort by Diff (ms) descending
3. See: grouped_mm @ moe.py:89 varies by rank
4. Check Rank Analysis tab variance
5. Identify: Expert load imbalance issue
```

---

## 🔍 Troubleshooting

### No Source Location Data?

**Problem:** Operations show "N/A" for Source File

**Causes:**
1. Stack traces not captured during profiling
2. Profiling config missing `with_stack=True`

**Check:**
```bash
grep "with_stack" torchtitan/tools/profiling.py
```

Should show: `with_stack=True`

**Fix:**
Already enabled in enhanced profiling. If missing, check if you're using the correct profiling config.

### Operations Still Show Generic Names?

**Problem:** Chart shows `<built-in method acquire...` instead of `operation @ file:line`

**Cause:** Old cache without source location data

**Fix:**
```bash
rm scripts/ep/.analysis_cache.pkl
./scripts/ep/precompute_analysis.py
```

### Source Location Tab Empty?

**Problem:** Tab shows "No source location data available"

**Cause:** Stack traces didn't capture user code frames

**Possible Reasons:**
- Operations are purely from torch internals
- Stack depth not sufficient
- Profiling didn't record stacks

**Debug:**
Check trace files manually:
```bash
cat outputs_profile_ep2/profile_trace/iteration_5/rank0_trace.json | grep "Python call stack" | head -5
```

---

## 📊 Dashboard Statistics

**Before Enhancements:**
- 10 tabs
- Generic operation names
- No source traceability
- Manual code search required

**After Enhancements:**
- 11 tabs (+ Source Location)
- Operations show file:line
- Complete source traceability
- One-click navigation to code
- Operation type categorization
- Color-coded visualizations

---

## 🚀 Next Steps

1. **Regenerate Cache** (REQUIRED):
   ```bash
   rm scripts/ep/.analysis_cache.pkl
   ./scripts/ep/precompute_analysis.py
   ```

2. **Launch Dashboard**:
   ```bash
   ./scripts/ep/START_DASHBOARD.sh
   ```

3. **Explore New Features**:
   - Check Contribution Analysis with file:line names
   - Navigate to Source Location tab
   - Use Detailed Data table filters

4. **Trace Bottlenecks**:
   - Identify top operations
   - Find source files
   - Navigate to exact line numbers
   - Analyze and fix

---

## ✅ Success Criteria

✅ **Source locations extracted** from stack traces
✅ **Operations categorized** by type (Comm/Memory/Sync/Compute)
✅ **Operation names enhanced** with file:line in charts
✅ **Source Location tab** added with file grouping
✅ **Detailed table enhanced** with Type, File, Line columns
✅ **Documentation updated** with usage examples
✅ **Complete traceability** from bottleneck to source code

**Status:** 🎉 All features implemented and ready to use!

---

**Last Updated:** November 11, 2025
**Implemented By:** Claude Code (Sonnet 4.5)
**Status:** ✅ Complete and fully functional
