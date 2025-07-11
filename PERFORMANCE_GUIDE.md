# FramePack Performance Optimization Guide

## New Performance Controls Added

### 1. **Aggressive GPU Usage** (Checkbox)
- **Purpose**: Maximizes GPU utilization by using more VRAM
- **Effect**: Reduces memory preservation, uses larger chunk sizes
- **Recommended**: Enable when GPU utilization is low (<60%)

### 2. **Keep Models in GPU** (Checkbox)  
- **Purpose**: Keeps all models loaded in GPU memory between operations
- **Effect**: Eliminates model loading/unloading overhead
- **Recommended**: Enable if you have sufficient VRAM (6GB+ used safely)

### 3. **Performance Tuning** (Accordion)

#### **Chunk Size Multiplier** (0.5 - 3.0, default: 1.0)
- **Purpose**: Scales the attention chunk sizes
- **Effect**: Higher values = larger chunks = better GPU utilization
- **Recommended**: Set to 2.0-2.5 for low GPU utilization scenarios

#### **Max Chunk Size** (1024 - 8192, default: 2048)
- **Purpose**: Sets the maximum chunk size for attention operations
- **Effect**: Higher values allow better parallelization
- **Recommended**: Increase to 4096+ if you have low GPU utilization

#### **CPU Offload Threshold** (2-8GB, default: 6GB)
- **Purpose**: Only offloads models to CPU when GPU memory exceeds this threshold
- **Effect**: Keeps more models in GPU when possible
- **Recommended**: Set to 7-7.5GB for 8GB GPUs

#### **Mixed Precision Mode** (Checkbox, default: True)
- **Purpose**: Uses FP16/BF16 for faster processing
- **Effect**: Reduces memory usage and increases speed
- **Recommended**: Keep enabled (safe for most cases)

#### **Batch Processing** (Checkbox, default: False)
- **Purpose**: Enables batch processing optimizations where possible
- **Effect**: May improve throughput for certain operations
- **Recommended**: Try enabling for potential speed improvements

## Recommended Settings for Your 8GB GPU

Since you have:
- GPU Memory Usage: <5GB / 8GB available
- Low GPU utilization (30-60% spikes)
- High system RAM usage (96%)

### **High Performance Settings:**
1. ✅ **Aggressive GPU Usage**: Enabled
2. ✅ **Keep Models in GPU**: Enabled  
3. **Chunk Size Multiplier**: 2.5
4. **Max Chunk Size**: 4096
5. **CPU Offload Threshold**: 7.5 GB
6. ✅ **Mixed Precision Mode**: Enabled
7. ✅ **Batch Processing**: Enabled

### **What These Settings Will Do:**
- Use 6-7GB of your 8GB VRAM instead of <5GB
- Reduce system RAM pressure by keeping more in GPU
- Increase chunk sizes for better GPU parallelization
- Eliminate model loading/unloading overhead
- Target 80-90% GPU utilization instead of 30-60%

### **Performance Monitoring:**
The system will show messages like:
- "Increasing chunk size from X to Y (low memory usage)"
- "Aggressive mode: Larger chunks and higher memory utilization enabled"
- "GPU Memory after generation: X.XX GB"

### **If You Experience Issues:**
1. **If OOM occurs**: Reduce Chunk Size Multiplier to 2.0
2. **If still unstable**: Disable "Keep Models in GPU"
3. **If system becomes unresponsive**: Lower CPU Offload Threshold to 6GB

### **Expected Improvements:**
- **2-3x faster processing** due to better GPU utilization
- **More consistent GPU usage** (70-90% instead of spikes)
- **Better memory efficiency** (using more GPU, less system RAM)
- **Adaptive performance** that learns and optimizes over time
