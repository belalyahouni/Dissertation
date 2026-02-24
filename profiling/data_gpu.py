import pandas as pd
import numpy as np
import sys

# 1. Load the file
if len(sys.argv) != 2:
    sys.exit("Error: Please provide a single filename.")

try:
    df = pd.read_csv(sys.argv[1])
    print(f"Loaded {len(df)} rows.")
except Exception as e:
    sys.exit(f"Error loading file: {e}")

# 2. Time Parsing Logic
def parse_time(val):
    val = str(val).strip().lower().replace('μs', 'us')
    if pd.isna(val) or val == "": return 0.0
    factors = {'ms': 1, 'us': 1e-3, 'ns': 1e-6, 's': 1e3}
    for unit, factor in factors.items():
        if unit in val:
            return float(val.replace(unit, '')) * factor
    try: return float(val)
    except: return 0.0

# 3. Core Logic: Flatten overlaps into sequential "wall-clock" time
def calculate_sequential_time(df_slice):
    if df_slice.empty: return 0.0
    df_slice = df_slice.sort_values('Start_ms')
    s, e = df_slice['Start_ms'].values, df_slice['End_ms'].values
    
    me = np.maximum.accumulate(e)
    is_new = np.ones(len(s), dtype=bool)
    if len(s) > 1: is_new[1:] = s[1:] > me[:-1]
    
    idx = np.where(is_new)[0]
    return np.sum(np.append(me[idx[1:] - 1], me[-1]) - s[idx])

# 4. Categorization Logic
def categorize(name):
    name = str(name).lower()

    # 1. MoE Logic (Mixture of Experts)
    if any(x in name for x in ['moe', 'topkgating', 'count_and_sort', 'align_block_size']):
        return 'MoE Logic'

    # 2. GEMM / Linear (Matrix Multiplications & BLAS)
    if any(x in name for x in ['cutlass', 'ampere', 'gemm', 'gemv', 'cublas', 'wma_tensorop']):
        return 'GEMM / Linear'

    # 3. Attention & Cache (FlashAttention & vLLM KV Cache)
    if any(x in name for x in ['flash', 'reshape_and_cache', 'attention']):
        return 'Attention / KV Cache'

    # 4. LLM Specific Layers (Norms, Embeddings, Loss)
    if any(x in name for x in ['cross_entropy', 'rotary_embedding', 'rms_norm', 'softmax']):
        return 'LLM Layers (Norm/Embed/Loss)'

    # 5. Communication (Multi-GPU / NCCL)
    if any(x in name for x in ['nccl', 'all_reduce', 'all_gather', 'cross_device_reduce']):
        return 'Comm (NCCL)'

    # 6. Activations & Triton Fusions (Pointwise & Reductions)
    if any(x in name for x in ['triton_red', 'triton_poi', 'act_and_mul', 'silu', 'gelu', 'pow_rsqrt']):
        return 'Activations / Triton Fusion'

    # 7. PyTorch Native / Indexing / Sorting
    # This captures the massive amount of 'at::native' and 'cub' sorting in your image
    if any(x in name for x in ['at::native', 'at_cuda', 'radixsort', 'index_elementwise', 'scatter_gather']):
        return 'PyTorch Native / Indexing'

    # 8. Memory Operations
    if any(x in name for x in ['memcpy', 'memset', 'direct_copy']):
        return 'Memory'

    # 9. Infrastructure / C++ Templates
    if any(x in name for x in ['std::enable_if', 'thrust']):
        return 'Utility / Overhead'

    return 'Other'

# Prepare Data
print("Parsing timestamps...")
df['Start_ms'] = df['Start'].apply(parse_time)
df['End_ms'] = df['Start_ms'] + df['Duration'].apply(parse_time)
df['Category'] = df['Name'].apply(categorize)

# 5. Establish Shared Global Window (Crucial for TP/PP comparison)
# We want to see if one GPU is waiting on the other, so we use the 
# min/max of the ENTIRE trace, not just the specific GPU.
global_start = df['Start_ms'].min()
global_end = df['End_ms'].max()
total_window_time = global_end - global_start

print(f"\nGlobal Experiment Window: {total_window_time:.4f} ms")
print("=" * 60)

# 6. Loop through each unique Device
devices = sorted(df['Device'].unique())

for device in devices:
    print(f"\nREPORT FOR: {device}")
    
    # Filter data for this specific GPU
    gpu_df = df[df['Device'] == device].copy()
    
    # Calculate Metrics for this GPU
    gpu_active_time = calculate_sequential_time(gpu_df)
    gpu_idle_time = total_window_time - gpu_active_time
    
    # Group by Category for this GPU
    summary = gpu_df.groupby('Category').apply(
        lambda x: pd.Series({
            'Duration_ms': calculate_sequential_time(x),
            'Count': len(x)
        }), include_groups=False
    ).reset_index()
    
    # Add Idle Row
    idle_row = pd.DataFrame([{
        'Category': 'Idle Time (Waste)', 
        'Duration_ms': gpu_idle_time, 
        'Count': 0
    }])
    summary = pd.concat([summary, idle_row], ignore_index=True)
    
    # Percentages against SHARED Global Window
    summary['% of Total'] = (summary['Duration_ms'] / total_window_time * 100).round(4)
    summary = summary.sort_values('Duration_ms', ascending=False)
    
    print(f"Active Time: {gpu_active_time:.4f} ms | Utilization: {(gpu_active_time/total_window_time*100):.2f}%")
    print("-" * 50)
    print(summary.to_string(index=False))
    print("=" * 60)