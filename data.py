import pandas as pd
import numpy as np
import sys

# 1. Load the file from CLI
if len(sys.argv) != 2:
    print("Error: Please provide a single filename.")
    sys.exit()

file_name = sys.argv[1]

try:
    df = pd.read_csv(file_name)
    print(f"Loaded {len(df)} rows.")
except:
    print(f"Error: File ({file_name}) not found.")
    sys.exit()

# 2. Time Parsing Function
def parse_time_to_ms(val):
    val = str(val).strip().lower()
    if pd.isna(val) or val == "": return 0.0
    val = val.replace('μs', 'us')
    
    try:
        if 'ms' in val: return float(val.replace('ms', ''))
        if 'us' in val: return float(val.replace('us', '')) / 1000.0
        if 'ns' in val: return float(val.replace('ns', '')) / 1_000_000.0
        if 's' in val:  return float(val.replace('s', '')) * 1000.0
        return float(val)
    except:
        return 0.0

# 3. Core Logic: Calculate "Sequential" Time (Flattening overlaps)
def calculate_sequential_time(df_slice):
    if df_slice.empty: return 0.0
    # Sort by start time to handle the timeline correctly
    df_slice = df_slice.sort_values('Start_ms')
    s = df_slice['Start_ms'].values
    e = df_slice['End_ms'].values
    
    # "High-Water Mark" Logic to merge overlaps
    me = np.maximum.accumulate(e)
    is_new = np.zeros(len(s), dtype=bool)
    is_new[0] = True
    if len(s) > 1:
        is_new[1:] = s[1:] > me[:-1]
    
    idx = np.where(is_new)[0]
    m_starts = s[idx]
    # The end of this block is either the max end seen so far, or the end of the very last kernel
    m_ends = np.append(me[idx[1:] - 1], me[-1])
    
    return np.sum(m_ends - m_starts)

# Apply Parsing
df['Start_ms'] = df['Start'].apply(parse_time_to_ms)
df['Duration_ms'] = df['Duration'].apply(parse_time_to_ms)
df['End_ms'] = df['Start_ms'] + df['Duration_ms']

# 4. Calculate Global Window & Total Active Time
total_window_time = df['End_ms'].max() - df['Start_ms'].min()

# Re-using the sequential logic on the WHOLE dataframe gives us True Active Time
true_active_time = calculate_sequential_time(df)
idle_time = total_window_time - true_active_time

# 5. Group Categories
def categorize_kernel(name):
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

df['Category'] = df['Name'].apply(categorize_kernel)

# 6. Generate Summary Report
# We apply the sequential calculation to each category individually
summary_data = []
for cat in df['Category'].unique():
    group = df[df['Category'] == cat]
    summary_data.append({
        'Category': cat,
        'Duration_ms': calculate_sequential_time(group),
        'Count': len(group)
    })

summary = pd.DataFrame(summary_data)

# Add Idle Row
idle_row = pd.DataFrame([{
    'Category': 'Idle Time (Waste)', 
    'Duration_ms': idle_time, 
    'Count': 0
}])
summary = pd.concat([summary, idle_row], ignore_index=True)

# Final Percentages: CALCULATED AGAINST TOTAL WINDOW for consistency
summary['% of Total'] = (summary['Duration_ms'] / total_window_time * 100).round(4)
summary = summary.sort_values('Duration_ms', ascending=False)

# Formatting
print("\n=== FINAL BREAKDOWN ===")
print(f"Total Window Analyzed: {total_window_time:.4f} ms")
print(f"True Active GPU Time (Wall-clock): {true_active_time:.4f} ms")
print("-" * 40)
print(summary.to_string(index=False))