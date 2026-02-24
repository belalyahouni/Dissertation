import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ==========================================
# 1. HELPER FUNCTIONS 
# ==========================================

def parse_time(val):
    val = str(val).strip().lower().replace('μs', 'us')
    if pd.isna(val) or val == "": return 0.0
    factors = {'ms': 1, 'us': 1e-3, 'ns': 1e-6, 's': 1e3}
    for unit, factor in factors.items():
        if unit in val:
            return float(val.replace(unit, '')) * factor
    try: return float(val)
    except: return 0.0

def calculate_sequential_time(df_slice):
    if df_slice.empty: return 0.0
    df_slice = df_slice.sort_values('Start_ms')
    s, e = df_slice['Start_ms'].values, df_slice['End_ms'].values
    
    me = np.maximum.accumulate(e)
    is_new = np.ones(len(s), dtype=bool)
    if len(s) > 1: is_new[1:] = s[1:] > me[:-1]
    
    idx = np.where(is_new)[0]
    return np.sum(np.append(me[idx[1:] - 1], me[-1]) - s[idx])

def categorize(name):
    name = str(name).lower()
    if any(x in name for x in ['moe', 'topkgating', 'count_and_sort', 'align_block_size']): return 'MoE Logic'
    if any(x in name for x in ['cutlass', 'ampere', 'gemm', 'gemv', 'cublas', 'wma_tensorop']): return 'GEMM / Linear'
    if any(x in name for x in ['flash', 'reshape_and_cache', 'attention']): return 'Attention / KV Cache'
    if any(x in name for x in ['cross_entropy', 'rotary_embedding', 'rms_norm', 'softmax']): return 'LLM Layers'
    if any(x in name for x in ['nccl', 'all_reduce', 'all_gather', 'cross_device_reduce']): return 'Comm (NCCL)'
    if any(x in name for x in ['triton_red', 'triton_poi', 'act_and_mul', 'silu', 'gelu', 'pow_rsqrt']): return 'Activations'
    if any(x in name for x in ['at::native', 'at_cuda', 'radixsort', 'index_elementwise', 'scatter_gather']): return 'Indexing/Sort'
    if any(x in name for x in ['memcpy', 'memset', 'direct_copy']): return 'Memory'
    if any(x in name for x in ['std::enable_if', 'thrust']): return 'Overhead'
    return 'Other'

# ==========================================
# 2. FILE PROCESSING LOGIC
# ==========================================

def process_file(filepath):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        return None

    # Parse Columns
    df['Start_ms'] = df['Start'].apply(parse_time)
    df['End_ms'] = df['Start_ms'] + df['Duration'].apply(parse_time)
    df['Category'] = df['Name'].apply(categorize)

    # Global Window for this specific file
    global_start = df['Start_ms'].min()
    global_end = df['End_ms'].max()
    total_window = global_end - global_start

    # Get Devices
    try:
        devices = sorted(df['Device'].unique())
    except:
        devices = df['Device'].unique() # Fallback if sort fails

    # Filter out CPU/Host if necessary
    devices = [d for d in devices if "host" not in str(d).lower()]

    if not devices:
        return None

    # Aggregate metrics across all GPUs
    aggregated_profile = {} 
    
    for device in devices:
        gpu_df = df[df['Device'] == device].copy()
        
        # Calculate active time per category
        cat_times = gpu_df.groupby('Category').apply(
            lambda x: calculate_sequential_time(x), include_groups=False
        )
        
        # Calculate total active and idle
        total_active = calculate_sequential_time(gpu_df)
        idle_time = total_window - total_active
        
        # Store
        profile = cat_times.to_dict()
        profile['Idle (Waste)'] = idle_time
        
        # Add to aggregate
        for k, v in profile.items():
            aggregated_profile[k] = aggregated_profile.get(k, 0) + v

    # Average the results by number of devices
    final_avg_profile = {k: v / len(devices) for k, v in aggregated_profile.items()}
    return final_avg_profile

# ==========================================
# 3. MAIN EXECUTION & PLOTTING
# ==========================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python visualize_gpu_norm.py file1.csv file2.csv ...")

    files = sys.argv[1:]
    all_data = {}

    print(f"Analyzing {len(files)} files...")

    # 1. Process all files
    for f in files:
        fname = os.path.basename(f)
        print(f"Processing: {fname}...")
        result = process_file(f)
        if result:
            all_data[fname] = result

    if not all_data:
        sys.exit("No valid data found.")

    # 2. Create DataFrame (Absolute Values first)
    df_abs = pd.DataFrame(all_data).T 
    df_abs = df_abs.fillna(0)

    # Calculate Total Times for Labels later
    total_times = df_abs.sum(axis=1)

    # 3. NORMALIZE to Percentages (0-100%)
    df_plot = df_abs.div(total_times, axis=0) * 100

    # 4. Sort Columns for consistent stacking order
    # 'Idle' last so it's on top/end of bar
    priority_order = [
        'GEMM / Linear', 'Attention / KV Cache', 'MoE Logic', 'Comm (NCCL)', 
        'Activations', 'LLM Layers', 'Indexing/Sort', 'Memory', 'Other', 'Overhead', 'Idle (Waste)'
    ]
    existing_cols = [c for c in priority_order if c in df_plot.columns]
    remaining_cols = [c for c in df_plot.columns if c not in existing_cols]
    final_cols = existing_cols + remaining_cols
    df_plot = df_plot[final_cols]

    # 5. PLOTTING
    print("\nGenerating Normalized Comparison Plot...")
    
    # Colors
    colors = {
        'GEMM / Linear': '#1f77b4',       
        'Attention / KV Cache': '#ff7f0e',
        'MoE Logic': '#2ca02c',           
        'Comm (NCCL)': '#d62728',         
        'Activations': '#9467bd',         
        'Idle (Waste)': '#d3d3d3',        
        'Memory': '#8c564b',              
        'LLM Layers': '#e377c2',          
        'Indexing/Sort': '#7f7f7f',       
        'Other': '#bcbd22',               
        'Overhead': '#17becf'             
    }
    plot_colors = [colors.get(c, '#333333') for c in df_plot.columns]

    # Create Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Stacked Bar Chart
    df_plot.plot(kind='bar', stacked=True, color=plot_colors, ax=ax, edgecolor='black', width=0.6)

    # Formatting
    plt.title('GPU Breakdown (Normalized %)', fontsize=14, pad=20)
    plt.ylabel('Percentage of Total Time (%)', fontsize=12)
    plt.xlabel('Experiment Trace', fontsize=12)
    plt.xticks(rotation=0 if len(files) < 4 else 45)
    plt.ylim(0, 110) # Give space for text labels
    
    # Add Legend (Reverse order to match visual stack usually)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), title='Category', bbox_to_anchor=(1.01, 1), loc='upper left')

    # Add Total Time Labels on top of bars
    for i, (fname, total_ms) in enumerate(total_times.items()):
        ax.text(i, 101, f"Total:\n{total_ms:.0f} ms", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save
    output_filename = 'gpu_comparison_normalized.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to {output_filename}")
    plt.show()