import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re

# ==========================================
# 1. ROBUST TIME PARSING
# ==========================================

def parse_time(val):
    # 1. Convert to string and standardise
    val = str(val).strip().lower().replace('μs', 'us')
    
    # 2. CLEANUP: Remove question marks and other common noise
    val = val.replace('?', '').replace('~', '').strip()
    
    if pd.isna(val) or val == "": return 0.0
    
    factors = {'ms': 1, 'us': 1e-3, 'ns': 1e-6, 's': 1e3}
    
    # 3. Try to find unit and convert
    for unit, factor in factors.items():
        if unit in val:
            try:
                # Remove unit, strip whitespace, and convert
                clean_val = val.replace(unit, '').strip()
                return float(clean_val) * factor
            except ValueError:
                continue
                
    # 4. Fallback: Try converting raw number
    try: 
        return float(val)
    except ValueError: 
        return 0.0

# ==========================================
# 2. CORE LOGIC
# ==========================================

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
# 3. FILE PROCESSING (Returns Dict of GPUs)
# ==========================================

def process_file(filepath):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        return {}

    # Parse Columns
    df['Start_ms'] = df['Start'].apply(parse_time)
    df['End_ms'] = df['Start_ms'] + df['Duration'].apply(parse_time)
    df['Category'] = df['Name'].apply(categorize)

    # Global Window (Shared across all GPUs in this file)
    global_start = df['Start_ms'].min()
    global_end = df['End_ms'].max()
    total_window = global_end - global_start

    if total_window <= 0:
        return {}

    # Get Devices
    try:
        devices = sorted(df['Device'].unique())
    except:
        devices = df['Device'].unique()

    # Filter out CPU/Host
    devices = [d for d in devices if "host" not in str(d).lower()]

    gpu_profiles = {}

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
        
        # Save to result dict keyed by Device Name
        clean_device_name = str(device).replace('NVIDIA GeForce RTX', '').strip()
        gpu_profiles[clean_device_name] = profile

    return gpu_profiles, total_window

# ==========================================
# 4. MAIN EXECUTION & PLOTTING
# ==========================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python visualize_gpu_per_device.py file1.csv ...")

    files = sys.argv[1:]
    
    # We will build a list of dictionaries to convert to DataFrame later
    rows = [] 
    row_labels = []
    
    print(f"Analyzing {len(files)} files...")

    # 1. Process all files
    for f in files:
        fname = os.path.basename(f)
        # Shorten filename for display (remove .csv)
        short_fname = fname.replace('.csv', '')
        
        print(f"Processing: {fname}...")
        gpu_data, total_window = process_file(f)
        
        if gpu_data:
            # Sort GPUs so 0 comes before 1
            for gpu_name in sorted(gpu_data.keys()):
                data = gpu_data[gpu_name]
                rows.append(data)
                
                # Label format: "filename (GPU 0)"
                # Try to extract just the ID if it's long
                gpu_id_match = re.search(r'\((\d+)\)', gpu_name)
                if gpu_id_match:
                    gpu_short = f"GPU {gpu_id_match.group(1)}"
                else:
                    gpu_short = gpu_name
                
                row_labels.append(f"{short_fname}\n[{gpu_short}]")

    if not rows:
        sys.exit("No valid data found.")

    # 2. Create DataFrame
    df_abs = pd.DataFrame(rows, index=row_labels)
    df_abs = df_abs.fillna(0)

    # Calculate Total Times (should be roughly same per file, but good to check)
    total_times = df_abs.sum(axis=1)

    # 3. NORMALIZE
    df_plot = df_abs.div(total_times, axis=0) * 100

    # 4. Sort Columns
    priority_order = [
        'GEMM / Linear', 'Attention / KV Cache', 'MoE Logic', 'Comm (NCCL)', 
        'Activations', 'LLM Layers', 'Indexing/Sort', 'Memory', 'Other', 'Overhead', 'Idle (Waste)'
    ]
    existing_cols = [c for c in priority_order if c in df_plot.columns]
    remaining_cols = [c for c in df_plot.columns if c not in existing_cols]
    final_cols = existing_cols + remaining_cols
    df_plot = df_plot[final_cols]

    # 5. PLOTTING
    print("\nGenerating Normalized Per-Device Plot...")
    
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

    # Increase figure height if there are many bars
    fig_height = 6 + (len(rows) * 0.3)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    
    # Horizontal Bar Chart is often better for many labels
    # But let's stick to Vertical as requested, just wider
    df_plot.plot(kind='bar', stacked=True, color=plot_colors, ax=ax, edgecolor='black', width=0.8)

    # Formatting
    plt.title('GPU Breakdown per Device (Normalized %)', fontsize=14, pad=20)
    plt.ylabel('Percentage of Total Time (%)', fontsize=12)
    plt.xlabel('Experiment / GPU', fontsize=12)
    
    # Rotate x-labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.ylim(0, 115) 
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), title='Category', bbox_to_anchor=(1.01, 1), loc='upper left')

    # Add Total Time Labels
    for i, t in enumerate(total_times):
        ax.text(i, 102, f"{t:.0f}ms", ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=90)

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save
    output_filename = 'gpu_per_device_comparison.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to {output_filename}")
    plt.show()