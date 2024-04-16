import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import gmean

# Read the CSV file into a dataframe
results_4060_df = pd.read_csv('../4060_thedeep_results.csv')
results_t400_df = pd.read_csv('../T400_holmes_results.csv')

PLOT_FILE_EXTENSION = 'png'

def plot_bar_graph_of_abs_times_for_batch_size(df, batch_size):
    filtered_df = df[df['Batch size'] == batch_size]
    bar_names = filtered_df['Benchmark'].to_list()
    tb_kernel_time = filtered_df['TBKernel(AT)'].to_list()
    rapids_kernel_time = filtered_df['RAPIDS(kernel)'].to_list()
    tb_total_time = filtered_df['TB (AT)'].to_list()
    rapids_total_time = filtered_df['RAPIDS (Total)'].to_list()

    # Calculate the difference between kernel time and total time
    tb_diff = [tb_total - tb_kernel for tb_total, tb_kernel in zip(tb_total_time, tb_kernel_time)]
    rapids_diff = [rapids_total - rapids_kernel for rapids_total, rapids_kernel in zip(rapids_total_time, rapids_kernel_time)]

    # Normalize w.r.t rapids_total_time
    norm_tb_kernel_time = [time/ rapids_total for time, rapids_total in zip(tb_kernel_time, rapids_total_time)]
    norm_rapids_kernel_time = [time/ rapids_total for time, rapids_total in zip(rapids_kernel_time, rapids_total_time)]
    norm_tb_diff = [time/ rapids_total for time, rapids_total in zip(tb_diff, rapids_total_time)]
    norm_rapids_diff = [time/ rapids_total for time, rapids_total in zip(rapids_diff, rapids_total_time)]
    norm_tb_total_time = [time/ rapids_total for time, rapids_total in zip(tb_total_time, rapids_total_time)]
    norm_rapids_total_time = [time/ rapids_total for time, rapids_total in zip(rapids_total_time, rapids_total_time)]

    # Prepare data for plotting
    width = 0.35
    pos = np.arange(len(bar_names))
    _, ax = plt.subplots()
    b1 = ax.bar(pos, norm_tb_kernel_time, width, label='Kernel Time')
    b2 = ax.bar(pos + width, norm_rapids_kernel_time, width, label='RAPIDS Kernel Time')
    ax.bar(pos, norm_tb_diff, width, bottom=norm_tb_kernel_time, label='Transfer Time', hatch='//////', color=b1.patches[0].get_facecolor())
    ax.bar(pos + width, norm_rapids_diff, width, bottom=norm_rapids_kernel_time, label='RAPIDS Transfer Time', hatch='//////', color=b2.patches[0].get_facecolor())

    # Add labels and title
    ax.set_xlabel('Benchmark')
    ax.set_ylabel('Normalized Time')
    # ax.set_title(f'Absolute Times for Batch Size {batch_size}')
    ax.set_xticks(pos + width / 2)
    ax.set_xticklabels(bar_names, rotation=45)

    ax.set_ylim(0, 1.5)

    # Add value labels on top of each bar
    for benchmark, tbt, rt, ntt, nrt in zip(pos, tb_total_time, rapids_total_time, norm_tb_total_time, norm_rapids_total_time):
        tbt = round(tbt * 10 ** 6, 2)
        rt = round(rt * 10 ** 6, 2)
        plt.text(benchmark, ntt + 0.02, str(tbt), ha='center', fontsize=8)
        plt.text(benchmark + width, nrt + 0.02, str(rt), ha='center', fontsize=8)

    # Show the legend and plot
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'abs_times_bar_graph_{batch_size}.{PLOT_FILE_EXTENSION}', format=PLOT_FILE_EXTENSION, bbox_inches='tight')

# Query the 'Batch size' field for a specific value (e.g., 512)
def plot_bar_graph_speedups_for_batch_size(df, batch_size, kwargs=None):
    filtered_df = df[df['Batch size'] == batch_size]
    # print(filtered_df)

    # Prepare data for plotting
    bar_names = filtered_df['Benchmark'].to_list() + ['geomean']
    tb_time = filtered_df['TBKernel(AT)'].to_list()
    rapids_time = filtered_df['RAPIDS(kernel)'].to_list()
    tahoe_time = filtered_df['Tahoe(kernel)'].to_list()

    rapids_speedup = [rt/ tbt for rt, tbt in zip(rapids_time, tb_time)]
    tahoe_speedup = [tt/ tbt for tt, tbt in zip(tahoe_time, tb_time)]

    rapids_speedup += [gmean(rapids_speedup)]
    tahoe_speedup += [gmean(tahoe_speedup)]


    # show label in microseconds
    rapids_bar_top_label = [x * 10 ** 6 for x in rapids_time] 
    tahoe_bar_top_label = [x * 10 ** 6 for x in tahoe_time]

    rapids_bar_top_label += [-1]
    tahoe_bar_top_label += [-1]

    _, ax = plt.subplots()

    # Plot the bar graph
    width = 0.6
    gap = 0.4
    pos = np.array([float(i) + i * gap for i in range(len(bar_names))])

    ax.bar(pos, rapids_speedup, width, label='RAPIDS')
    ax.bar(pos+width, tahoe_speedup, width, label='Tahoe')

    # Add labels and title
    ax.set_xlabel('Benchmark')
    ax.set_ylabel('Speedup')
    # ax.set_title(f'Speedups for Batch Size {batch_size} (4060)')
    ax.set_xticks(pos + width / 2)
    ax.set_xticklabels(bar_names, rotation=45)

    # Add value labels on top of each bar
    for benchmark, rs, rt, ts, tt in zip(pos, rapids_speedup, rapids_bar_top_label, tahoe_speedup, tahoe_bar_top_label):
        if rt == -1:
            continue

        rt = round(rt, 2)
        tt = round(tt, 2)

        plt.text(benchmark, rs + 0.2 , str(rt), ha='center', fontsize=8)
        plt.text(benchmark + width, ts + 0.2, str(tt), ha='center', fontsize=8)

    # Show the legend and plot
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'speedup_bar_graph_{batch_size}.{PLOT_FILE_EXTENSION}')

def plot_line_graph_geomean_speedups_over_batch_size(df, tb_time_col, compare_time_cols, line_labels, gpu_name, time_measured):
    batch_sizes = df['Batch size'].unique()
    batch_sizes = [batch_size for batch_size in batch_sizes if batch_size != 256]
    speedups = []

    assert(len(compare_time_cols) == len(line_labels))

    for compare_time_col in compare_time_cols:
        batch_size_speedups = []
        for batch_size in batch_sizes:
            filtered_df = df[df['Batch size'] == batch_size]
            compare_time = filtered_df[compare_time_col].to_list()
            tb_time = filtered_df[tb_time_col].to_list()
            speedup = [ct/ tbt for ct, tbt in zip(compare_time, tb_time)]
            batch_size_speedups.append(gmean(speedup))
        
        speedups.append(batch_size_speedups)

    _, ax = plt.subplots()

    # Plot the line graph
    for batch_size_speedup, label in zip(speedups, line_labels):
        ax.plot(batch_sizes, batch_size_speedup, 'o-', label=label)
    
    ax.set_xscale('log', base=2)
    ax.set_ylim(1)

    # Add labels and title
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Speedup')
    # ax.set_title(f'Geomean Speedups Over Batch Size ({gpu_name})')
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels(batch_sizes)

    # Show the legend and plot
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'geomean_speedup_{gpu_name}_{time_measured}.{PLOT_FILE_EXTENSION}')

# plot absolute times bar graph for 1024 and 8192 batch sizes
plot_bar_graph_of_abs_times_for_batch_size(results_4060_df, 1024)
plot_bar_graph_of_abs_times_for_batch_size(results_4060_df, 8192)

# plot speedup bar graph for 1024 and 8192 batch sizes
plot_bar_graph_speedups_for_batch_size(results_4060_df, 1024)
plot_bar_graph_speedups_for_batch_size(results_4060_df, 8192)

# plot geomean speedup over batch sizes
plot_line_graph_geomean_speedups_over_batch_size(results_4060_df, 'TBKernel(AT)', ['RAPIDS(kernel)', 'Tahoe(kernel)'], ['RAPIDS', 'Tahoe'], '4060', 'kernel_time')
plot_line_graph_geomean_speedups_over_batch_size(results_4060_df, 'TB (AT)', ['RAPIDS (Total)'], ['RAPIDS'], '4060', 'total_time')
plot_line_graph_geomean_speedups_over_batch_size(results_t400_df, 'TBKernel(AT)', ['RAPIDS(kernel)', 'Tahoe(kernel)'], ['RAPIDS', 'Tahoe'], 'T400', 'kernel_time')

