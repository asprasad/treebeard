import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import gmean

# Read the CSV file into a dataframe
results_4060_df = pd.read_csv('../4060_thedeep_results.csv')
results_t400_df = pd.read_csv('../T400_holmes_results.csv')
fullexp_vs_at_df = pd.read_csv('../fullexp_vs_at.csv')
results_amdmi2160_df = pd.read_csv('../AMDMI2160_results.csv')

results_4060_df = results_4060_df[results_4060_df['Batch size'] != 256]
results_t400_df = results_t400_df[results_t400_df['Batch size'] != 256]
fullexp_vs_at_df = fullexp_vs_at_df[fullexp_vs_at_df['Batch size'] != 256]
results_amdmi2160_df = results_amdmi2160_df[results_amdmi2160_df['Batch size'] != 256]

PLOT_FILE_EXTENSION = 'png'
AXIS_FONT_SIZE = 15
LABEL_FONT_SIZE = 15
LEGEND_FONT_SIZE = 12
BAR_LABEL_FONT_SIZE = 8

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
    width = BAR_LABEL_FONT_SIZE
    gap = BAR_LABEL_FONT_SIZE * 0.3
    pos = np.array([float(2 * i * width + i * gap) for i in range(len(bar_names))])

    ax.bar(pos, rapids_speedup, width, label='Speedup of SilvanForge vs RAPIDS', hatch='//////')
    ax.bar(pos+width, tahoe_speedup, width, label='Speedup of SilvanForge vs Tahoe', hatch='......')

    # enable grid
    # ax.grid(True, which='both', axis='y')

    ax.set_ylim(0, max(max(rapids_speedup), max(tahoe_speedup)) + 1)

    # Add labels and title
    # ax.set_xlabel('Benchmark', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('Speedup', fontsize=LABEL_FONT_SIZE)
    # ax.set_title(f'Speedups for Batch Size {batch_size} (4060)')
    ax.set_xticks(pos + width / 2)
    ax.set_xticklabels(bar_names, rotation=90, fontsize=AXIS_FONT_SIZE)
    ax.axes.yaxis.set_tick_params(labelsize=AXIS_FONT_SIZE)        

    # Add value labels on top of each bar
    for benchmark, rs, rt, ts, tt in zip(pos, rapids_speedup, rapids_bar_top_label, tahoe_speedup, tahoe_bar_top_label):
        if rt == -1:
            continue

        rt = round(rt, 2)
        tt = round(tt, 2)

        plt.text(benchmark, rs + 0.2 , str(rt), ha='center', fontsize=BAR_LABEL_FONT_SIZE)
        plt.text(benchmark + width, ts + 0.2, str(tt), ha='center', fontsize=BAR_LABEL_FONT_SIZE)

    # Show the legend and plot
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'speedup_bar_graph_{batch_size}.{PLOT_FILE_EXTENSION}')

def plot_bar_graph_geomean_speedups_for_models(df, gpu_name):
    # Prepare data for plotting
    bar_names = df['Benchmark'].unique()
    bar_names.sort()
    batch_sizes = df['Batch size'].unique()
    batch_sizes.sort()
    speedups = [[],[]]

    for benchmark in bar_names:
        filtered_df = df[df['Benchmark'] == benchmark]
        rapids_speedups = []
        tahoe_speedups = []
        for batch_size in batch_sizes:
            batch_size_df = filtered_df[filtered_df['Batch size'] == batch_size]
            assert(batch_size_df.shape[0] == 1)

            tb_time = batch_size_df['TBKernel(AT)'].values[0]
            rapids_time = batch_size_df['RAPIDS(kernel)'].values[0]
            tahoe_time = batch_size_df['Tahoe(kernel)'].values[0]
            rapids_speedups.append(rapids_time / tb_time)
            tahoe_speedups.append(tahoe_time / tb_time)
        
        speedups[0].append(gmean(rapids_speedups))
        speedups[1].append(gmean(tahoe_speedups))

    # Plot the bar graph
    width = BAR_LABEL_FONT_SIZE
    gap = BAR_LABEL_FONT_SIZE * 0.3
    pos = np.array([float(2 * i * width + i * gap) for i in range(len(bar_names))])

    _, ax = plt.subplots()
    
    ax.bar(pos, speedups[0], width, label='Speedup of SilvanForge vs RAPIDS', hatch='//////')
    ax.bar(pos + width, speedups[1], width, label='Speedup of SilvanForge vs Tahoe', hatch='......')

    # Add labels and title
    ax.set_xlabel('Benchmark', fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel('Speedup', fontsize=AXIS_FONT_SIZE)
    ax.set_xticks(pos + width / 2)
    ax.set_xticklabels(bar_names, rotation=90, fontsize=AXIS_FONT_SIZE)
    ax.axes.yaxis.set_tick_params(labelsize=AXIS_FONT_SIZE)

    # Show the legend and plot
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'geomean_speedup_{gpu_name}_bar_graph.png', format='png', bbox_inches='tight')

def plot_double_y_axis_line_graph(df_4060, full_exp_df, gpu_name):
    # Prepare data for plotting
    batch_sizes = df_4060['Batch size'].unique()
    batch_sizes.sort()
    speedups = []
    norm_times = []

    for batch_size in batch_sizes:
        filtered_df = df_4060[df_4060['Batch size'] == batch_size]
        at_time = filtered_df['TBKernel(AT)'].to_list()
        best_time = filtered_df['TBKernel'].to_list()
        speedup = [ct/ tbt for ct, tbt in zip(best_time, at_time)]
        speedups.append(round(min(gmean(speedup), 1), 2))

    for batch_size in batch_sizes:
        filtered_df = full_exp_df[full_exp_df['Batch size'] == batch_size]
        fe_time = filtered_df['FullExplore'].to_list()
        heur_time = filtered_df['AT'].to_list()
        normalized_times = [ht/ft for ht, ft in zip(heur_time, fe_time)]
        norm_times.append(gmean(normalized_times))

    # Plot the line graph
    _, ax1 = plt.subplots()
    ax1.grid(True, which='both', axis='both')
    ax2 = ax1.twinx()
    ax1.plot(batch_sizes, speedups, 'o-', label='Heuristic Schedule vs Best 4060 Schedule')
    ax2.plot(batch_sizes, norm_times, 'o-', color='orange', label='Heuristic\'s Normalized Exploration Time')
    
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Batch size', fontsize=AXIS_FONT_SIZE)
    ax1.set_ylabel('Speedup', fontsize=AXIS_FONT_SIZE)
    ax2.set_ylabel('Normalized Time', fontsize=AXIS_FONT_SIZE)
    ax1.set_ylim(0, 1.5)
    ax2.set_ylim(0.005, 0.02)
    # ax1.set_title(f'Geomean Speedups Over Batch Size ({gpu_name})')
    #ax1.set_yticks(speedups)
    #ax1.set_yticklabels(speedups, fontsize=AXIS_FONT_SIZE)

    ax1.xaxis.set_ticks(batch_sizes)
    ax1.xaxis.set_ticklabels(batch_sizes, fontsize=AXIS_FONT_SIZE)
    ax1.yaxis.set_tick_params(labelsize=AXIS_FONT_SIZE)
    ax2.yaxis.set_tick_params(labelsize=AXIS_FONT_SIZE)

    # plt.show()
    ax1.legend(fontsize=10, loc='upper right')
    ax2.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig(f'speedup_vs_norm_time_line_graph_{gpu_name}.png')



class LineGraphPlotter:
    def __init__(self, gpu_name, file_name_suffix, lower_lim = 1, upper_lim = 1, round_up=True) -> None:
        self.gpu_name = gpu_name
        self.file_name_suffix = file_name_suffix
        _, self.ax = plt.subplots()
        self.ylim = (lower_lim, upper_lim)
        self.round_up = round_up

    def plot_geomean_speedups_over_batch_size(self, df, tb_time_col, compare_time_cols, line_labels):
        batch_sizes = df['Batch size'].unique()
        batch_sizes.sort()
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

        # Plot the line graph
        for batch_size_speedup, label in zip(speedups, line_labels):
            self.ax.plot(batch_sizes, batch_size_speedup, 'o-', label=label)
        
        self.ax.set_xscale('log', base=2)
        min_speedup = min([min(speedup) for speedup in speedups])
        diff = min_speedup - np.floor(min_speedup)
        lower_lim = np.floor(min_speedup) if diff <= 0.5 else np.floor(min_speedup) + 0.5
        max_speedup = max([max(speedup) for speedup in speedups])
        upper_lim = max_speedup if not self.round_up else round(max_speedup) + 0.5
        self.ylim = (min(self.ylim[0], lower_lim), max(self.ylim[1], upper_lim))

        # enable grid.
        self.ax.grid(True, which='both', axis='both')

        # Add labels and title
        self.ax.set_xlabel('Batch size', fontsize=LABEL_FONT_SIZE)
        self.ax.set_ylabel('Speedup', fontsize=LABEL_FONT_SIZE)
        # self.ax.set_title(f'Geomean Speedups Over Batch Size ({gpu_name})')
        self.ax.set_xticks(batch_sizes)
        self.ax.set_xticklabels(batch_sizes, fontsize=AXIS_FONT_SIZE)
        self.ax.axes.yaxis.set_tick_params(labelsize=AXIS_FONT_SIZE)

        # plt.show()
        # plt.savefig(f'geomean_speedup_{gpu_name}_{time_measured}.{PLOT_FILE_EXTENSION}')

    def save_plot(self, leg_font_size=LEGEND_FONT_SIZE):
        self.ax.set_ylim(self.ylim[0], self.ylim[1])
        plt.legend(fontsize=leg_font_size)
        plt.tight_layout()
        plt.savefig(f'geomean_speedup_{self.gpu_name}_{self.file_name_suffix}.{PLOT_FILE_EXTENSION}')

# plot absolute times bar graph for 1024 and 8192 batch sizes
# plot_bar_graph_of_abs_times_for_batch_size(results_4060_df, 1024)
# plot_bar_graph_of_abs_times_for_batch_size(results_4060_df, 8192)

# plot speedup bar graph for 1024 and 8192 batch sizes
plot_bar_graph_speedups_for_batch_size(results_4060_df, 1024)
plot_bar_graph_speedups_for_batch_size(results_4060_df, 8192)

# plot geomean speedup over batch sizes
line_graph_plotter = LineGraphPlotter('4060', 'kernel_time_total_time', lower_lim=1, upper_lim=5)
line_graph_plotter.plot_geomean_speedups_over_batch_size(results_4060_df, 'TBKernel(AT)', ['RAPIDS(kernel)', 'Tahoe(kernel)'], ['Speedup of SilvanForge vs RAPIDS(kernel)', 'Speedup of SilvanForge vs Tahoe(kernel)'])
line_graph_plotter.plot_geomean_speedups_over_batch_size(results_4060_df, 'TB (AT)', ['RAPIDS (Total)'], ['Speedup of SilvanForge vs RAPIDS(total)'])
line_graph_plotter.save_plot()

line_graph_plotter = LineGraphPlotter('T400', 'kernel_time')
line_graph_plotter.plot_geomean_speedups_over_batch_size(results_t400_df, 'TBKernel(AT)', ['RAPIDS(kernel)', 'Tahoe(kernel)'], ['Speedup of SilvanForge vs RAPIDS(kernel)', 'Speedup of SilvanForge vs Tahoe(kernel)'])
line_graph_plotter.plot_geomean_speedups_over_batch_size(results_t400_df, 'TB(AT)', ['RAPIDS(Total)'], ['Speedup of SilvanForge vs RAPIDS(total)'])
line_graph_plotter.save_plot()

line_graph_plotter = LineGraphPlotter('T400', '4060_vs_T400_vs_MI2160', lower_lim=1, upper_lim=1.6)
line_graph_plotter.plot_geomean_speedups_over_batch_size(results_t400_df, 'TBKernel(AT)', ['TBKernel(4060)'], ['Scheduling heuristic(T400) vs 4060 heuristic schedule'])
line_graph_plotter.plot_geomean_speedups_over_batch_size(results_amdmi2160_df, 'TBKernel(MI2160)', ['TBKernel(4060)'], ['Scheduling heuristic(MI2160) vs 4060 heuristic schedule'])
line_graph_plotter.save_plot(leg_font_size=11.5)

line_graph_plotter = LineGraphPlotter('4060', 'full_exp_vs_at', lower_lim=1, upper_lim=110)
line_graph_plotter.plot_geomean_speedups_over_batch_size(fullexp_vs_at_df, 'AT', ['FullExplore'], ['Scheduling heuristic vs Full Exploration'])
line_graph_plotter.save_plot()

plot_bar_graph_geomean_speedups_for_models(results_4060_df, '4060')
plot_bar_graph_geomean_speedups_for_models(results_t400_df, 'T400')

plot_double_y_axis_line_graph(results_4060_df, fullexp_vs_at_df, '4060')
