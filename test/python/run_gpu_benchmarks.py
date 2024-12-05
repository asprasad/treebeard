import os
import sys
import pandas as pd
import numpy as np
import math
import time
import subprocess
import matplotlib.pyplot as plt
from scipy.stats import gmean

PLOT_FILE_EXTENSION = 'png'
AXIS_FONT_SIZE = 15
LABEL_FONT_SIZE = 15
LEGEND_FONT_SIZE = 11.5
BAR_LABEL_FONT_SIZE = 11

# Query the 'Batch size' field for a specific value (e.g., 512)
def plot_bar_graph_speedups_for_batch_size(df, batch_size, filename, kwargs=None):
    filtered_df = df[df['batch_size'] == batch_size]
    # print(filtered_df)

    # Prepare data for plotting
    bar_names = filtered_df['benchmark'].to_list() + ['geomean']
    tb_time = filtered_df['tb_kernel_time'].to_list()
    rapids_time = filtered_df['rapids_kernel_time'].to_list()
    tahoe_time = filtered_df['tahoe_total_time'].to_list()

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

    ax.bar(pos, rapids_speedup, width, label='Speedup of SilvanForge over RAPIDS', hatch='//////')
    ax.bar(pos+width, tahoe_speedup, width, label='Speedup of SilvanForge over Tahoe', hatch='......')

    # enable grid
    # ax.grid(True, which='both', axis='y')

    ax.set_ylim(0, max(max(rapids_speedup), max(tahoe_speedup)) + 2)

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

        plt.text(benchmark, rs + 0.2 , str(rt), ha='center', fontsize=BAR_LABEL_FONT_SIZE, rotation=90)
        plt.text(benchmark + width, ts + 0.2, str(tt), ha='center', fontsize=BAR_LABEL_FONT_SIZE, rotation=90)

    # Show the legend and plot
    plt.legend(fontsize=LEGEND_FONT_SIZE, loc='upper left')
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)

class LineGraphPlotter:
    def __init__(self, filename, gpu_name, lower_lim = 1, upper_lim = 1, round_up=True) -> None:
        self.filename = filename
        self.gpu_name = gpu_name
        _, self.ax = plt.subplots()
        self.ylim = (lower_lim, upper_lim)
        self.round_up = round_up

    def plot_geomean_speedups_over_batch_size(self, df, compare_time_cols, line_labels):
        batch_sizes = df['batch_size'].unique()
        batch_sizes.sort()
        speedups = []

        assert(len(compare_time_cols) == len(line_labels))

        for compare_time_col in compare_time_cols:
            batch_size_speedups = df[compare_time_col]
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
        plt.savefig(self.filename)


def parse_rapids_cmp_result(result_lines):
    # remove the last line from the result_lines
    result_lines = result_lines[:-1]

    results = []
    for line in result_lines[:-1]:
        cells = line.split(" ")
        benchmark = cells[0]
        batch_size = int(cells[1])
        rapids_total_time = float(cells[2])
        tb_total_time = float(cells[3])
        total_speedup = float(cells[4])
        rapids_kernel_time = float(cells[5])
        tb_kernel_time = float(cells[6])
        tb_kernel_speedup = float(cells[7])
        res = { "benchmark": benchmark, "batch_size": batch_size, 
                "rapids_total_time": rapids_total_time, "tb_total_time": tb_total_time, 
                "total_speedup": total_speedup, "rapids_kernel_time": rapids_kernel_time, 
                "tb_kernel_time": tb_kernel_time, "tb_kernel_speedup": tb_kernel_speedup }
        results.append(res)
    avg_speedups = result_lines[-1].split(" ")
    avg_total_speedup = float(avg_speedups[0])
    avg_kernel_speedup = float(avg_speedups[1])
    return results, avg_total_speedup, avg_kernel_speedup

def run_tahoe_benchmark(benchmark_name, batch_size):
    model_filename = benchmark_name + "_xgb_model_save.json.txt"
    model_filepath = os.path.join(tahoe_benchmarks_dir, model_filename)

    data_filename = model_filename + ".test.sampled.txt"
    data_filepath = os.path.join(tahoe_benchmarks_dir, str(batch_size), data_filename)

    command = [tahoe_exe_path, model_filepath, data_filepath]
    results = subprocess.run(command, capture_output=True, text=True)
    tahoe_results = results.stdout.split("\n")
    lines = [line for line in tahoe_results if line.startswith('*******')]
    assert len(lines) == 1
    total_time = float(lines[0].split(" ")[1])
    return total_time * 1e-6

if __name__ == "__main__":
    filepath = os.path.abspath(__file__)
    treebeard_repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(filepath)))
    treebeard_runtime_dir = os.path.join(os.path.join(treebeard_repo_dir, "src"), "python")

    rapids_script_path = os.path.join(treebeard_repo_dir, "test", "python", "RAPIDs", "benchmark_rapids.py")
    tahoe_dir = os.path.join(treebeard_repo_dir, "..", "Tahoe")
    tahoe_exe_path = os.path.join(tahoe_dir, "Tahoe")
    tahoe_benchmarks_dir = os.path.join(tahoe_dir, "Tahoe_expts", "treebeard_models")

    stderr_file = os.path.join(treebeard_repo_dir, "test", "python", "stderr.txt")

    # delete stderr file if it exists
    if os.path.exists(stderr_file):
        os.remove(stderr_file)

    BATCH_SIZES = [512, 1024, 2048, 4096, 8192, 16384]
    benchmarks = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "higgs", "letters", "year_prediction_msd"]

    benchmark_results = []
    summary_results = []

    for batch_size in BATCH_SIZES:
        command = ["python", rapids_script_path, "--batch_size", str(batch_size), "--num_trials", "3"]
        results = subprocess.run(command, capture_output=True, text=True)
        result_str = results.stdout
        rapids_results = result_str.split("\n")
        assert len(rapids_results) == 10
        rapids_results, avg_total_speedup, avg_kernel_speedup = parse_rapids_cmp_result(rapids_results)

        # append stderr to stderr_file
        with open(stderr_file, "a") as f:
           f.write(results.stderr)

        tahoe_speedups = []
        for (i, benchmark) in zip(range(0, len(benchmarks)), benchmarks):
            tahoe_total_time = run_tahoe_benchmark(benchmark, batch_size)
            rapids_results[i]["tahoe_total_time"] = tahoe_total_time
            speedup = tahoe_total_time / rapids_results[i]["tb_kernel_time"]
            rapids_results[i]["tahoe_speedup"] = speedup 
            tahoe_speedups.append(speedup)
        
        # find tahoe geomean speedup
        tahoe_avg_speedup = np.prod(tahoe_speedups)**(1.0/len(tahoe_speedups))

        benchmark_results.extend(rapids_results)
        
        summary = {"batch_size": batch_size, "avg_total_speedup": avg_total_speedup, 
                    "avg_kernel_speedup": avg_kernel_speedup, "tahoe_avg_speedup": tahoe_avg_speedup}
        summary_results.append(summary)

        print(rapids_results)
        print(summary)

    results_df = pd.DataFrame(benchmark_results)
    summary_df = pd.DataFrame(summary_results)

    print(results_df.to_string())
    print(summary_df.to_string())

    # import test_inputs
    # results_df, summary_df = test_inputs.get_result_dfs()

    # plot speedup bar graphs for each batch size
    plot_bar_graph_speedups_for_batch_size(results_df, 1024, "figure7b.png")
    plot_bar_graph_speedups_for_batch_size(results_df, 8192, "figure7c.png")

    # plot geomean speedup over batch sizes
    line_graph_plotter = LineGraphPlotter('figure7a.png', '4060', lower_lim=1, upper_lim=5)
    line_graph_plotter.plot_geomean_speedups_over_batch_size(summary_df, ['avg_kernel_speedup', 'tahoe_avg_speedup'], ['Speedup of SilvanForge over RAPIDS(kernel)', 'Speedup of SilvanForge over Tahoe(kernel)'])
    line_graph_plotter.plot_geomean_speedups_over_batch_size(summary_df, ['avg_total_speedup'], ['Speedup of SilvanForge over RAPIDS(total)'])
    line_graph_plotter.save_plot()
