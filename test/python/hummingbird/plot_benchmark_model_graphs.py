from io import StringIO
import pandas as pd
import numpy as np

results_str = """
              benchmark  batch_size   hb_jit_time  hb_torch_time   hb_tvm_time  tb_total_time  hb_jit_speedup  hb_torch_speedup  hb_tvm_speedup
0               abalone         512  6.174724e-07   9.548569e-07  5.432970e-07   9.807856e-08        6.295691          9.735633        5.539406
1               airline         512  4.515468e-07   6.649090e-07  1.927008e-07   7.849104e-08        5.752846          8.471145        2.455068
2           airline-ohe         512  1.112831e-06   1.617967e-06  1.212663e-06   5.996790e-07        1.855711          2.698055        2.022187
3               covtype         512  7.271694e-07   1.071535e-06  5.716667e-07   1.480040e-07        4.913173          7.239903        3.862508
4               epsilon         512  1.193970e-06   1.388234e-06  1.164111e-06   1.059253e-06        1.127181          1.310578        1.098992
5                 higgs         512  4.726780e-07   6.792749e-07  2.064045e-07   8.720009e-08        5.420614          7.789841        2.367022
6               letters         512  3.726803e-05   5.680939e-05  1.095718e-05   2.379520e-06       15.661993         23.874303        4.604785
7   year_prediction_msd         512  4.592577e-07   6.638960e-07  2.214128e-07   1.129363e-07        4.066521          5.878501        1.960511
8               abalone        1024  6.344735e-07   1.171183e-06  4.545278e-07   7.044335e-08        9.006861         16.625885        6.452388
9               airline        1024  2.456673e-07   3.297351e-07  1.168005e-07   4.622985e-08        5.314039          7.132514        2.526516
10          airline-ohe        1024  1.213017e-06   2.024952e-06  1.001485e-06   4.585800e-07        2.645159          4.415701        2.183883
11              covtype        1024  6.016380e-07   1.190222e-06  4.805163e-07   1.897336e-07        3.170961          6.273119        2.532583
12              epsilon        1024  9.311603e-07   1.022600e-06  9.456691e-07   9.110766e-07        1.022044          1.122408        1.037969
13                higgs        1024  2.586921e-07   3.466802e-07  1.246850e-07   5.405418e-08        4.785793          6.413568        2.306666
14              letters        1024  3.700639e-05   5.648594e-05  1.081216e-05   1.839639e-06       20.116117         30.704910        5.877328
15  year_prediction_msd        1024  2.768569e-07   3.691902e-07  1.463950e-07   8.934560e-08        3.098719          4.132159        1.638525
16              abalone        2048  1.273858e-06   1.850706e-06  4.076118e-07   5.170968e-08       24.634811         35.790325        7.882698
17              airline        2048  1.574941e-07   1.753832e-07  7.291894e-08   3.275584e-08        4.808124          5.354256        2.226135
18          airline-ohe        2048  2.104718e-06   2.850296e-06  8.753489e-07   3.695196e-07        5.695822          7.713517        2.368883
19              covtype        2048  1.286910e-06   1.836872e-06  4.408417e-07   9.222280e-08       13.954356         19.917771        4.780181
20              epsilon        2048  8.124589e-07   8.280914e-07  8.617409e-07   8.464766e-07        0.959813          0.978280        1.018033
21                higgs        2048  1.627873e-07   1.766596e-07  1.067506e-07   8.148329e-08        1.997800          2.168047        1.310092
22              letters        2048  3.693084e-05   5.637238e-05  1.086731e-05   1.764722e-06       20.927286         31.944059        6.158089
23  year_prediction_msd        2048  1.984599e-07   2.085734e-07  1.364219e-07   1.110395e-07        1.787291          1.878372        1.228589
24              abalone        4096  1.484244e-06   2.194628e-06  4.252558e-07   4.838056e-08       30.678522         45.361782        8.789807
25              airline        4096  1.425005e-07   1.357378e-07  5.803420e-08   2.560342e-08        5.565683          5.301552        2.266659
26          airline-ohe        4096  2.320987e-06   3.273492e-06  8.447077e-07   3.384487e-07        6.857721          9.672047        2.495822
27              covtype        4096  1.651630e-06   2.342443e-06  4.263663e-07   9.302855e-08       17.754008         25.179827        4.583176
28              epsilon        4096  8.544174e-07   8.432631e-07  8.174919e-07   8.233142e-07        1.037778          1.024230        0.992928
29                higgs        4096  1.686308e-07   1.434002e-07  6.593977e-08   3.152575e-08        5.348985          4.548668        2.091616
30              letters        4096  3.701527e-05   5.639412e-05  1.082332e-05   1.867802e-06       19.817550         30.192765        5.794682
31  year_prediction_msd        4096  1.792618e-07   1.640559e-07  1.157077e-07   5.610093e-08        3.195345          2.924298        2.062492
32              abalone        8192  1.458514e-06   2.199171e-06  4.183202e-07   4.197267e-08       34.749143         52.395297        9.966489
33              airline        8192  9.560099e-08   1.505255e-07  4.660076e-08   1.895615e-08        5.043271          7.940724        2.458345
34          airline-ohe        8192  2.273907e-06   3.274753e-06  8.207820e-07   3.261951e-07        6.971003         10.039246        2.516230
35              covtype        8192  1.619791e-06   2.413561e-06  3.962591e-07   5.436562e-08       29.794411         44.394986        7.288782
36              epsilon        8192  8.573441e-07   8.944107e-07  7.927612e-07   7.901448e-07        1.085047          1.131958        1.003311
37                higgs        8192  1.036406e-07   1.597967e-07  5.513527e-08   3.065186e-08        3.381218          5.213279        1.798757
38              letters        8192  1.000000e+00   1.000000e+00  1.080486e-05   1.869327e-06   534951.939091     534951.939091        5.780083
39  year_prediction_msd        8192  1.363504e-07   1.799270e-07  8.422071e-08   5.448625e-08        2.502474          3.302246        1.545724
40              abalone       16384  1.436962e-06   2.183755e-06  3.983857e-07   4.240203e-08       33.889001         51.501196        9.395441
41              airline       16384  1.741275e-07   2.323783e-07  4.022831e-08   1.492478e-08       11.667001         15.569961        2.695403
42          airline-ohe       16384  2.247030e-06   3.257467e-06  8.037296e-07   3.219812e-07        6.978762         10.116948        2.496200
43              covtype       16384  1.588820e-06   2.395662e-06  3.827577e-07   5.538951e-08       28.684487         43.251184        6.910293
44              epsilon       16384  8.692124e-07   9.251148e-07  7.748092e-07   7.794342e-07        1.115184          1.186906        0.994066
45                higgs       16384  1.850138e-07   2.396667e-07  5.358144e-08   2.575859e-08        7.182605          9.304341        2.080139
46              letters       16384  1.000000e+00   1.000000e+00  8.998938e-06   1.725320e-06   579602.734933     579602.734933        5.215809
47  year_prediction_msd       16384  2.078216e-07   2.643270e-07  7.135464e-08   4.500973e-08        4.617259          5.872664        1.585316
"""
summary_str = """
   batch_size  avg_jit_speedup  avg_torch_speedup  avg_tvm_speedup
0         512         4.351248           6.243237         2.658423
1        1024         4.324878           6.451240         2.598542
2        2048         5.354017           6.632965         2.586821
3        4096         7.210603           8.370161         2.944902
4        8192         6.153747           8.646658         3.037443
5       16384         8.390338          11.246380         3.049002
"""

results_io = StringIO(results_str)
summary_io = StringIO(summary_str)

results_df = pd.read_table(results_io, delim_whitespace=True)
summary_df = pd.read_table(summary_io, delim_whitespace=True)

print(results_df.to_string())
print(summary_df.to_string())

def get_result_dfs():
    return results_df, summary_df

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
    # change the bar_names element with "year_prediction_msd" to "year"
    bar_names = [x if x != "year_prediction_msd" else "year" for x in bar_names]

    tb_time = filtered_df['tb_total_time'].to_list()
    hb_jit_time = filtered_df['hb_jit_time'].to_list()
    hb_torch_time = filtered_df['hb_torch_time'].to_list()
    hb_tvm_time = filtered_df['hb_tvm_time'].to_list()

    jit_speedup = [rt/ tbt if rt != 1.0 else 0.0 for rt, tbt in zip(hb_jit_time, tb_time)]
    torch_speedup = [tt/ tbt if tt != 1.0 else 0.0 for tt, tbt in zip(hb_torch_time, tb_time)]
    tvm_speedup = [tvmt/ tbt for tvmt, tbt in zip(hb_tvm_time, tb_time)]

    jit_speedup += [gmean(jit_speedup)]
    torch_speedup += [gmean(torch_speedup)]
    tvm_speedup += [gmean(tvm_speedup)]


    # show label in microseconds
    jit_bar_top_label = [x * 10 ** 6 if x!=1.0 else -1 for x in hb_jit_time] 
    torch_bar_top_label = [x * 10 ** 6 if x!=1.0 else -1 for x in hb_torch_time]
    tvm_bar_top_label = [x * 10 ** 6 for x in hb_tvm_time]

    jit_bar_top_label += [-1]
    torch_bar_top_label += [-1]
    tvm_bar_top_label += [-1]

    _, ax = plt.subplots()

    # Plot the bar graph
    width = BAR_LABEL_FONT_SIZE
    gap = BAR_LABEL_FONT_SIZE * 0.3
    pos = np.array([float(3 * i * width + i * gap) for i in range(len(bar_names))])

    ax.bar(pos, jit_speedup, width, label='Speedup of SilvanForge over RAPIDS', hatch='//////')
    ax.bar(pos+width, torch_speedup, width, label='Speedup of SilvanForge over Tahoe', hatch='......')
    ax.bar(pos+2*width, tvm_speedup, width, label='Speedup of SilvanForge over TVM', hatch='----')

    # enable grid
    # ax.grid(True, which='both', axis='y')

    ax.set_ylim(0, max(max(jit_speedup), max(torch_speedup), max(tvm_speedup)) + 2)

    # Add labels and title
    # ax.set_xlabel('Benchmark', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('Speedup', fontsize=LABEL_FONT_SIZE)
    # ax.set_title(f'Speedups for Batch Size {batch_size} (4060)')
    ax.set_xticks(pos + width / 2)
    ax.set_xticklabels(bar_names, rotation=90, fontsize=AXIS_FONT_SIZE)
    ax.axes.yaxis.set_tick_params(labelsize=AXIS_FONT_SIZE)        

    # Add value labels on top of each bar
    for benchmark, rs, rt, ts, tt, tvm_speedup, tvm_label in zip(pos, jit_speedup, jit_bar_top_label, torch_speedup, torch_bar_top_label, tvm_speedup, tvm_bar_top_label):
        if rt != -1:
            rt = round(rt, 2)
            tt = round(tt, 2)

            plt.text(benchmark, rs + 0.2 , str(rt), ha='center', fontsize=BAR_LABEL_FONT_SIZE, rotation=90)
            plt.text(benchmark + width, ts + 0.2, str(tt), ha='center', fontsize=BAR_LABEL_FONT_SIZE, rotation=90)

        if tvm_label != -1:
            tvm_label = round(tvm_label, 2)
            plt.text(benchmark + 2*width, tvm_speedup + 0.2, str(tvm_label), ha='center', fontsize=BAR_LABEL_FONT_SIZE, rotation=90)

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

if __name__ == "__main__":
    # plot speedup bar graphs for each batch size
    plot_bar_graph_speedups_for_batch_size(results_df, 1024, "figure9b.png")
    plot_bar_graph_speedups_for_batch_size(results_df, 8192, "figure9c.png")

    # plot geomean speedup over batch sizes
    line_graph_plotter = LineGraphPlotter('figure9a.png', '4060', lower_lim=0, upper_lim=5)
    line_graph_plotter.plot_geomean_speedups_over_batch_size(summary_df, ['avg_jit_speedup', 'avg_torch_speedup'], ['Speedup of SilvanForge over HB(torch.jit)', 'Speedup of SilvanForge over HB(torch)'])
    line_graph_plotter.plot_geomean_speedups_over_batch_size(summary_df, ['avg_tvm_speedup'], ['Speedup of SilvanForge over HB(tvm)'])
    line_graph_plotter.save_plot()
