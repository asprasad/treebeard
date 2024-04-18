# plot a 3-d graph of num_trees, num_features and tb_kernel_speedup
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# get the path of the current script
import os
script_path = os.path.dirname(os.path.realpath(__file__))
# append ../thedeep/gpu/sensitivity to the path
sensitivity_path = os.path.join(script_path, "..", "thedeep", "gpu", "sensitivity")

def get_degradation_vals(df_benchmark, X, Y, x_col="cur_batch_size", y_col="cmp_batch_size"):
    # get the degradation values for the given benchmark
    degradation_vals = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            x = X[i]
            y = Y[j]
            df_depth = df_benchmark[(df_benchmark[x_col] == x) & (df_benchmark[y_col] == y)]
            # assert df_depth.shape[0] == 1, f"Expected 1 row for {x}, {y}, got {df_depth.shape[0]}"
            if df_depth.shape[0] == 0:
                degradation_vals[i, j] = -1
                continue
            degradation_vals[i, j] = df_depth["degradation"].values[0]
            # if x < y:
            #     degradation_vals[i, j] = 10 * degradation_vals[i, j]

    max_val = np.max(degradation_vals)
    for deg_row in degradation_vals:
        for i in range(len(deg_row)):
            if deg_row[i] == -1:
                deg_row[i] = max_val
    return degradation_vals

def plot_batch_sensitivity_for_single_benchmark(df_depth, benchmark_name):
    # draw an intensity plot of the degradation values
    # x axis is cur_batch_size, y axis is cmp_batch_size, intensity is degradation
    # x and y axis in log scale

    # get a numpy array for the x, y and z axis
    x = df_depth["cur_batch_size"].unique()
    y = df_depth["cmp_batch_size"].unique()
    x.sort()
    y.sort()
    Z = get_degradation_vals(df_depth, x, y) # df_depth['degradation'].values.reshape(len(y), len(x))
    # print(Z)

    da = xr.DataArray(
        Z,
        coords={"Batch Size": x, "Comparison Batch Size": y},
        dims=("Batch Size", "Comparison Batch Size"),
    )

    fg = da.plot(xscale="log", yscale="log", aspect=1.3, size=20)
    fg.colorbar.ax.tick_params(labelsize=40)
    for axis in [fg.axes.xaxis, fg.axes.yaxis]:
        axis.label.set_size(60)
        axis.set_tick_params(labelsize=60)

    # plt.show()
    # plt.title(f"{benchmark_name} Batch Sensitivity")
    # plt.colorbar(im, label='tb_kernel_speedup')
    plt.tight_layout()
    plt.savefig(f'batch_sensitivity_{benchmark_name}.png')
    plt.close('all')

def plot_model_sensitivity_for_single_benchmark(df_depth, model_name):
    # draw an intensity plot of the degradation values
    # x axis is cur_batch_size, y axis is cmp_batch_size, intensity is degradation
    # x and y axis in log scale

    # get a numpy array for the x, y and z axis
    x = df_depth["benchmark"].unique()
    y = df_depth["cmp_benchmark"].unique()

    x.sort()
    y.sort()
    Z = get_degradation_vals(df_depth, x, y, "benchmark", "cmp_benchmark") # df_depth['degradation'].values.reshape(len(y), len(x))
    # print(Z)

    x_positions = [(i+1) * 10 for i in range(len(x))]
    y_positions = [(i+1) * 10 for i in range(len(y))]
    
    da = xr.DataArray(
        Z,
        coords={"Model": x_positions, "Comparison Model": y_positions},
        dims=("Model", "Comparison Model"),
    )
    fg = da.plot(aspect=1.3, size=20)
    fg.colorbar.ax.tick_params(labelsize=40)
    plt.axis('off')

    for name, x_pos in zip(x, x_positions):
        name = name if name != "year_prediction_msd" else "year" 
        plt.text(x_pos, 2, name, rotation=90, fontsize=60, va = 'top')
    
    for name, y_pos in zip(y, y_positions):
        name = name if name != "year_prediction_msd" else "year"
        plt.text(2, y_pos, name, fontsize=60, ha='right')
    
    # plt.show()
    # plt.title(f"Batch Size {batch_size} Model Sensitivity")
    plt.tight_layout()
    # plt.colorbar(im, label='tb_kernel_speedup')
    plt.savefig(f'model_sensitivity_{batch_size}.png')
    plt.close('all')

batch_csv = os.path.join(sensitivity_path, "batch.csv")
model_csv = os.path.join(sensitivity_path, "model.csv")
df = pd.read_csv(batch_csv)
# print(df.head()) 
# add 1 to every entry in the degradation column of df
df["degradation"] = df["degradation"] + 1
# get the unique benchmark names
benchmark_names = df["benchmark"].unique()
benchmark_names.sort()
for benchmark_name in benchmark_names:
    df_depth = df[df["benchmark"] == benchmark_name]
    assert df_depth.shape[0] == 49, f"Expected 49 rows for {benchmark_name}, got {df_depth.shape[0]}"
    plot_batch_sensitivity_for_single_benchmark(df_depth, benchmark_name)
    # break

# filter out the rows where both batch sizes are equal
df = df[df["cur_batch_size"] != df["cmp_batch_size"]]
# plot a histogram of the degradation values
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(df["degradation"], bins=100)
ax.set_xlabel('Slowdown')
ax.set_ylabel('Frequency')
plt.title("Batch Sensitivity Histogram")
plt.savefig('batch_sensitivity_histogram.png')
plt.close("all")

df_model = pd.read_csv(model_csv)
df_model["degradation"] = df_model["degradation"] + 1
batch_sizes = df_model["cur_batch_size"].unique()
batch_sizes.sort()
for batch_size in batch_sizes:
    df_depth = df_model[df_model["cur_batch_size"] == batch_size]
    # assert df_depth.shape[0] == 64, f"Expected 64 rows for {batch_size}, got {df_depth.shape[0]}"
    plot_model_sensitivity_for_single_benchmark(df_depth, f"batch_size_{batch_size}")
    # break

# plot a histogram of the degradation values
df_model = df_model[df_model["benchmark"] != df_model["cmp_benchmark"]]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(df_model["degradation"], bins=100)
ax.set_xlabel('Slowdown')
ax.set_ylabel('Frequency')
plt.title("Model Sensitivity Histogram")
plt.savefig('model_sensitivity_histogram.png')