import os
import matplotlib.pyplot as plt

batch_size = 4096
file_path = "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/random_models/20240320/random_cmp_output_mp.txt"
# file_path = "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/random_models/20240320/random_cmp_output_mp_b512.txt"
# file_path = "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/random_models/20240320/random_cmp_output_mp_b4k_ft500.txt"

# read the file line by line and filter out lines that don't start with "/"
with open(file_path, "r") as file:
    lines = file.readlines()
    lines = [line for line in lines if line.startswith("/")]

# split each line by the space char
lines = [line.split(" ") for line in lines]

# construct a pandas dataframe from the list of lists
import pandas as pd
cols = ["model_path", "batch_size", "fil_time", "tb_time", "total_speedup", "fil_kernel_time", "tb_kernel_time", "tb_kernel_speedup"]
df = pd.DataFrame(lines, columns=cols)

# cast the tb_kernel_speedup and total_speedup column to float
df["tb_kernel_speedup"] = df["tb_kernel_speedup"].astype(float)
df["total_speedup"] = df["total_speedup"].astype(float)

# compute it from the model_path as follows
# - split the model_path by the "/" char and get the file name
# - split the file name by the "_" char and get the third element
def get_num_trees(model_path):
    return int(model_path.split("/")[-1].split("_")[2])

# add a column to the dataframe called "num_trees". 
# This column contains the number of trees in the model
df["num_trees"] = df["model_path"].apply(get_num_trees)

def get_num_features(model_path):
    return int(model_path.split("/")[-1].split("_")[3])

df["num_features"] = df["model_path"].apply(get_num_features)

def get_tree_depth(model_path):
    return int(model_path.split("/")[-1].split("_")[4])

df["tree_depth"] = df["model_path"].apply(get_tree_depth)

# print(df.shape)
# print the min, max and geomean of the tb_kernel_speedup column
# print(df["tb_kernel_speedup"].min())
# print(df["tb_kernel_speedup"].max())
# print the geometric mean of the tb_kernel_speedup column
# compute geomean using scipy.stats.gmean
import scipy.stats as stats
# print(stats.gmean(df["tb_kernel_speedup"]))
#print(df["tb_kernel_speedup"].prod() ** (1.0 / df.shape[0]))

for depth in range(6, 9):
    # get entries with tree_depth == depth
    df_depth = df[df["tree_depth"] == depth]
    # print(df_depth.shape)

    # plot a 3-d graph of num_trees, num_features and tb_kernel_speedup

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 3D scatter plot
    # ax.scatter3D(df_depth["num_trees"], df_depth["num_features"], df_depth["tb_kernel_speedup"], c='red')
    surf = ax.plot_trisurf(df_depth["num_trees"], df_depth["num_features"], df_depth["tb_kernel_speedup"], cmap='viridis')

    ax.set_xlabel('Number of Trees')
    ax.set_ylabel('Number of Features')
    ax.set_zlabel('Speedup')
    min_speedup = df_depth["tb_kernel_speedup"].min()
    ax.set_zlim3d(zmin=0)
    max_speedup = df_depth["tb_kernel_speedup"].max()
    geomean_speedup = stats.gmean(df_depth["tb_kernel_speedup"])
    # limit the speedup values to 1 decimal digit
    plt.title(f"Min:{min_speedup:.2f}x    Max:{max_speedup:.2f}x    Geomean:{geomean_speedup:.2f}x")
    plt.colorbar(surf, label='Speedup')
    plt.savefig(f'kernel_speedup_b{batch_size}_depth{depth}.png')
    # plt.show()

# # Sort data in ascending order
# sorted_values = df["tb_kernel_speedup"].sort_values(ascending=True)

# # Calculate relative cumulative frequencies
# percentages = sorted_values.rank(pct=True)
# plt.plot(sorted_values, percentages) #, marker='o', linestyle='-')
# plt.xlabel('SilvanForge Speedup vs RAPIDS', fontsize=14)
# plt.ylabel('Cumulative Frequency', fontsize=14)
# plt.tick_params(labelsize=14)
# if batch_size == 4096:
#   plt.axis([0, 7, -0.05, 1.05])
# # else:
# #   plt.axis([0, 9, -0.05, 1.05])
# # plt.title('Ogive (CDF) of Values')
# plt.grid(True)
# plt.savefig(f'ogive_b{batch_size}.png')
