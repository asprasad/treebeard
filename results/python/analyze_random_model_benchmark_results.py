import os

batch_size = 4096
# file_path = "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/random_models/20240320/random_cmp_output_mp.txt"
# file_path = "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/random_models/20240320/random_cmp_output_mp_b512.txt"
file_path = "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/random_models/20240320/random_cmp_output_mp_b4k_ft500.txt"

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

print(df.shape)
# print the min, max and geomean of the tb_kernel_speedup column
print(df["tb_kernel_speedup"].min())
print(df["tb_kernel_speedup"].max())
# print the geometric mean of the tb_kernel_speedup column
# compute geomean using scipy.stats.gmean
import scipy.stats as stats
print(stats.gmean(df["tb_kernel_speedup"]))
#print(df["tb_kernel_speedup"].prod() ** (1.0 / df.shape[0]))

for depth in range(6, 9):
    # get entries with tree_depth == depth
    df_depth = df[df["tree_depth"] == depth]
    print(df_depth.shape)

    # plot a 3-d graph of num_trees, num_features and tb_kernel_speedup
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 3D scatter plot
    # ax.scatter3D(df_depth["num_trees"], df_depth["num_features"], df_depth["tb_kernel_speedup"], c='red')
    surf = ax.plot_trisurf(df_depth["num_trees"], df_depth["num_features"], df_depth["tb_kernel_speedup"], cmap='viridis')

    ax.set_xlabel('num_trees')
    ax.set_ylabel('num_features')
    ax.set_zlabel('tb_kernel_speedup')

    plt.title(f"Batch Size: {batch_size}, Tree Depth: {depth}")
    plt.colorbar(surf, label='tb_kernel_speedup')
    plt.savefig(f'kernel_speedup_b{batch_size}_depth{depth}_ft500.png')
    # plt.show()
