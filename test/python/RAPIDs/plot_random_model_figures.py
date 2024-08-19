import os
import matplotlib.pyplot as plt

def plot_random_model_graph(batch_size, result_file_path, graph_file_path, depth):
    # read the file line by line and filter out lines that don't start with "/"
    with open(result_file_path, "r") as file:
        lines = file.readlines()
        lines = [line for line in lines if line.startswith("/")]

    # split each line by the space char
    lines = [line.split(" ") for line in lines]

    # construct a pandas dataframe from the list of lists
    import pandas as pd
    cols = ["model_path", "batch_size", "fil_kernel_time", "tb_kernel_time", "tb_kernel_speedup"]
    df = pd.DataFrame(lines, columns=cols)

    # cast the tb_kernel_speedup and total_speedup column to float
    df["tb_kernel_speedup"] = df["tb_kernel_speedup"].astype(float)

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

    # compute geomean using scipy.stats.gmean
    import scipy.stats as stats

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
    plt.savefig(graph_file_path)
    # plt.show()

if __name__ == "__main__":
    results_file_path_512 = os.path.join(os.path.dirname(__file__), "random_models_512.txt")
    graph_file_path = os.path.join(os.path.dirname(__file__), "Figure8a.png")
    plot_random_model_graph(512, results_file_path_512, graph_file_path, 8)

    results_file_path_4k = os.path.join(os.path.dirname(__file__), "random_models_4k.txt")
    graph_file_path = os.path.join(os.path.dirname(__file__), "Figure8b.png")
    plot_random_model_graph(4096, results_file_path_4k, graph_file_path, 6)
