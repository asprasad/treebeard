from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def time_diff_computation(x):
    five_diff = (x.iloc[9] - x.iloc[0])/x.iloc[0]
    ten_diff = (x.iloc[9] - x.iloc[0])/x.iloc[0]
    twentyfive_diff = (x.iloc[25] - x.iloc[0])/x.iloc[0]
    return pd.Series({ 'five_diff': five_diff, 'ten_diff': ten_diff, 'twentyfive_diff': twentyfive_diff})

def compute_rank_in_other_table(df_sorted, df_best, benchmark_names, batch_sizes):
    for benchmark in benchmark_names:
        for batch_size in batch_sizes:
            results_df = df_sorted[(df_sorted['model'] == benchmark) & (df_sorted['batch_size'] == batch_size)]
            best_schedule = df_best[(df_best['model'] == benchmark) & (df_best['batch_size'] == batch_size)]
            assert (best_schedule.shape[0] == 1)
            best_schedule_name = best_schedule.iloc[0]['schedule']
            best_schedule_rep = best_schedule.iloc[0]['representation']
            # reset the indices for results_df
            results_df = results_df.reset_index(drop=True)
            # find the index of the entry with schedule=best_schedule_name and representation=best_schedule_rep in results_df
            best_schedule_index = results_df.index[(results_df['schedule'] == best_schedule_name) & (results_df['representation'] == best_schedule_rep)].tolist()[0]
            # print(best_schedule_index)
            # add best_schedule_index to df_best_total_time as a new column "kernelRank"
            df_best.loc[(df_best['model'] == benchmark) & (df_best['batch_size'] == batch_size), 'kernelRank'] = best_schedule_index

def plot_normalized_time_histogram(df_sorted, df_best, col_name):
    # for every row in df_sorted, find the row in df_best with the same benchmark and batch_size and 
    # divide the total time of the row in df_sorted by the total time of the row in df_best
    # plot a histogram of the normalized times
    for index, row in df_sorted.iterrows():
        benchmark = row['model']
        batch_size = row['batch_size']
        best_schedule = df_best[(df_best['model'] == benchmark) & (df_best['batch_size'] == batch_size)]
        assert (best_schedule.shape[0] == 1)
        best_schedule_total_time = best_schedule.iloc[0][col_name]
        df_sorted.loc[index, 'normalized_'+col_name] = row[col_name]/best_schedule_total_time
    # print(df_sorted)
    # plot a histogram of the normalized times and save it to a file
    df_sorted.hist(column='normalized_'+col_name, bins=100)
    plt.savefig(f'normalized_{col_name}_histogram.png')

    # plot a histogram of values with normalized times < 5
    df_sorted[df_sorted['normalized_'+col_name] < 5].hist(column='normalized_'+col_name, bins=500)
    plt.savefig(f'normalized_{col_name}_histogram_lt5.png')

def write_best_config_maps_to_file(df_best_total_time, benchmark_names, batch_sizes, file_name):
    # write the best configs to a file
    batch_sizes.sort()
    f = open(file_name, "w")
    for benchmark in benchmark_names:
        map_name = f"{benchmark}_best_configs"
        if benchmark == "airline-ohe":
            map_name = "airline_ohe_best_configs"
        f.write(f"{map_name} = {{\n")
        for batch_size in batch_sizes:
            best_schedule = df_best_total_time[(df_best_total_time['model'] == benchmark) & (df_best_total_time['batch_size'] == batch_size)]
            assert (best_schedule.shape[0] == 1)
            # 256: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 50, True, False, True, 2), "gpu_array"),
            rows_per_tb = best_schedule.iloc[0]['rowsPerTB']
            rows_per_t = best_schedule.iloc[0]['rowsPerT']
            num_tree_threads = best_schedule.iloc[0]['numTreeThreads']
            tree_interleave_depth = best_schedule.iloc[0]['treeInterleaveDepth']
            cache_rows = best_schedule.iloc[0]['cache_rows']
            cache_trees = best_schedule.iloc[0]['cache_trees']
            unroll = best_schedule.iloc[0]['unroll']
            representation = best_schedule.iloc[0]['representation']
            f.write(f"  {batch_size}: (treebeard.GPUAutoScheduleOptions.Construct({rows_per_tb}, {rows_per_t}, {num_tree_threads}, {tree_interleave_depth}, {cache_rows}, {cache_trees}, {unroll}), \"{representation}\"),\n")
        f.write("}\n")
    
    benchmark_to_config_map_map_name = "benchmark_to_config_map_map";
    f.write(f"{benchmark_to_config_map_map_name} = {{\n")
    for benchmark in benchmark_names:
        map_name = f"{benchmark}_best_configs"
        if benchmark == "airline-ohe":
            map_name = "airline_ohe_best_configs"
        f.write(f"  \"{benchmark}\": {map_name},\n")
    f.write("}\n")
    f.close()

def compare_across_batch_sizes(df_best_kernel_time, df_sorted_by_kernel_time, batch_sizes, benchmarks):
    # for every benchmark and batch size, get the best schedule from df_best_kernel_time 
    for batch_size in batch_sizes:
        for benchmark in benchmarks:
            best_schedule = df_best_kernel_time[(df_best_kernel_time['model'] == benchmark) & (df_best_kernel_time['batch_size'] == batch_size)]
            # print(best_schedule)
            assert (best_schedule.shape[0] == 1)
            best_schedule_time = best_schedule.iloc[0]['kernel']
            best_schedule_rep = best_schedule.iloc[0]['representation']
            for cmp_batch_size in batch_sizes:
                entry_at_batch_size = df_sorted_by_kernel_time[(df_sorted_by_kernel_time['model'] == benchmark) & 
                                                               (df_sorted_by_kernel_time['batch_size'] == cmp_batch_size) &
                                                               (df_sorted_by_kernel_time['representation'] == best_schedule_rep) &
                                                               (df_sorted_by_kernel_time['rowsPerTB'] == best_schedule.iloc[0]['rowsPerTB']) &
                                                               (df_sorted_by_kernel_time['rowsPerT'] == best_schedule.iloc[0]['rowsPerT']) &
                                                               (df_sorted_by_kernel_time['numTreeThreads'] == best_schedule.iloc[0]['numTreeThreads']) &
                                                               (df_sorted_by_kernel_time['treeInterleaveDepth'] == best_schedule.iloc[0]['treeInterleaveDepth']) &
                                                               (df_sorted_by_kernel_time['cache_rows'] == best_schedule.iloc[0]['cache_rows']) &
                                                               (df_sorted_by_kernel_time['cache_trees'] == best_schedule.iloc[0]['cache_trees']) &
                                                               (df_sorted_by_kernel_time['unroll'] == best_schedule.iloc[0]['unroll']) &
                                                               (df_sorted_by_kernel_time['sharedReduce'] == best_schedule.iloc[0]['sharedReduce'])]
                # print(entry_at_batch_size)
                if (entry_at_batch_size.shape[0] != 1):
                    continue
                cmp_time = entry_at_batch_size.iloc[0]['kernel']
                best_time_at_cmp_batch_size = df_best_kernel_time[(df_best_kernel_time['model'] == benchmark) & (df_best_kernel_time['batch_size'] == cmp_batch_size)].iloc[0]['kernel']
                degradation = (cmp_time - best_time_at_cmp_batch_size)/best_time_at_cmp_batch_size
                print(f"Degradation for {benchmark} at batch size {batch_size} compared to batch size {cmp_batch_size}: {degradation}")
# results_file = "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/exploration/gpu_auto-schedule-results_thedeep_600reps_tree-pipelining_noletters_plus1TreeThread.txt"
# results_file = "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/exploration/gpu_auto-schedule-results_thedeep_200reps_tree-pipelining_letters_plus1TreeThread.txt"
# results_file = "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/exploration/gpu_auto-schedule-results_thedeep_600reps_tree-pipelining_smallBatch_plus1TreeThread.txt"

# result_files = ["/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/exploration/gpu_auto-schedule-results_thedeep_600reps_tree-pipelining_noletters_plus1TreeThread.txt",
# "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/exploration/gpu_auto-schedule-results_thedeep_200reps_tree-pipelining_letters_plus1TreeThread.txt",
# "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/exploration/gpu_auto-schedule-results_thedeep_600reps_tree-pipelining_smallBatch_plus1TreeThread.txt"]

result_files = [
   "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/exploration/20240229/exploration_b4k-16k_stderr_filtered.txt",
   "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/exploration/20240229/exploration_b4k-16k_stderr_letters_filtered.txt",
   "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/exploration/20240229/exploration_b256_stderr_filtered.txt",
   "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/exploration/20240229/exploration_b512-2k_stderr_filtered.txt",
   "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/exploration/20240229/exploration_b512-2k_stderr_letters_filtered.txt"
]

col_names=["model", "representation", "batch_size", "rowsPerTB", 
           "rowsPerT", "numTreeThreads", "treeInterleaveDepth", 
           "cache_rows", "cache_trees", "unroll", "sharedReduce",
           "total", "kernel"]

dataframes = [pd.read_csv(f, names=col_names) for f in result_files]

# results_file = "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/exploration/gpu_auto-sched_thedeep_600reps_tree-pipelining_transferOpt.txt"
# results_file = "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/exploration/gpu_auto-schedule-results_thedeep_200reps_tree-pipelining_lettersOnly.txt"
# results_file = "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/exploration/gpu_auto-schedule-results_thedeep_1kreps_tree-pipelining_smallbatch.txt"

# results_file = "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/holmes/gpu/exploration/gpu_results_holmes.txt"
# read results file into pandas dataframe
df = pd.concat(dataframes, ignore_index=True)
# print(df)

# filter out all the entries where total = -1
df = df[df.total != -1]

# get all the unique benchmark naems from df
benchmark_names = df.model.unique()
# print(benchmark_names)
batch_sizes = df.batch_size.unique()

# Get the best schedule for each benchmark for each batch size in terms of total time. 
# Also print the schedule name and the kernel
# df_best_total_time = df.loc[df.groupby(['model', 'batch_size'])['total'].idxmin()]
# print(df_best_total_time)

# do the same for kernel
df_best_kernel_time = df.loc[df.groupby(['model', 'batch_size'])['kernel'].idxmin()]
# print(df_best_kernel_time)

# Group the data by benchmark and batchSize. Sort each group on the total
# df_sorted_by_total_time = df.sort_values(['model', 'batch_size', 'total'])
df_sorted_by_kernel_time = df.sort_values(['model', 'batch_size', 'kernel'])

# compute_rank_in_other_table(df_sorted_by_kernel_time, df_best_total_time, benchmark_names, batch_sizes)
# compute_rank_in_other_table(df_sorted_by_total_time, df_best_kernel_time, benchmark_names, batch_sizes)

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000) 

# for each benchmark and batchSize, get the percentage difference between the best schedule and the 10th best schedule and the percentage difference 
# between the best schedule and the 25th best schedule
# total_time_diff_df = df_sorted_by_total_time.groupby(['model', 'batch_size'])['total'].apply(time_diff_computation)
# total_time_diff_df = df_sorted_by_total_time.groupby(['model', 'batch_size']).agg(five_diff=('total', lambda x: (x.iloc[5] - x.iloc[0])/x.iloc[0]),
#                                                                                      ten_diff=('total', lambda x: (x.iloc[9] - x.iloc[0])/x.iloc[0]),
#                                                                                      twentyfive_diff=('total', lambda x: (x.iloc[25] - x.iloc[0])/x.iloc[0]))
# print the full dataframe
# print(total_time_diff_df)

# get all the benchmark names from total_time_diff_df
# print(total_time_diff_df.index.get_level_values(0).unique())

# for benchmark in benchmark_names:
#     for batch_size in batch_sizes:
#         benchmark_group = total_time_diff_df.loc[benchmark]
#         bencmark_batch_diffs = benchmark_group.loc[batch_size]
#         df_best_total_time.loc[(df_best_total_time['model'] == benchmark) & (df_best_total_time['batch_size'] == batch_size), 'five_diff'] = bencmark_batch_diffs.loc['five_diff']
#         df_best_total_time.loc[(df_best_total_time['model'] == benchmark) & (df_best_total_time['batch_size'] == batch_size), 'ten_diff'] = bencmark_batch_diffs.loc['ten_diff']
#         df_best_total_time.loc[(df_best_total_time['model'] == benchmark) & (df_best_total_time['batch_size'] == batch_size), 'twentyfive_diff'] = bencmark_batch_diffs.loc['twentyfive_diff']

# print(df_best_kernel_time)

# print(total_time_diff_df["abalone_xgb_model_save.json"])

# do the same with kernel times
# kernel_time_diff_df = df_sorted_by_kernel_time.groupby(['model', 'batch_size'])['kernel'].apply(time_diff_computation)
kernel_time_diff_df = df_sorted_by_kernel_time.groupby(['model', 'batch_size']).agg(five_diff=('kernel', lambda x: (x.iloc[5] - x.iloc[0])/x.iloc[0]),
                                                                                     ten_diff=('kernel', lambda x: (x.iloc[9] - x.iloc[0])/x.iloc[0]),
                                                                                     twentyfive_diff=('kernel', lambda x: (x.iloc[25] - x.iloc[0])/x.iloc[0]))
# print(kernel_time_diff_df)

for benchmark in benchmark_names:
    for batch_size in batch_sizes:
        benchmark_group = kernel_time_diff_df.loc[benchmark]
        bencmark_batch_diffs = benchmark_group.loc[batch_size]
        df_best_kernel_time.loc[(df_best_kernel_time['model'] == benchmark) & (df_best_kernel_time['batch_size'] == batch_size), 'five_diff_kernel'] = bencmark_batch_diffs.loc['five_diff']
        df_best_kernel_time.loc[(df_best_kernel_time['model'] == benchmark) & (df_best_kernel_time['batch_size'] == batch_size), 'ten_diff_kernel'] = bencmark_batch_diffs.loc['ten_diff']
        df_best_kernel_time.loc[(df_best_kernel_time['model'] == benchmark) & (df_best_kernel_time['batch_size'] == batch_size), 'twentyfive_diff_kernel'] = bencmark_batch_diffs.loc['twentyfive_diff']

print(df_best_kernel_time)

compare_across_batch_sizes(df_best_kernel_time, df_sorted_by_kernel_time, batch_sizes, benchmark_names)
# plot_normalized_time_histogram(df_sorted_by_kernel_time, df_best_kernel_time, 'kernel')
# write_best_config_maps_to_file(df_best_total_time, benchmark_names, batch_sizes, "best_configs.txt")