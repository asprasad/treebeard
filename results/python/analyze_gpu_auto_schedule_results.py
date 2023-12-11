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

results_file = "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/exploration/gpu_auto-sched_thedeep_600reps_tree-pipelining_transferOpt.txt"
# results_file = "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/exploration/gpu_results_thedeep_try2.txt"
# results_file = "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/holmes/gpu/exploration/gpu_results_holmes.txt"
# read results file into pandas dataframe
col_names=["model", "representation", "batch_size", "rowsPerTB", "rowsPerT", "numTreeThreads", "treeInterleaveDepth", "cache_rows", "cache_trees", "unroll", "total", "kernel"]
df = pd.read_csv(results_file)
# print(df)

# filter out all the entries where total = -1
df = df[df.total != -1]

# get all the unique benchmark naems from df
benchmark_names = df.model.unique()
# print(benchmark_names)
batch_sizes = df.batch_size.unique()

# Get the best schedule for each benchmark for each batch size in terms of total time. 
# Also print the schedule name and the kernel
df_best_total_time = df.loc[df.groupby(['model', 'batch_size'])['total'].idxmin()]
# print(df_best_total_time)

# do the same for kernel
df_best_kernel_time = df.loc[df.groupby(['model', 'batch_size'])['kernel'].idxmin()]
# print(df_best_kernel_time)

# Group the data by benchmark and batchSize. Sort each group on the total
df_sorted_by_total_time = df.sort_values(['model', 'batch_size', 'total'])
df_sorted_by_kernel_time = df.sort_values(['model', 'batch_size', 'kernel'])

# compute_rank_in_other_table(df_sorted_by_kernel_time, df_best_total_time, benchmark_names, batch_sizes)
# compute_rank_in_other_table(df_sorted_by_total_time, df_best_kernel_time, benchmark_names, batch_sizes)

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000) 

# for each benchmark and batchSize, get the percentage difference between the best schedule and the 10th best schedule and the percentage difference 
# between the best schedule and the 25th best schedule
# total_time_diff_df = df_sorted_by_total_time.groupby(['model', 'batch_size'])['total'].apply(time_diff_computation)
total_time_diff_df = df_sorted_by_total_time.groupby(['model', 'batch_size']).agg(five_diff=('total', lambda x: (x.iloc[5] - x.iloc[0])/x.iloc[0]),
                                                                                     ten_diff=('total', lambda x: (x.iloc[9] - x.iloc[0])/x.iloc[0]),
                                                                                     twentyfive_diff=('total', lambda x: (x.iloc[25] - x.iloc[0])/x.iloc[0]))
# print the full dataframe
# print(total_time_diff_df)

# get all the benchmark names from total_time_diff_df
# print(total_time_diff_df.index.get_level_values(0).unique())

for benchmark in benchmark_names:
    for batch_size in batch_sizes:
        benchmark_group = total_time_diff_df.loc[benchmark]
        bencmark_batch_diffs = benchmark_group.loc[batch_size]
        df_best_total_time.loc[(df_best_total_time['model'] == benchmark) & (df_best_total_time['batch_size'] == batch_size), 'five_diff'] = bencmark_batch_diffs.loc['five_diff']
        df_best_total_time.loc[(df_best_total_time['model'] == benchmark) & (df_best_total_time['batch_size'] == batch_size), 'ten_diff'] = bencmark_batch_diffs.loc['ten_diff']
        df_best_total_time.loc[(df_best_total_time['model'] == benchmark) & (df_best_total_time['batch_size'] == batch_size), 'twentyfive_diff'] = bencmark_batch_diffs.loc['twentyfive_diff']

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
        df_best_total_time.loc[(df_best_total_time['model'] == benchmark) & (df_best_total_time['batch_size'] == batch_size), 'five_diff_kernel'] = bencmark_batch_diffs.loc['five_diff']
        df_best_total_time.loc[(df_best_total_time['model'] == benchmark) & (df_best_total_time['batch_size'] == batch_size), 'ten_diff_kernel'] = bencmark_batch_diffs.loc['ten_diff']
        df_best_total_time.loc[(df_best_total_time['model'] == benchmark) & (df_best_total_time['batch_size'] == batch_size), 'twentyfive_diff_kernel'] = bencmark_batch_diffs.loc['twentyfive_diff']

print(df_best_total_time)
