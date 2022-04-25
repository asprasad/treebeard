from stat import FILE_ATTRIBUTE_NORMAL
import pandas
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import math
import os

current_directory = os.path.dirname(os.path.realpath(__file__))

class SingleBatchSizeResult:
  def __init__(self, batch_size, min_scalar_times, unif_tiling_times) -> None:
    self.batch_size = batch_size
    self.scalar_times = min_scalar_times
    self.unif_tiling_times = unif_tiling_times
    self.unif_tiling_speedups = [i / j for i, j in zip(min_scalar_times, unif_tiling_times)]
    self.unif_tiling_mean_speedup = gmean(self.unif_tiling_speedups)
    self.prob_tiling_times = []
    self.prob_tiling_speedups = []
    self.prob_tiling_mean_speedup = -1
    self.pipelining_times = []
    self.pipelining_speedups_over_unif = []
    self.pipelining_mean_speedup = -1
    self.overall_speedups = []
    self.overall_mean_speedup = -1
  
  def set_prob_tiling_times(self, times):
    self.prob_tiling_times = times
    self.prob_tiling_speedups = [i / j for i, j in zip(self.unif_tiling_times, times)]
    self.prob_tiling_speedups[4] = 1
    self.prob_tiling_mean_speedup = gmean(self.prob_tiling_speedups)
  
  def set_pipelining_times(self, times):
    self.pipelining_times = times
    self.pipelining_speedups_over_unif = [i / j for i, j in zip(self.unif_tiling_times, times)]
    self.pipelining_mean_speedup = gmean(self.pipelining_speedups_over_unif)
    self.best_times = [min(i, j, k) for i, j, k in zip(self.unif_tiling_times, self.prob_tiling_times, self.pipelining_times)]
    self.overall_speedups = [i / j for i, j in zip(self.scalar_times, self.best_times)]
    self.overall_mean_speedup = gmean(self.overall_speedups)

class OptimizationSpeedups:
  def __init__(self, uniform_tiling_speedups, prob_tiling_speedups, pipelining_speedups) -> None:
    self.uniform_tiling_speedups = uniform_tiling_speedups
    self.prob_tiling_speedups = [prob_tiling_speedups[i] * speedup for i, speedup in enumerate(uniform_tiling_speedups)]
    self.pipelining_speedups = [pipelining_speedups[i] * speedup for i, speedup in enumerate(uniform_tiling_speedups)]

class FrameworkSpeedups:
  def __init__(self, xgboost_speedups, treelite_speedups):
    self.xgboost_speedups = xgboost_speedups
    self.treelite_speedups = treelite_speedups

    self.mean_speedup_over_xgboost = gmean(xgboost_speedups)
    self.mean_speedup_over_treelite = gmean(treelite_speedups)
  
  def __str__(self) -> str:
      return "XGBoost Speedup: " + self.mean_speedup_over_xgboost + ", Treelite speedup: " + self.mean_speedup_over_treelite
# print (data_df)


def PlotSpeedups(batch_sizes, title, plot_xlabel, plot_ylabel, plot_name, **speedups_data):
  maxSpeedup = 0
  minSpeedup = float('INF')
  for speed_up, name, marker in zip(speedups_data['speedups'], speedups_data['names'], speedups_data['markers']):
    maxSpeedup = max(maxSpeedup, max(speed_up))
    minSpeedup = min(minSpeedup, min(speed_up))
    plt.plot(batch_sizes, np.array(speed_up), marker=marker, label=name)

  plt.xlabel(plot_xlabel, fontsize=12)
  plt.ylabel(plot_ylabel, fontsize=12)

  plt.yticks(np.arange(math.floor(minSpeedup), math.ceil(maxSpeedup), 0.5))

  plt.legend()
  # plt.title(title)
  plt.grid()

  plt.savefig(plot_name)
  plt.close()

batch_sizes = [64, 128, 256, 512, 1024, 2000]

def ParseFrameworkTimes(data_file_path, model_names, df_headers):
  data_df = pandas.read_csv(data_file_path, header=None)
  data_df.columns = df_headers

  speedups = dict()
  for batch_size in batch_sizes:
    xgboost_speedup = data_df[(data_df.batch_size == batch_size) & (data_df.config == "xgboost speedup")]
    treelite_speedup = data_df[(data_df.batch_size == batch_size) & (data_df.config == "treelite speedup")]

    #print(treelite_speedup.values[0][2:])
    speedups[batch_size] = FrameworkSpeedups(xgboost_speedup.values.tolist()[0][2:], treelite_speedup.values.tolist()[0][2:])
  
  #print(speedups)
  return speedups


def ParseResultsAndGetSpeedups(model_names, data_df, batch_sizes, parallel = False):
    uniform_tiling_speedups = []
    prob_tiling_speedups = []
    pipeline_speedups = []
    batch_size_results = dict()

    batch_results = []
    for batch_size in batch_sizes:
      batch_df = data_df[data_df["batch size"] == batch_size]
      # print (batch_df)
      scalar_df = batch_df[batch_df["tile size"] == 1]
      # For each benchmark, batch size pair find the fastest scalar time
      min_scalar_times = []
      for model in model_names:
        model_times = scalar_df[model]
        min_scalar_times.append(min(model_times))
        # print(model, min_scalar_times)

      # For each benchmark, batch size pair, find the fastest uniform tiling configuration
      uniform_tiling_times = []
      for model in model_names:
        model_times_configs = batch_df[[model, 'config']].values.tolist()
        model_times = [model_time for model_time, config in model_times_configs if 'one_tree' in config and 'prob' not in config]
        uniform_tiling_times.append(min(model_times))

      batch_result = SingleBatchSizeResult(
          batch_size, min_scalar_times, uniform_tiling_times)
      batch_results.append(batch_result)
      
      if not parallel:
        prob_tiling_times = []
        for model in model_names:
          model_times_configs = batch_df[[model, 'config']].values.tolist()
          model_times = [model_time for model_time, config in model_times_configs if 'prob' in config]
          if model_times:
            prob_tiling_times.append(min(model_times))
        
        batch_result.set_prob_tiling_times(prob_tiling_times)
      else:
        batch_result.set_prob_tiling_times(uniform_tiling_times) # hack! -_-

      pipelining_times = []
      for model in model_names:
        model_times_configs = batch_df[[model, 'config']].values.tolist()
        model_times = [model_time for model_time, config in model_times_configs if 'pipelined' in config]
        if model_times:
          pipelining_times.append(min(model_times))
      
      batch_result.set_pipelining_times(pipelining_times)

      uniform_tiling_speedups.append(batch_result.unif_tiling_mean_speedup)
      prob_tiling_speedups.append(batch_result.prob_tiling_mean_speedup)
      pipeline_speedups.append(batch_result.pipelining_mean_speedup)

      print(batch_result.batch_size, batch_result.unif_tiling_mean_speedup, batch_result.prob_tiling_mean_speedup,
            batch_result.pipelining_mean_speedup, batch_result.overall_mean_speedup)
      batch_size_results[batch_size] = batch_result

    if parallel:
      speedups = OptimizationSpeedups(uniform_tiling_speedups, uniform_tiling_speedups, pipeline_speedups)
    else:
      speedups = OptimizationSpeedups(uniform_tiling_speedups, prob_tiling_speedups, pipeline_speedups)

    return speedups, batch_size_results

def PlotSpeedupsForMultiCore(speedups : OptimizationSpeedups, prefix):
    # For each batch size, find geomean speedup and plot
    batch_sizes_np = np.array(batch_sizes)
    plot_xlabel = 'Batch sizes'
    plot_ylabel = 'Speedup'
    PlotSpeedups(
        batch_sizes_np,
        "scalar vs parallel - 16 cores",
        plot_xlabel,
        plot_ylabel,
        prefix + '_parallel_speedups_vs_scalar.png',
        speedups=[speedups.pipelining_speedups],
        names=['speedup from parallelization'],
        markers = ['o', 'D'])

def PlotSpeedupsForSingleCore(speedups, prefix):
    # For each batch size, find geomean speedup and plot
    batch_sizes_np = np.array(batch_sizes)
    plot_xlabel = 'Batch sizes'
    plot_ylabel = 'Speedup'

    PlotSpeedups(
        batch_sizes_np,
        "scalar vs uniform tiling vs {uniform + prob.} tiling",
        plot_xlabel,
        plot_ylabel,
        prefix + '_mean_speedups_prob_tiling_vs_scalar.png',
        speedups=[speedups.uniform_tiling_speedups,
                  speedups.prob_tiling_speedups],
        names=['Basic tiling', 'Probability based tiling'],
        markers = ['o', 'D'])

    PlotSpeedups(
        batch_sizes_np,
        "scalar vs uniform tiling vs {uniform + pipelining}",
        plot_xlabel,
        plot_ylabel,
        prefix + '_mean_speedups_pipelining_vs_scalar.png',
        speedups=[speedups.uniform_tiling_speedups,
                  speedups.pipelining_speedups],
        names=['Basic tiling', 'Basic tiling + Interleaving + Peeling and Unrolling'],
        markers = ['o', 'D']
    )

def PlotLineGraphForMultipleCpus(batch_results_intel, batch_results_amd, prefix):
    # For each batch size, find geomean speedup and plot
    batch_sizes_np = np.array(batch_sizes)
    plot_xlabel = 'Batch sizes'
    plot_ylabel = 'Speedup'

    intel_speedups = []
    amd_speedups = []

    for batch_size in batch_sizes:
      intel_speedups.append(batch_results_intel[batch_size].overall_mean_speedup)
      amd_speedups.append(batch_results_amd[batch_size].overall_mean_speedup)

    PlotSpeedups(
        batch_sizes_np,
        "scalar vs uniform tiling vs {uniform + prob.} tiling",
        plot_xlabel,
        plot_ylabel,
        prefix + '_overall_mean_speedups_intel_amd.png',
        speedups=[intel_speedups, amd_speedups],
        names=['Intel', 'AMD'],
        markers = ['o', 'D'])


def PlotSpeedupsOverOtherFrameworks(speedups, prefix):
    # For each batch size, find geomean speedup and plot
    batch_sizes_np = np.array(batch_sizes)
    plot_xlabel = 'Batch sizes'
    plot_ylabel = 'Speedup'

    speedups_over_xgboost = [speedups[batch_size].mean_speedup_over_xgboost for batch_size in batch_sizes]
    speedups_over_treelite = [speedups[batch_size].mean_speedup_over_treelite for batch_size in batch_sizes]

    PlotSpeedups(
        batch_sizes_np,
        "treebeard vs {xgboost, treelite} - " + prefix,
        plot_xlabel,
        plot_ylabel,
        prefix + '_mean_speedups_treebeard_vs_xgboost_treelite.png',
        speedups=[speedups_over_xgboost, speedups_over_treelite],
        names=['vs xgboost', 'vs treelite'],
        markers = ['o', 'D'])

def PlotBarGraphForBenchmarkSpeedupsOverOtherFrameworks(speedups, model_names: List[str], batch_size, prefix):
    xgboost_speedups = speedups[batch_size].xgboost_speedups + [speedups[batch_size].mean_speedup_over_xgboost]
    treelite_speedups = speedups[batch_size].treelite_speedups + [speedups[batch_size].mean_speedup_over_treelite]

    file_name = prefix + "_speedup_per_benchmark_xgoost_treelite.png"
    PlotBarGraph(
      model_names + ['mean speedup'],
      xgboost_speedups,
      treelite_speedups,
      "xgboost",
      "treelite",
      file_name,
      "vs xgboost, treelite", 
      batch_size)

def PlotBarGraph(model_names, first_speedups, second_speedups, first_speedup_name, second_speedup_name, file_name, batch_size, prefix):
  x_axis = np.arange(len(model_names))# np.array(model_names)

  plt.figure(figsize=(16, 9))
  width = 0.2

  plt.bar(x_axis - 0.1, np.array(first_speedups), width=width, hatch='///')
  plt.bar(x_axis + 0.1, np.array(second_speedups), width=width, hatch='...')

  plt.xticks(x_axis, model_names)

  plt.xlabel("Benchmark", fontsize=12)
  plt.ylabel("Speedup", fontsize=12)
  plt.legend([first_speedup_name, second_speedup_name])
  # plt.title(title + ". Tile size = " + str(tile_size))

  plt.savefig(file_name)
  plt.close()

def PlotBarGraphForScalarSpeedups(batch_size_results, model_names: List[str], batch_size, prefix):
  unif_tiling_speedups = []
  prob_tiling_speedups = []
  pipeline_speedups = []

  for i, model_name in enumerate(model_names):
    cur_result = batch_size_results[batch_size]
    unif_tiling_speedups.append(cur_result.unif_tiling_speedups[i])
    prob_tiling_speedups.append(cur_result.prob_tiling_speedups[i] * cur_result.unif_tiling_speedups[i])
    pipeline_speedups.append(cur_result.pipelining_speedups_over_unif[i] * cur_result.unif_tiling_speedups[i])
  
  file_name = prefix + "_max_speedup_unif_prob_tiling.png"
  PlotBarGraph(
    model_names + ['mean speedup'],
    unif_tiling_speedups + [gmean(unif_tiling_speedups)],
    prob_tiling_speedups + [gmean(prob_tiling_speedups)],
    "Basic tiling",
    "Probability based tiling",
    file_name,
    "scalar vs uniform tiling vs {uniform + probabilistic} tiling", 
    batch_size)

  file_name = prefix + "_max_speedup_unif_tiling_pipelining.png"
  PlotBarGraph(
    model_names + ['mean speedup'],
    unif_tiling_speedups + [gmean(unif_tiling_speedups)],
    pipeline_speedups + [gmean(pipeline_speedups)],
    "Basic tiling",
    "Basic tiling + Interleaving + Peeling and Unrolling",
    file_name,
    "scalar vs uniform tiling vs uniform tiling + pipelining",
    batch_size)

def PlotBarGraphForParallelSpeedups(batch_size_results, model_names: List[str], tile_size, prefix):
  unif_tiling_speedups = []
  parallel_speedups = []

  for i, model_name in enumerate(model_names):
    cur_result = batch_size_results[tile_size]
    unif_tiling_speedups.append(cur_result.unif_tiling_speedups[i])
    parallel_speedups.append(cur_result.pipelining_speedups_over_unif[i] * cur_result.unif_tiling_speedups[i])
  
  file_name = prefix + "_max_speedup_unif_tiling_pipelining.png"
  x_axis = np.arange(len(model_names))# np.array(model_names)

  plt.figure(figsize=(16, 9))
  width = 0.2

  plt.bar(x_axis, np.array(parallel_speedups), width=width)
  plt.xticks(x_axis, model_names)

  plt.xlabel("Benchmark", fontsize=12)
  plt.ylabel("Speedup", fontsize=12)
  plt.legend(["speedup from parallelization"])
  # plt.title("Speedup of parallel execution over base scalar version. Tile size = " + str(tile_size))

  plt.savefig(file_name)
  plt.close()

def PlotBarGraphForIntelAndAMDSpeedups(batch_size_results_intel, batch_size_results_amd, model_names: List[str], batch_size, prefix):
  overall_speedups_intel = []
  overall_speedups_amd = []
  time_for_intel = []
  time_for_amd = []

  cur_result_intel = batch_size_results_intel[batch_size]
  cur_result_amd = batch_size_results_amd[batch_size]
  for i, model_name in enumerate(model_names):
    overall_speedups_intel.append(cur_result_intel.overall_speedups[i])
    overall_speedups_amd.append(cur_result_amd.overall_speedups[i])
    time_for_intel.append(1 / cur_result_intel.best_times[i])
    time_for_amd.append(1 /cur_result_amd.best_times[i])
  
  file_name = prefix + "_overall_speedup_benchmark_intel_amd.png"
  PlotBarGraph(
    model_names + ['geomean'],
    overall_speedups_intel + [cur_result_intel.overall_mean_speedup],
    overall_speedups_amd + [cur_result_amd.overall_mean_speedup],
    "Intel",
    "AMD",
    file_name,
    "Intel vs AMD", 
    batch_size)

def ParseOptimizationSpeedups(data_file_path, model_names, df_headers, parallel = False):
    data_df = pandas.read_csv(data_file_path, header=None)
    data_df.columns = df_headers

    speedups = ParseResultsAndGetSpeedups(model_names, data_df, batch_sizes, parallel)
    return speedups

model_names = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "letters", "higgs", "year"]
df_headers = ["config", "type", "batch size", "tile size"] + model_names
data_file_path = current_directory + "/holmes/multiple_batchSizes_results_final.txt"
speedups, batch_size_results_intel = ParseOptimizationSpeedups(data_file_path, model_names, df_headers)
PlotSpeedupsForSingleCore(speedups, "scalar_Intel")
PlotBarGraphForScalarSpeedups(batch_size_results_intel, model_names, 1024, "scalar_Intel")

data_file_path = current_directory + "/bhagheera/multiple_batchSizes_results_serial_final.txt"
speedups, batch_size_results_amd = ParseOptimizationSpeedups(data_file_path, model_names, df_headers)
PlotSpeedupsForSingleCore(speedups, "scalar_AMD")
PlotBarGraphForScalarSpeedups(batch_size_results_amd, model_names, 1024, "scalar_AMD")

PlotLineGraphForMultipleCpus(batch_size_results_intel, batch_size_results_amd, 'scalar')
PlotBarGraphForIntelAndAMDSpeedups(batch_size_results_intel, batch_size_results_amd, model_names, 1024, 'scalar')

data_file_path = current_directory + "/holmes/multiple_batchSizes_results_parallel_16cores_final.txt"
speedups, batch_size_results_intel = ParseOptimizationSpeedups(data_file_path, model_names, df_headers, True)
PlotSpeedupsForMultiCore(speedups, 'parallel_Intel')
PlotBarGraphForParallelSpeedups(batch_size_results_intel, model_names, 1024, "parallel_Intel")

data_file_path = current_directory + "/bhagheera/multiple_batchSizes_results_parallel_final.txt"
speedups, batch_size_results_amd = ParseOptimizationSpeedups(data_file_path, model_names, df_headers, True)
PlotSpeedupsForMultiCore(speedups, 'parallel_AMD')
PlotBarGraphForParallelSpeedups(batch_size_results_amd, model_names, 1024, "parallel_AMD")

PlotLineGraphForMultipleCpus(batch_size_results_intel, batch_size_results_amd, 'parallel')
# PlotBarGraphForIntelAndAMDSpeedups(batch_size_results_intel, batch_size_results_amd, model_names, 1024, 'parallel')

df_headers = ["config", "batch_size"] + model_names
data_file_path = current_directory + "/holmes/xgboost_treelite_compare_serial.txt"
speedups = ParseFrameworkTimes(data_file_path, model_names, df_headers)
PlotSpeedupsOverOtherFrameworks(speedups, "scalar_Intel")
PlotBarGraphForBenchmarkSpeedupsOverOtherFrameworks(speedups, model_names, 1024, "serial")

df_headers = ["config", "batch_size"] + model_names
data_file_path = current_directory + "/holmes/xgboost_treelite_compare_parallel.txt"
speedups = ParseFrameworkTimes(data_file_path, model_names, df_headers)
PlotSpeedupsOverOtherFrameworks(speedups, "parallel_Intel")
PlotBarGraphForBenchmarkSpeedupsOverOtherFrameworks(speedups, model_names, 1024, "parallel")