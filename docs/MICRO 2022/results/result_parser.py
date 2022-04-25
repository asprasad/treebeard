import pandas
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import math
import os

CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
GRAPH_BATCH_SIZE = 1024
BATCH_SIZES = [64, 128, 256, 512, 1024, 2000]
CORE_COUNTS = [2, 4, 8, 16]

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
  def __init__(self, xgboost_times, treelite_times, xgboost_speedups, treelite_speedups):
    # the times are in seconds. Convert to microseconds
    self.xgboost_times = [time * 10 ** 6 for time in xgboost_times]
    self.treelite_times = [time * 10 ** 6 for time in treelite_times]

    self.xgboost_speedups = xgboost_speedups
    self.treelite_speedups = treelite_speedups

    self.mean_speedup_over_xgboost = gmean(xgboost_speedups)
    self.mean_speedup_over_treelite = gmean(treelite_speedups)
  
  def __str__(self) -> str:
      return "XGBoost Speedup: " + self.mean_speedup_over_xgboost + ", Treelite speedup: " + self.mean_speedup_over_treelite
# print (data_df)

class ParallelSpeedupForCore:
  def __init__(self, scalar_times, parallel_times):
    self.scalar_times = scalar_times
    self.parallel_times = parallel_times
    self.speedups = [scalar/parallel for scalar, parallel in zip(scalar_times, parallel_times)]
    self.mean_speedup = gmean(self.speedups)

def ParseFrameworkTimes(data_file_path, model_names, df_headers):
  data_df = pandas.read_csv(data_file_path, header=None)
  data_df.columns = df_headers

  speedups = dict()
  for batch_size in BATCH_SIZES:
    xgboost_times = data_df[(data_df.batch_size == batch_size) & (data_df.config == "xgboost")]
    treelite_times = data_df[(data_df.batch_size == batch_size) & (data_df.config == "treelite")]

    xgboost_speedup = data_df[(data_df.batch_size == batch_size) & (data_df.config == "xgboost speedup")]
    treelite_speedup = data_df[(data_df.batch_size == batch_size) & (data_df.config == "treelite speedup")]

    #print(treelite_speedup.values[0][2:])
    speedups[batch_size] = FrameworkSpeedups(
      xgboost_times.values.tolist()[0][2:],
      treelite_times.values.tolist()[0][2:],
      xgboost_speedup.values.tolist()[0][2:],
      treelite_speedup.values.tolist()[0][2:])
  
  #print(speedups)
  return speedups

def ParseParallelScalingResultsAndGetSpeedups(model_names, data_df):
  batch_core_scaling = {}

  for batch_size in BATCH_SIZES:
    batch_df = data_df[data_df["batch size"] == batch_size]
    scalar_batch_df = batch_df[batch_df.ncores == 1]

    scalar_times = []
    parallel_times = []

    for model in model_names:
      model_times = scalar_batch_df[model]
      scalar_times.append(min(model_times))
    
    core_scaling = {}
    for core in CORE_COUNTS:
      min_core_times = []
      core_df = batch_df[batch_df.ncores == core]
      for model in model_names:
        model_times = core_df[model].values.tolist()
        min_core_times.append(min(model_times))
      
      core_scaling[core] = ParallelSpeedupForCore(scalar_times, min_core_times)
    
    batch_core_scaling[batch_size] = core_scaling
  
  return batch_core_scaling

def ParseResultsAndGetSpeedups(model_names, data_df, parallel = False):
    uniform_tiling_speedups = []
    prob_tiling_speedups = []
    pipeline_speedups = []
    batch_size_results = dict()

    batch_results = []
    for batch_size in BATCH_SIZES:
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

def ParseOptimizationSpeedups(data_file_path, model_names, df_headers, parallel = False):
    data_df = pandas.read_csv(data_file_path, header=None)
    data_df.columns = df_headers

    speedups = ParseResultsAndGetSpeedups(model_names, data_df, parallel)
    return speedups

def ParseParallelScalingSpeedups(data_file_path, model_names, df_headers):
    data_df = pandas.read_csv(data_file_path, header=None)
    data_df.columns = df_headers

    batch_core_scaling = ParseParallelScalingResultsAndGetSpeedups(model_names, data_df)
    return batch_core_scaling


def PlotLineGraph(batch_sizes, title, plot_xlabel, plot_ylabel, plot_name, step_size=0.5, **data):
  maxSpeedup = 0
  # minSpeedup = float('INF')
  minSpeedup = 0
  for speed_up, name, marker in zip(data['speedups'], data['names'], data['markers']):
    maxSpeedup = max(maxSpeedup, max(speed_up))
    # minSpeedup = min(minSpeedup, min(speed_up))
    plt.plot(batch_sizes, np.array(speed_up), marker=marker, label=name)

  plt.xlabel(plot_xlabel, fontsize=12)
  plt.ylabel(plot_ylabel, fontsize=12)

  plt.yticks(np.arange(math.floor(minSpeedup), math.ceil(maxSpeedup) + 0.5, step_size))

  plt.legend()
  # plt.title(title)
  plt.grid()

  plt.savefig(plot_name)
  plt.close()


def PlotLineGraphForMultiCoreSpeedup(speedups : OptimizationSpeedups, prefix):
    # For each batch size, find geomean speedup and plot
    batch_sizes_np = np.array(BATCH_SIZES)
    plot_xlabel = 'Batch sizes'
    plot_ylabel = 'Speedup'
    PlotLineGraph(
        batch_sizes_np,
        "scalar vs parallel - 16 cores",
        plot_xlabel,
        plot_ylabel,
        prefix + '_parallel_speedups_vs_scalar.png',
        speedups=[speedups.pipelining_speedups],
        names=['speedup from parallelization'],
        markers = ['o', 'D'])

def PlotLineGraphForCoreScaling(batch_core_scaling_intel, batch_core_scaling_amd, prefix, batch_size):
    core_counts_np = np.array(CORE_COUNTS)
    plot_xlabel = 'Number of Cores'
    plot_ylabel = 'Speedup'
    
    core_speedup_intel = [batch_core_scaling_intel[batch_size][core].mean_speedup for core in CORE_COUNTS]
    core_speedup_amd = [batch_core_scaling_amd[batch_size][core].mean_speedup for core in CORE_COUNTS]

    PlotLineGraph(
        core_counts_np,
        "Variation of speedup with number of cores",
        plot_xlabel,
        plot_ylabel,
        prefix + '_core_scaling_intel_amd.png',
        step_size=1,
        speedups=[core_speedup_intel, core_speedup_amd],
        names=['Intel RocketLake', 'AMD Ryzen 7'],
        markers = ['o', 'D'])


def PlotLineGraphForScalarSpeedup(speedups, prefix):
    # For each batch size, find geomean speedup and plot
    batch_sizes_np = np.array(BATCH_SIZES)
    plot_xlabel = 'Batch sizes'
    plot_ylabel = 'Speedup'

    PlotLineGraph(
        batch_sizes_np,
        "scalar vs uniform tiling vs {uniform + prob.} tiling",
        plot_xlabel,
        plot_ylabel,
        prefix + '_mean_speedups_prob_tiling_vs_scalar.png',
        speedups=[speedups.uniform_tiling_speedups,
                  speedups.prob_tiling_speedups],
        names=['Basic tiling', 'Probability based tiling'],
        markers = ['o', 'D'])

    PlotLineGraph(
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

def PlotLineGraphForDifferentCpus(batch_results_intel, batch_results_amd, prefix):
    # For each batch size, find geomean speedup and plot
    batch_sizes_np = np.array(BATCH_SIZES)
    plot_xlabel = 'Batch sizes'
    plot_ylabel = 'Speedup'

    intel_speedups = []
    amd_speedups = []

    for batch_size in BATCH_SIZES:
      intel_speedups.append(batch_results_intel[batch_size].overall_mean_speedup)
      amd_speedups.append(batch_results_amd[batch_size].overall_mean_speedup)

    PlotLineGraph(
        batch_sizes_np,
        "scalar vs uniform tiling vs {uniform + prob.} tiling",
        plot_xlabel,
        plot_ylabel,
        prefix + '_overall_mean_speedups_intel_amd.png',
        speedups=[intel_speedups, amd_speedups],
        names=['Intel RocketLake', 'AMD Ryzen 7'],
        markers = ['o', 'D'])


def PlotLineGraphForSpeedupOverFrameworks(speedups, prefix):
    # For each batch size, find geomean speedup and plot
    batch_sizes_np = np.array(BATCH_SIZES)
    plot_xlabel = 'Batch sizes'
    plot_ylabel = 'Speedup'

    speedups_over_xgboost = [speedups[batch_size].mean_speedup_over_xgboost for batch_size in BATCH_SIZES]
    speedups_over_treelite = [speedups[batch_size].mean_speedup_over_treelite for batch_size in BATCH_SIZES]

    PlotLineGraph(
        batch_sizes_np,
        "treebeard vs {xgboost, treelite} - " + prefix,
        plot_xlabel,
        plot_ylabel,
        prefix + '_mean_speedups_treebeard_vs_xgboost_treelite.png',
        speedups=[speedups_over_xgboost, speedups_over_treelite],
        names=['vs XGBoost', 'vs Treelite'],
        markers = ['o', 'D'])

def ConvertFloatArrayToStringArray(floatArray: List[float], unit = ''):
  return [f'{float:.2f}' + unit for float in floatArray]

def PlotBarGraphForBenchmarkSpeedupsOverFrameworks(speedups, model_names: List[str], batch_size, prefix):
    xgboost_speedups = speedups[batch_size].xgboost_speedups + [speedups[batch_size].mean_speedup_over_xgboost]
    treelite_speedups = speedups[batch_size].treelite_speedups + [speedups[batch_size].mean_speedup_over_treelite]

    xgboost_times = speedups[batch_size].xgboost_times + [speedups[batch_size].mean_speedup_over_xgboost]
    treelite_times = speedups[batch_size].treelite_times + [speedups[batch_size].mean_speedup_over_treelite]

    file_name = prefix + "_speedup_per_benchmark_xgoost_treelite.png"
    PlotBarGraph(
      model_names + ['geomean'],
      file_name,
      speedups = [xgboost_speedups, treelite_speedups],
      names = ["XGBoost", "Treelite"],
      patterns = ['///', '...'],
      cut_off = 8 if prefix == 'serial' else -1,
      bar_labels = [ConvertFloatArrayToStringArray(xgboost_times, ''), ConvertFloatArrayToStringArray(treelite_times, '')])

def PlotBarGraph(model_names, file_name, cut_off = -1, **data):
  x_axis = np.arange(0, len(model_names) / 2, step=0.5)
  bar = None
  # np.array(model_names)

  plt.figure(figsize=(12, 5))
  width = 0.2
  step = width / 2
  left_most = -len(data['speedups']) * step + step
  rect = None
  for speedup, pattern in zip(data['speedups'], data['patterns']):
    speedup = speedup if cut_off == -1 else [min(x, cut_off) for x in speedup]
    bar = plt.bar(x_axis + left_most, np.array(speedup), alpha=.99, width=width, hatch=pattern)
    if not rect:
      rect = bar
    else:
      rect += bar
    left_most += width
  
  if 'bar_labels' in data:
    bar_labels = []
    for bar_label in data['bar_labels']:
      bar_labels += bar_label
    
    # print(len(rect), len(bar_labels))
    hack_offset = 0
    for i, bar in enumerate(rect):
      height = bar.get_height()
      if len(bar_labels[i]) == 6:
        hack_offset = 0.05 * (1 if i % 2 == 0 else -1)

      # plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom', fontsize=11)
      plt.text(bar.get_x() + bar.get_width() / 2.0 + hack_offset, height, bar_labels[i], ha='center', va='bottom', fontsize=12)
      hack_offset = 0

  plt.xticks(x_axis, model_names, fontsize=15)
  plt.yticks(fontsize=20)

  plt.xlabel("Benchmark", fontsize=20)
  plt.ylabel("Speedup", fontsize=20)
  plt.legend(data['names'], fontsize=15)
  plt.tight_layout(h_pad=0.0)
  # plt.title(title + ". Tile size = " + str(tile_size))

  plt.savefig(file_name, bbox_inches='tight')
  plt.close()

def PlotBarGraphForScalarSpeedups(batch_size_results, model_names: List[str], batch_size, prefix):
  unif_tiling_speedups = []
  prob_tiling_speedups = []
  pipeline_speedups = []

  batch_size_result = batch_size_results[batch_size]
  for i, model_name in enumerate(model_names):
    unif_tiling_speedups.append(batch_size_result.unif_tiling_speedups[i])
    prob_tiling_speedups.append(batch_size_result.prob_tiling_speedups[i] * batch_size_result.unif_tiling_speedups[i])
    pipeline_speedups.append(batch_size_result.pipelining_speedups_over_unif[i] * batch_size_result.unif_tiling_speedups[i])
  
  file_name = prefix + "_max_speedup_unif_prob_tiling.png"
  bar_labels_basic_tiling = batch_size_result.unif_tiling_times + [gmean(unif_tiling_speedups)]
  bar_labels_prob_tiling = batch_size_result.prob_tiling_times + [gmean(prob_tiling_speedups)]

  PlotBarGraph(
    model_names + ['geomean'],
    file_name,
    speedups = [unif_tiling_speedups + [gmean(unif_tiling_speedups)], prob_tiling_speedups + [gmean(prob_tiling_speedups)]],
    names = ["Basic tiling", "Probability based tiling"],
    patterns = ['///', '...'],
    bar_labels = [ConvertFloatArrayToStringArray(bar_labels_basic_tiling), ConvertFloatArrayToStringArray(bar_labels_prob_tiling)])

  file_name = prefix + "_max_speedup_unif_tiling_pipelining.png"
  bar_labels_pipeline = batch_size_result.pipelining_times + [gmean(pipeline_speedups)]
  PlotBarGraph(
    model_names + ['geomean'],
    file_name,
    speedups = [unif_tiling_speedups + [gmean(unif_tiling_speedups)], pipeline_speedups + [gmean(pipeline_speedups)]],
    names = ["Basic tiling", "Basic tiling + Interleaving + Peeling and Unrolling"],
    patterns = ['///', '...'],
    bar_labels = [ConvertFloatArrayToStringArray(bar_labels_basic_tiling), ConvertFloatArrayToStringArray(bar_labels_pipeline)])

def PlotBarGraphForParallelSpeedups(batch_size_results, model_names: List[str], tile_size, prefix):
  unif_tiling_speedups = []
  parallel_speedups = []

  for i, model_name in enumerate(model_names):
    cur_result = batch_size_results[tile_size]
    unif_tiling_speedups.append(cur_result.unif_tiling_speedups[i])
    parallel_speedups.append(cur_result.pipelining_speedups_over_unif[i] * cur_result.unif_tiling_speedups[i])
  

  file_name = prefix + "_max_speedup_unif_tiling_pipelining.png"
  PlotBarGraph(model_names, file_name, speedups=[parallel_speedups], names=["speedup from parallelization"], patterns=[''])

def PlotBarGraphForIntelAndAMDSpeedups(batch_size_results_intel, batch_size_results_amd, model_names: List[str], batch_size, prefix):
  overall_speedups_intel = []
  overall_speedups_amd = []
  bar_labels_intel = []
  bar_labels_amd = []

  cur_result_intel = batch_size_results_intel[batch_size]
  cur_result_amd = batch_size_results_amd[batch_size]
  for i, model_name in enumerate(model_names):
    overall_speedups_intel.append(cur_result_intel.overall_speedups[i])
    overall_speedups_amd.append(cur_result_amd.overall_speedups[i])
    bar_labels_intel.append(cur_result_intel.best_times[i])
    bar_labels_amd.append(cur_result_amd.best_times[i])
  
  bar_labels_intel += [cur_result_intel.overall_mean_speedup]
  bar_labels_amd += [cur_result_amd.overall_mean_speedup]

  file_name = prefix + "_overall_speedup_benchmark_intel_amd.png"
  PlotBarGraph(
    model_names + ['geomean'],
    file_name,
    speedups = [overall_speedups_intel + [cur_result_intel.overall_mean_speedup], overall_speedups_amd + [cur_result_amd.overall_mean_speedup]],
    names = ['Intel RocketLake', 'AMD Ryzen 7'],
    patterns = ['///', '...'],
    bar_labels = [ConvertFloatArrayToStringArray(bar_labels_intel, ''), ConvertFloatArrayToStringArray(bar_labels_amd, '')])


model_names = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "letter", "higgs", "year"]
data_file_path = CURRENT_DIRECTORY + "/holmes/multiple_batchSizes_results_parallel_varyingcores_final.txt"
df_headers = ['config', 'type', 'batch size', 'ncores'] + model_names
batch_core_scaling_intel = ParseParallelScalingSpeedups(data_file_path, model_names, df_headers)

data_file_path = CURRENT_DIRECTORY + "/bhagheera/multiple_batchSizes_results_parallel_varyingcores_final.txt"
batch_core_scaling_amd = ParseParallelScalingSpeedups(data_file_path, model_names, df_headers)
PlotLineGraphForCoreScaling(batch_core_scaling_intel, batch_core_scaling_amd, 'parallel', GRAPH_BATCH_SIZE)

df_headers = ["config", "type", "batch size", "tile size"] + model_names
data_file_path = CURRENT_DIRECTORY + "/holmes/multiple_batchSizes_results_final.txt"
speedups, batch_size_results_intel = ParseOptimizationSpeedups(data_file_path, model_names, df_headers)
PlotLineGraphForScalarSpeedup(speedups, "scalar_Intel")
PlotBarGraphForScalarSpeedups(batch_size_results_intel, model_names, GRAPH_BATCH_SIZE, "scalar_Intel")

data_file_path = CURRENT_DIRECTORY + "/bhagheera/multiple_batchSizes_results_serial_final.txt"
speedups, batch_size_results_amd = ParseOptimizationSpeedups(data_file_path, model_names, df_headers)
PlotLineGraphForScalarSpeedup(speedups, "scalar_AMD")
PlotBarGraphForScalarSpeedups(batch_size_results_amd, model_names, GRAPH_BATCH_SIZE, "scalar_AMD")

PlotLineGraphForDifferentCpus(batch_size_results_intel, batch_size_results_amd, 'scalar')
PlotBarGraphForIntelAndAMDSpeedups(batch_size_results_intel, batch_size_results_amd, model_names, GRAPH_BATCH_SIZE, 'scalar')

data_file_path = CURRENT_DIRECTORY + "/holmes/multiple_batchSizes_results_parallel_16cores_final.txt"
speedups, batch_size_results_intel = ParseOptimizationSpeedups(data_file_path, model_names, df_headers, True)
# PlotLineGraphForMultiCoreSpeedup(speedups, 'parallel_Intel')
# PlotBarGraphForParallelSpeedups(batch_size_results_intel, model_names, GRAPH_BATCH_SIZE, "parallel_Intel")

data_file_path = CURRENT_DIRECTORY + "/bhagheera/multiple_batchSizes_results_parallel_final.txt"
speedups, batch_size_results_amd = ParseOptimizationSpeedups(data_file_path, model_names, df_headers, True)
# PlotLineGraphForMultiCoreSpeedup(speedups, 'parallel_AMD')
# PlotBarGraphForParallelSpeedups(batch_size_results_amd, model_names, GRAPH_BATCH_SIZE, "parallel_AMD")

PlotLineGraphForDifferentCpus(batch_size_results_intel, batch_size_results_amd, 'parallel')
PlotBarGraphForIntelAndAMDSpeedups(batch_size_results_intel, batch_size_results_amd, model_names, GRAPH_BATCH_SIZE, 'parallel')

df_headers = ["config", "batch_size"] + model_names
data_file_path = CURRENT_DIRECTORY + "/holmes/xgboost_treelite_compare_serial.txt"
speedups = ParseFrameworkTimes(data_file_path, model_names, df_headers)
PlotLineGraphForSpeedupOverFrameworks(speedups, "scalar_Intel")
PlotBarGraphForBenchmarkSpeedupsOverFrameworks(speedups, model_names, GRAPH_BATCH_SIZE, "serial")

df_headers = ["config", "batch_size"] + model_names
data_file_path = CURRENT_DIRECTORY + "/holmes/xgboost_treelite_compare_parallel.txt"
speedups = ParseFrameworkTimes(data_file_path, model_names, df_headers)
PlotLineGraphForSpeedupOverFrameworks(speedups, "parallel_Intel")
PlotBarGraphForBenchmarkSpeedupsOverFrameworks(speedups, model_names, GRAPH_BATCH_SIZE, "parallel")
