import argparse
import os
import math
import cupy as cp
import time
import numpy
import pandas
import treebeard
import matplotlib.pyplot as plt
import warnings
from cuml import ForestInference
from functools import partial

filepath = os.path.abspath(__file__)
treebeard_repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(filepath))))

# num_repeats = 50
num_tiles = 4
batchSize = 4096 * 2

def get_compiler_options_and_result_type(model_name : str, batch_size: int, unroll_walks: bool, tile_size: int=1):
  defaultOptions = treebeard.CompilerOptions(batch_size, tile_size)
  defaultOptions.SetMakeAllLeavesSameDepth(1 if unroll_walks else 0)
  defaultMulticlassOptions = treebeard.CompilerOptions(batch_size, tile_size)
  defaultMulticlassOptions.SetReturnTypeWidth(8)
  defaultMulticlassOptions.SetReturnTypeIsFloatType(False)
  defaultMulticlassOptions.SetMakeAllLeavesSameDepth(1 if unroll_walks else 0)
  if model_name == "covtype" or model_name == "letters":
    return defaultMulticlassOptions, numpy.int8
  else:
    return defaultOptions, numpy.float32

#############################################################
### Common Utils
#############################################################

def construct_inputs(csvPath : str, batch_size) -> numpy.ndarray:
  data_df = pandas.read_csv(csvPath, header=None)
  full_test_array = numpy.array(data_df, order='C')
  num_repeats = num_tiles
  if (full_test_array.shape[0] * num_repeats < batch_size):
    num_repeats = math.ceil(batch_size/full_test_array.shape[0])
  full_test_array = numpy.tile(full_test_array, (num_repeats, 1))
  numRows = int(math.floor(full_test_array.shape[0]/batch_size) * batch_size)
  inputs = numpy.array(full_test_array[0:numRows, :-1], numpy.float32, order='C')

  num_batches = int(math.floor(full_test_array.shape[0]/batch_size))

  return num_batches, inputs

def run_benchmark_function_and_return_median_time(benchmark_function, num_repeats):
  times = []
  for i in range(num_repeats):
    time = benchmark_function()
    times.append(time)
  return numpy.median(times)

def get_num_repeats(model_name : str):
  if model_name == "letters":
    return 200
  else:
    return 600

#############################################################
### RAPIDs Total Time Benchmarking Utils
#############################################################

def RunSingleTest_RAPIDs(modelJSONPath, csvPath, output_class, num_repeats) -> float:
  # booster = xgb.Booster(model_file=modelJSONPath)
  # booster.set_param({"predictor": "gpu_predictor"})
  # cu_model = ForestInference.load(booster)
  cu_model = ForestInference.load(filename=modelJSONPath, output_class=output_class, model_type='xgboost_json', precision='float32')

  num_batches, inputs = construct_inputs(csvPath, batchSize)

  start = time.time()
  for i in range(num_repeats):
    for j in range(0, num_batches):
      start_index = j*batchSize
      stop_index = start_index + batchSize
      batch = inputs[start_index:stop_index, :]
      pred = cu_model.predict(batch)

  end = time.time()
  return (end-start)/(batchSize * num_batches * num_repeats)


def RunTestOnSingleModelTestInputs_RAPIDs(modelName : str) -> None:
  modelJSON = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json")
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
  output_class = modelName == "covtype" or modelName == "letters"
  num_repeats = get_num_repeats(modelName)
  return RunSingleTest_RAPIDs(modelJSON, csvPath, output_class, num_repeats)

#############################################################
### RAPIDs Kernel Time Benchmarking Utils
#############################################################

def RunSingleTest_RAPIDs_KernelTime(modelJSONPath, csvPath, output_class, num_repeats) -> float:
  # booster = xgb.Booster(model_file=modelJSONPath)
  # booster.set_param({"predictor": "gpu_predictor"})
  # cu_model = ForestInference.load(booster)
  cu_model = ForestInference.load(filename=modelJSONPath, output_class=output_class, model_type='xgboost_json', precision='float32')

  num_batches, inputs = construct_inputs(csvPath, batchSize)
  gpu_array = cp.array(inputs, dtype=numpy.float32)

  start = time.time()
  for j in range(0, num_batches):
    for i in range(num_repeats):
      start_index = j*batchSize
      stop_index = start_index + batchSize
      batch = gpu_array[start_index:stop_index, :]
      pred = cu_model.predict(batch)

  end = time.time()
  del gpu_array
  cp._default_memory_pool.free_all_blocks()
  return (end-start)/(batchSize * num_batches * num_repeats)


def RunTestOnSingleModelTestInputs_RAPIDs_KernelTime(modelName : str) -> None:
  modelJSON = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json")
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
  output_class = modelName == "covtype" or modelName == "letters"
  num_repeats = get_num_repeats(modelName)
  return RunSingleTest_RAPIDs_KernelTime(modelJSON, csvPath, output_class, num_repeats)

#############################################################
### Treebeard Auto-Tune Heuristic Benchmarking Utils
#############################################################

def RunSingleTest_Treebeard_AutoTuningHeuristic(modelName, modelJSONPath, csvPath, num_repeats) -> float:
  inputFileType = "xgboost_json"
  
  options, returnType = get_compiler_options_and_result_type(modelName, batchSize, False)
  globalsPath = modelJSONPath + ".treebeard-globals.json"
  tbContext = treebeard.TreebeardContext(modelJSONPath, globalsPath, options)
  tbContext.SetInputFiletype(inputFileType)

  inferenceRunner = treebeard.TreebeardInferenceRunner.AutoScheduleGPUInferenceRunnerFromTBContext(tbContext)
  
  num_batches, inputs = construct_inputs(csvPath, batchSize)

  times = []
  for i in range(num_trials):
    start = time.time()
    for i in range(num_repeats):
      for j in range(0, num_batches):
        start_index = j*batchSize
        stop_index = start_index + batchSize
        batch = inputs[start_index:stop_index, :]
        results = inferenceRunner.RunInference(batch, returnType)

    end = time.time()
    run_time = (end-start)/(batchSize * num_batches * num_repeats)
    times.append(run_time)
  
  return numpy.median(times)

def RunTestOnSingleModelTestInputs_Treebeard_AutoTuneHeuristic(modelName : str) -> None:
  modelJSON = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json")
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
  num_repeats = get_num_repeats(modelName)
  return RunSingleTest_Treebeard_AutoTuningHeuristic(modelName, modelJSON, csvPath, num_repeats)

def RunSingleTest_Treebeard_AutoTuningHeuristic_Kernel(modelName, modelJSONPath, csvPath, num_repeats) -> float:
  inputFileType = "xgboost_json"
  
  options, returnType = get_compiler_options_and_result_type(modelName, batchSize, False)
  globalsPath = modelJSONPath + ".treebeard-globals.json"
  tbContext = treebeard.TreebeardContext(modelJSONPath, globalsPath, options)
  tbContext.SetInputFiletype(inputFileType)
  
  treebeard.SetNumberOfKernelRuns(num_repeats)
  treebeard.SetEnableMeasureGpuKernelTime(1)

  inferenceRunner = treebeard.TreebeardInferenceRunner.AutoScheduleGPUInferenceRunnerFromTBContext(tbContext)
  
  num_batches, inputs = construct_inputs(csvPath, batchSize)

  times = []
  prev_time = 0
  for i in range(num_trials):
    for j in range(0, num_batches):
      start_index = j*batchSize
      stop_index = start_index + batchSize
      batch = inputs[start_index:stop_index, :]
      results = inferenceRunner.RunInference(batch, returnType)

    kernel_time = treebeard.GetGPUKernelExecutionTime(inferenceRunner)
    run_time = (kernel_time - prev_time)/(1e6 * batchSize * num_batches * num_repeats)
    prev_time = kernel_time
    times.append(run_time)
  
  treebeard.SetNumberOfKernelRuns(1)
  treebeard.SetEnableMeasureGpuKernelTime(0)

  return numpy.median(times)

def RunTestOnSingleModelTestInputs_Treebeard_AutoTuneHeuristic_Kernel(modelName : str) -> None:
  modelJSON = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json")
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
  num_repeats = get_num_repeats(modelName)
  return RunSingleTest_Treebeard_AutoTuningHeuristic_Kernel(modelName, modelJSON, csvPath, num_repeats)

if __name__ == "__main__":
  # create an argument parser
  parser = argparse.ArgumentParser(description="Benchmarking script for RAPIDs and Treebeard")
  parser.add_argument("--num_trials", type=int, default=3, help="Number of trials to run for each benchmark")
  parser.add_argument("--batch_size", type=int, default=4096, help="Batch sizes to run")
  parser.add_argument("--output_file", type=str, default="benchmark_results.csv", help="Output file to write the results to")
  args = parser.parse_args()

  num_trials = args.num_trials
  benchmarks = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "higgs", "letters", "year_prediction_msd"]
  batchSize = args.batch_size
  output_file = args.output_file

  rapids_total_times = []
  rapids_kernel_times = []
  treebeard_auto_tune_heuristic_total_times = []
  treebeard_auto_tune_heuristic_kernel_times = []
  
  autotune_total_time_speedups = []
  autotune_kernel_time_speedups = []

  for benchmark in benchmarks:
    rapids_total_func = partial(RunTestOnSingleModelTestInputs_RAPIDs, modelName=benchmark)
    rapids_total_time = run_benchmark_function_and_return_median_time(rapids_total_func, num_trials)
    rapids_total_times.append(rapids_total_time)

    rapids_kernel_time_func = partial(RunTestOnSingleModelTestInputs_RAPIDs_KernelTime, modelName=benchmark)
    rapids_kernel_time = run_benchmark_function_and_return_median_time(rapids_kernel_time_func, num_trials)
    rapids_kernel_times.append(rapids_total_time)

    treebeard_auto_tune_heuristic_total_func = partial(RunTestOnSingleModelTestInputs_Treebeard_AutoTuneHeuristic, modelName=benchmark)
    treebeard_auto_tune_heuristic_total_time = run_benchmark_function_and_return_median_time(treebeard_auto_tune_heuristic_total_func, num_trials)
    treebeard_auto_tune_heuristic_total_times.append(treebeard_auto_tune_heuristic_total_time)

    treebeard_auto_tune_heuristic_kernel_func = partial(RunTestOnSingleModelTestInputs_Treebeard_AutoTuneHeuristic_Kernel, modelName=benchmark)
    treebeard_auto_tune_heuristic_kernel_time = run_benchmark_function_and_return_median_time(treebeard_auto_tune_heuristic_kernel_func, num_trials)
    treebeard_auto_tune_heuristic_kernel_times.append(treebeard_auto_tune_heuristic_kernel_time)

    autotune_total_time_speedups.append(rapids_total_time/treebeard_auto_tune_heuristic_total_time)
    autotune_kernel_time_speedups.append(rapids_kernel_time/treebeard_auto_tune_heuristic_kernel_time)
    
    print(benchmark, batchSize, 
          rapids_total_time,  
          treebeard_auto_tune_heuristic_total_time,
          rapids_total_time/treebeard_auto_tune_heuristic_total_time, 
          rapids_kernel_time, treebeard_auto_tune_heuristic_kernel_time,
          rapids_kernel_time/treebeard_auto_tune_heuristic_kernel_time,
          flush=True)
  # print the geometric mean of the speedups for this batch size
  print(numpy.prod(autotune_total_time_speedups)**(1.0/len(autotune_total_time_speedups)),
        numpy.prod(autotune_kernel_time_speedups)**(1.0/len(autotune_kernel_time_speedups)), flush=True)
