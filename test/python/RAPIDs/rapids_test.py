import os
import math
import cupy as cp
import time
import numpy
import pandas
import xgboost as xgb
import treebeard
from cuml import ForestInference
from functools import partial

filepath = os.path.abspath(__file__)
treebeard_repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(filepath))))

# num_repeats = 50
run_parallel = False
num_tiles = 4
batchSize = 4096 * 2

#############################################################
### Treebeard Configs
#############################################################

def get_model_config(model_name : str, batch_size: int):
  abalone_best_configs = {
    256: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 50, True, False, True, 2), "gpu_array"),
    512: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 50, True, False, True, 2), "gpu_reorg"),
    1024: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 20, True, False, True, 2), "gpu_reorg"),
    4096: (treebeard.GPUAutoScheduleOptions.Construct(32, 1, 20, True, False, True, 2), "gpu_array"),
    8192: (treebeard.GPUAutoScheduleOptions.Construct(32, 1, 10, True, False, True, 4), "gpu_array"),
    16384: (treebeard.GPUAutoScheduleOptions.Construct(64, 1, 2, False, False, True, 2), "gpu_array"),
  }
  airline_best_configs = {
    256: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 50, False, False, True, -1), "gpu_array"),
    512: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 50, False, False, False, -1), "gpu_array"),
    1024: (treebeard.GPUAutoScheduleOptions.Construct(32, 1, 20, True, False, True, -1), "gpu_reorg"),
    4096: (treebeard.GPUAutoScheduleOptions.Construct(32, 2, 10, True, False, True, -1), "gpu_array"),
    8192: (treebeard.GPUAutoScheduleOptions.Construct(64, 1, 10, True, False, True, 2), "gpu_array"),
    16384: (treebeard.GPUAutoScheduleOptions.Construct(64, 1, 2, True, False, True, 2), "gpu_array"),
  }
  airline_ohe_best_configs = {
    256: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 50, True, False, True, 4), "gpu_array"),
    512: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 50, True, False, True, 2), "gpu_array"),
    1024: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 50, True, False, True, 2), "gpu_reorg"),
    4096: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 50, True, False, True, 2), "gpu_array"),
    8192: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 50, True, False, True, 2), "gpu_array"),
    16384: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 50, True, False, True, 2), "gpu_array"),
  }
  covtype_best_configs = {
    256: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 50, True, False, True, -1), "gpu_sparse"),
    512: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 50, True, False, True, -1), "gpu_sparse"),
    1024: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 20, True, False, True, 2), "gpu_array"),
    4096: (treebeard.GPUAutoScheduleOptions.Construct(32, 1, 2, True, False, True, 4), "gpu_array"),
    8192: (treebeard.GPUAutoScheduleOptions.Construct(32, 1, 1, True, False, True, 4), "gpu_array"),
    16384: (treebeard.GPUAutoScheduleOptions.Construct(64, 1, 2, True, False, True, 4), "gpu_array"),
  }
  epsilon_best_configs = {
    256: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 50, False, False, True, 2), "gpu_sparse"),
    512: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 50, False, False, True, -1), "gpu_sparse"),
    1024: (treebeard.GPUAutoScheduleOptions.Construct(32, 2, 50, False, False, True, 4), "gpu_array"),
    4096: (treebeard.GPUAutoScheduleOptions.Construct(32, 2, 50, False, False, True, 2), "gpu_array"),
    8192: (treebeard.GPUAutoScheduleOptions.Construct(32, 2, 50, False, False, True, 2), "gpu_sparse"),
    16384: (treebeard.GPUAutoScheduleOptions.Construct(32, 2, 50, False, False, True, 4), "gpu_sparse"),
  }
  higgs_best_configs = {
    256: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 50, True, False, True, -1), "gpu_array"),
    512: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 50, False, False, False, -1), "gpu_array"),
    1024: (treebeard.GPUAutoScheduleOptions.Construct(32, 1, 20, True, False, True, -1), "gpu_array"),
    4096: (treebeard.GPUAutoScheduleOptions.Construct(32, 1, 20, True, False, True, -1), "gpu_array"),
    8192: (treebeard.GPUAutoScheduleOptions.Construct(32, 1, 1, True, False, True, 4), "gpu_sparse"),
    16384: (treebeard.GPUAutoScheduleOptions.Construct(32, 1, 1, True, False, True, 2), "gpu_array"),
  }
  year_prediction_msd_best_configs = {
    256: (treebeard.GPUAutoScheduleOptions.Construct(2, 1, 20, True, False, False, -1), "gpu_array"),
    512: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 50, False, False, False, -1), "gpu_sparse"),
    1024: (treebeard.GPUAutoScheduleOptions.Construct(32, 1, 20, False, False, False, -1), "gpu_array"),
    4096: (treebeard.GPUAutoScheduleOptions.Construct(32, 1, 10, True, False, False, -1), "gpu_array"),
    8192: (treebeard.GPUAutoScheduleOptions.Construct(32, 1, 10, True, False, False, -1), "gpu_array"),
    16384: (treebeard.GPUAutoScheduleOptions.Construct(32, 1, 10, True, False, False, -1), "gpu_array"),
  }
  letters_best_configs = {
    256: (treebeard.GPUAutoScheduleOptions.Construct(8, 1, 50, True, False, False, -1), "gpu_sparse"),
    512: (treebeard.GPUAutoScheduleOptions.Construct(32, 1, 20, True, False, True, 2), "gpu_array"),
    1024: (treebeard.GPUAutoScheduleOptions.Construct(32, 1, 10, True, False, True, 2), "gpu_sparse"),
    4096: (treebeard.GPUAutoScheduleOptions.Construct(64, 1, 2, True, False, True, 4), "gpu_sparse"),
    8192: (treebeard.GPUAutoScheduleOptions.Construct(32, 1, 2, True, False, True, 4), "gpu_sparse"),
    16384: (treebeard.GPUAutoScheduleOptions.Construct(64, 1, 10, True, False, True, 4), "gpu_sparse"),
  }
  benchmark_to_config_map_map = {
    "abalone": abalone_best_configs,
    "airline": airline_best_configs,
    "airline-ohe": airline_ohe_best_configs,
    "covtype": covtype_best_configs,
    "epsilon": epsilon_best_configs,
    "higgs": higgs_best_configs,
    "year_prediction_msd": year_prediction_msd_best_configs,
    "letters": letters_best_configs,
  }
  return benchmark_to_config_map_map[model_name][batch_size]

def get_compiler_options_and_result_type(model_name : str, batch_size: int, tile_size: int=1):
  defaultOptions = treebeard.CompilerOptions(batch_size, tile_size)
  defaultOptions.SetMakeAllLeavesSameDepth(1)
  defaultMulticlassOptions = treebeard.CompilerOptions(batch_size, tile_size)
  defaultMulticlassOptions.SetReturnTypeWidth(8)
  defaultMulticlassOptions.SetReturnTypeIsFloatType(False)
  defaultMulticlassOptions.SetMakeAllLeavesSameDepth(1)
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
    return 50
  else:
    return 500

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
  for i in range(num_repeats):
    for j in range(0, num_batches):
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
### Treebeard Total Time Benchmarking Utils
#############################################################

def RunSingleTest_Treebeard(modelName, modelJSONPath, csvPath, num_repeats) -> float:
  inputFileType = "xgboost_json"
  gpu_schedule_options, representation = get_model_config(modelName, batchSize)
  options, returnType = get_compiler_options_and_result_type(modelName, batchSize)
  globalsPath = modelJSONPath + ".treebeard-globals.json"
  tbContext = treebeard.TreebeardContext(modelJSONPath, globalsPath, options, gpuAutoScheduleOptions=gpu_schedule_options)
  tbContext.SetRepresentationType(representation)
  tbContext.SetInputFiletype(inputFileType)

  inferenceRunner = treebeard.TreebeardInferenceRunner.GPUInferenceRunnerFromTBContext(tbContext)
  
  num_batches, inputs = construct_inputs(csvPath, batchSize)

  start = time.time()
  for i in range(num_repeats):
    for j in range(0, num_batches):
      start_index = j*batchSize
      stop_index = start_index + batchSize
      batch = inputs[start_index:stop_index, :]
      results = inferenceRunner.RunInference(batch, returnType)

  end = time.time()
  return (end-start)/(batchSize * num_batches * num_repeats)

def RunTestOnSingleModelTestInputs_Treebeard(modelName : str) -> None:
  modelJSON = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json")
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
  num_repeats = get_num_repeats(modelName)
  return RunSingleTest_Treebeard(modelName, modelJSON, csvPath, num_repeats)

#############################################################
### Treebeard Kernel Time Benchmarking Utils
#############################################################

def RunSingleTest_Treebeard_KernelTime(modelName, modelJSONPath, csvPath) -> float:
  inputFileType = "xgboost_json"
  gpu_schedule_options, representation = get_model_config(modelName, batchSize)
  options, returnType = get_compiler_options_and_result_type(modelName, batchSize)
  globalsPath = modelJSONPath + ".treebeard-globals.json"

  num_kernel_runs = get_num_repeats(modelName)
  treebeard.SetNumberOfKernelRuns(num_kernel_runs)
  treebeard.SetEnableMeasureGpuKernelTime(1)

  tbContext = treebeard.TreebeardContext(modelJSONPath, globalsPath, options, gpuAutoScheduleOptions=gpu_schedule_options)
  tbContext.SetRepresentationType(representation)
  tbContext.SetInputFiletype(inputFileType)

  inferenceRunner = treebeard.TreebeardInferenceRunner.GPUInferenceRunnerFromTBContext(tbContext)
  
  num_batches, inputs = construct_inputs(csvPath, batchSize)
  # start = time.time()
  for j in range(0, num_batches):
    start_index = j*batchSize
    stop_index = start_index + batchSize
    batch = inputs[start_index:stop_index, :]
    results = inferenceRunner.RunInference(batch, returnType)

  # end = time.time()
  kernel_time = treebeard.GetGPUKernelExecutionTime(inferenceRunner)
  
  treebeard.SetNumberOfKernelRuns(1)
  treebeard.SetEnableMeasureGpuKernelTime(0)

  return (kernel_time)/(1e6 * batchSize * num_batches * num_kernel_runs)

def RunTestOnSingleModelTestInputs_Treebeard_KernelTime(modelName : str) -> None:
  modelJSON = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json")
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
  return RunSingleTest_Treebeard_KernelTime(modelName, modelJSON, csvPath)


if __name__ == "__main__":
  num_trials = 5
  # benchmarks = ["abalone", "airline", "airline-ohe", "covtype", "higgs", "letters", "year_prediction_msd"]
  # benchmarks = ["epsilon"]
  # batch_sizes = [16384]

  benchmarks = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "higgs", "letters", "year_prediction_msd"]
  batch_sizes = [4096, 8192, 16384]
  
  rapids_total_times = []
  rapids_kernel_times = []
  treebeard_total_times = []
  treebeard_kernel_times = []
  total_time_speedups = []
  kernel_time_speedups = []
  for batchSize in batch_sizes:
    for benchmark in benchmarks:
      rapids_total_func = partial(RunTestOnSingleModelTestInputs_RAPIDs, modelName=benchmark)
      rapids_total_time = run_benchmark_function_and_return_median_time(rapids_total_func, num_trials)
      rapids_total_times.append(rapids_total_time)

      # if not ((benchmark == "airline-ohe" and batchSize == 16384) or (benchmark == "epsilon" and batchSize == 16384)):
      #   rapids_kernel_time_func = partial(RunTestOnSingleModelTestInputs_RAPIDs_KernelTime, modelName=benchmark)
      #   rapids_kernel_time = run_benchmark_function_and_return_median_time(rapids_kernel_time_func, num_trials)
      #   rapids_kernel_times.append(rapids_total_time)
      # else:
      #   rapids_kernel_time = 1
      #   rapids_kernel_times.append(1)

      rapids_kernel_time_func = partial(RunTestOnSingleModelTestInputs_RAPIDs_KernelTime, modelName=benchmark)
      rapids_kernel_time = run_benchmark_function_and_return_median_time(rapids_kernel_time_func, num_trials)
      rapids_kernel_times.append(rapids_total_time)

      treebeard_total_func = partial(RunTestOnSingleModelTestInputs_Treebeard, modelName=benchmark)
      treebeard_total_time = run_benchmark_function_and_return_median_time(treebeard_total_func, num_trials)
      treebeard_total_times.append(treebeard_total_time)
      
      treebeard_kernel_time_func = partial(RunTestOnSingleModelTestInputs_Treebeard_KernelTime, modelName=benchmark)
      treebeard_kernel_time = run_benchmark_function_and_return_median_time(treebeard_kernel_time_func, num_trials)
      treebeard_kernel_times.append(treebeard_kernel_time)

      total_time_speedups.append(rapids_total_time/treebeard_total_time)
      kernel_time_speedups.append(rapids_kernel_time/treebeard_kernel_time)
      
      print(benchmark, batchSize, rapids_total_time, treebeard_total_time, rapids_total_time/treebeard_total_time, rapids_kernel_time, treebeard_kernel_time, rapids_kernel_time/treebeard_kernel_time)
    # print the geometric mean of the speedups for this batch size
    print("Geomean speedup (Total):", batchSize, numpy.prod(total_time_speedups)**(1/len(total_time_speedups)))
    print("Geomean speedup (Kernel):", batchSize, numpy.prod(kernel_time_speedups)**(1/len(kernel_time_speedups)))