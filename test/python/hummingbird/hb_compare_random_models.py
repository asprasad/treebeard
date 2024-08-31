import os
import math
import time
import numpy
import pandas
import xgboost as xgb
import treebeard
import argparse
import json
from functools import partial
from hummingbird.ml import convert

filepath = os.path.abspath(__file__)
treebeard_repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(filepath))))

# num_repeats = 50
run_parallel = False
num_tiles = 1
batchSize = 4096 * 2

def get_compiler_options_and_result_type(batch_size: int, unroll_walks: bool, tile_size: int=1):
  defaultOptions = treebeard.CompilerOptions(batch_size, tile_size)
  defaultOptions.SetMakeAllLeavesSameDepth(1 if unroll_walks else 0)
  return defaultOptions, numpy.float32

#############################################################
### Common Utils
#############################################################

def construct_inputs(csvPath : str, batch_size) -> numpy.ndarray:
  data_df = pandas.read_csv(csvPath, header=None, skiprows=1, sep=' ')
  # with open(csvPath, 'r') as f:
  #   f.readline()
  #   data_df = pandas.read_csv(f, header=None)
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
### Treebeard Auto-Tune Heuristic Benchmarking Utils
#############################################################

def RunSingleTest_Treebeard_AutoTuningHeuristic(modelJSONPath, csvPath, num_repeats) -> float:
  inputFileType = "xgboost_json"
  
  options, returnType = get_compiler_options_and_result_type(batchSize, False)
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
  csvPath = modelName + ".csv"
  num_repeats = get_num_repeats(modelName)
  return RunSingleTest_Treebeard_AutoTuningHeuristic(modelName, csvPath, num_repeats)

#############################################################
### Hummingbird Total Time Benchmarking Utils
#############################################################
def RunSingleTest_Hummingbird(modelJSONPath, csvPath, num_repeats, regressor=True) -> float:
  num_batches, inputs = construct_inputs(csvPath, batchSize)
  test_batch = inputs[0:batchSize, :]
  if regressor:
    xgb_model = xgb.XGBRegressor()
  else:
    xgb_model = xgb.XGBClassifier()
  xgb_model.load_model(modelJSONPath)
  # hb_model = convert(xgb_model, 'torch.jit', test_input=test_batch, device='cuda')
  # hb_model = convert(xgb_model, 'pytorch', test_input=test_batch, device='cuda')
  hb_model = convert(xgb_model, 'tvm', test_input=test_batch, device='cuda')
  # print("Model converted to PyTorch")

  start = time.time()
  for i in range(num_repeats):
    for j in range(0, num_batches):
      start_index = j*batchSize
      stop_index = start_index + batchSize
      batch = inputs[start_index:stop_index, :]
      pred = hb_model.predict(batch)

  end = time.time()
  return (end-start)/(batchSize * num_batches * num_repeats)

def RunTestOnSingleModelTestInputs_Hummingbird(modelName : str) -> None:
  csvPath = modelName + ".csv"
  num_repeats = get_num_repeats(modelName)
  return RunSingleTest_Hummingbird(modelName, csvPath, num_repeats)

#############################################################
### Benchmark Runners
#############################################################

def get_benchmark_names(dir_name):
  # get all the json files in the directory dir_name
  files = os.listdir(dir_name)
  # filter out files that end with treebeard-globals.json
  files = [f for f in files if not f.endswith("treebeard-globals.json")]
  json_files = [os.path.join(dir_name, f) for f in files if f.endswith(".json")]
  return json_files

if __name__ == "__main__":
  # Parse command line arguments 
  # --benchmarks - a json string containing a list of names of the benchmarks to run
  # --batch_sizes - a json string containing a list of batch sizes
  
  # Create an argument parser
  arg_parser = argparse.ArgumentParser(description="")
  arg_parser.add_argument("-f", "--benchmarks", help="JSON string with a list of benchmarks", type=str)
  arg_parser.add_argument("-b", "--batch_size", help="Batch size", type=int)
  arg_parser.add_argument("-n", "--num_trials", help="Number of trials to run", type=int)

  # Parse the command line arguments
  args = arg_parser.parse_args()
  # print(args.benchmarks, args.batch_size, args.num_trials, flush=True)
  benchmarks_json = json.loads(args.benchmarks)
  benchmarks = benchmarks_json["benchmarks"]
  # benchmarks = ["/home/ashwin/sosp2024/llvm-project/mlir/examples/treebeard/xgb_models/test/GPUBenchmarks/TestModel_713_1000_1024_6_713.json"]
  assert len(benchmarks) <= 3

  batch_sizes = [args.batch_size]
  
  num_trials = args.num_trials
  hb_times = []
  treebeard_times = []
  
  speedups = []

  for batchSize in batch_sizes:
    for benchmark in benchmarks:
      hb_time_func = partial(RunTestOnSingleModelTestInputs_Hummingbird, modelName=benchmark)
      hb_time = run_benchmark_function_and_return_median_time(hb_time_func, num_trials)

      treebeard_func = partial(RunTestOnSingleModelTestInputs_Treebeard_AutoTuneHeuristic, modelName=benchmark)
      treebeard_time = run_benchmark_function_and_return_median_time(treebeard_func, num_trials)
      treebeard_times.append(treebeard_time)

      speedups.append(hb_time/treebeard_time)
      
      print(benchmark, batchSize, 
            hb_time, treebeard_time,
            hb_time/treebeard_time,
            flush=True)
    # print the geometric mean of the speedups for this batch size
    print("Geomean speedup (TB vs HB):", batchSize, numpy.prod(speedups)**(1/len(speedups)))
