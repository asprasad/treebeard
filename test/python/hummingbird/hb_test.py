import argparse
import os
import math
# import cudf
import time
import numpy
import pandas
import xgboost as xgb
import treebeard
from hummingbird.ml import convert

filepath = os.path.abspath(__file__)
treebeard_repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(filepath))))

run_parallel = False
num_tiles = 4
batchSize = 1024

def get_num_repeats(model_name : str):
  if model_name == "letters":
    return 200
  else:
    return 600

def is_regressor(model_name : str):
  classification_models = ["airline", "airline-ohe", "covtype", "epsilon", "higgs", "letters"]
  if model_name in classification_models:
    return False
  else:
    return True

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

def RunSingleTest_Hummingbird(modelJSONPath, csvPath, num_repeats, regressor, backend) -> float:
  num_batches, inputs = construct_inputs(csvPath, batchSize)
  test_batch = inputs[0:batchSize, :]
  if regressor:
    xgb_model = xgb.XGBRegressor()
  else:
    xgb_model = xgb.XGBClassifier()
  xgb_model.load_model(modelJSONPath)
  hb_model = convert(xgb_model, backend, test_input=test_batch, device='cuda')

  start = time.time()
  for i in range(num_repeats):
    for j in range(0, num_batches):
      start_index = j*batchSize
      stop_index = start_index + batchSize
      batch = inputs[start_index:stop_index, :]
      pred = hb_model.predict(batch)

  end = time.time()
  return (end-start)/(batchSize * num_batches * num_repeats)

def RunTestOnSingleModelTestInputs_RAPIDs(modelName : str, backend : str) -> None:
  modelJSON = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json")
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
  return RunSingleTest_Hummingbird(modelJSON, csvPath, get_num_repeats(modelName), is_regressor(modelName), backend)

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

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Benchmarking script for RAPIDs and Treebeard")
  parser.add_argument("--num_trials", type=int, default=3, help="Number of trials to run for each benchmark")
  parser.add_argument("--batch_size", type=int, default=4096, help="Batch sizes to run")
  args = parser.parse_args()

  num_trials = args.num_trials
  # batch_sizes = [512, 1024, 2048, 4096] 
  benchmarks = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "higgs", "letters", "year_prediction_msd"]
  batch_sizes = [args.batch_size]
  # batch_sizes = [8192, 16384]
  # batch_sizes = [512, 1024, 2048, 4096, 8192, 16384] 
  # benchmarks = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "higgs", "year_prediction_msd"]
  results = []
  jit_speedups = []
  torch_speedups = []
  tvm_speedups = []
  for batchSize in batch_sizes:
    # print("Batch size: ", batchSize)
    for benchmark in benchmarks:
      # print("Benchmark: ", benchmark)
      tb_time = RunTestOnSingleModelTestInputs_Treebeard_AutoTuneHeuristic(benchmark)
      if benchmark == "letters" and batchSize >= 8192:
        hb_torch_time = 1.0
        hb_jit_time = 1.0  
      else:
        hb_torch_time = RunTestOnSingleModelTestInputs_RAPIDs(benchmark, "torch")
        hb_jit_time = RunTestOnSingleModelTestInputs_RAPIDs(benchmark, "torch.jit")
      hb_tvm_time = RunTestOnSingleModelTestInputs_RAPIDs(benchmark, "tvm")
      # print(f"{benchmark}\t{batchSize}\t", hb_torch_time, "\t", hb_jit_time, "\t", hb_tvm_time)
      # result = {"benchmark": benchmark, 
      #           "batchSize": batchSize,
      #           "tb_time": tb_time,
      #           "hb_jit_time": hb_jit_time, 
      #           "hb_torch_time": hb_torch_time,
      #           "hb_tvm_time": hb_tvm_time,
      #           "hb_jit_speedup": hb_jit_time/tb_time,
      #           "hb_torch_speedup": hb_torch_time/tb_time,
      #           "hb_tvm_speedup": hb_tvm_time/tb_time}
      # print(result)
      # results.append(result)
      if hb_jit_time != 1.0 and hb_torch_time != 1.0:
        jit_speedups.append(hb_jit_time/tb_time)
        torch_speedups.append(hb_torch_time/tb_time)
      tvm_speedups.append(hb_tvm_time/tb_time)

      print(benchmark, batchSize, hb_jit_time, hb_torch_time, hb_tvm_time, 
            tb_time, hb_jit_time/tb_time, hb_torch_time/tb_time, 
            hb_tvm_time/tb_time, flush=True)
  
  # dataframe with the results
  # df = pandas.DataFrame(results)
  # print(df.to_string())

  print(numpy.prod(jit_speedups)**(1.0/len(jit_speedups)),
        numpy.prod(torch_speedups)**(1.0/len(torch_speedups)),
        numpy.prod(tvm_speedups)**(1.0/len(tvm_speedups)), flush=True)


  
