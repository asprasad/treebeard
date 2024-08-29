import os
import math
# import cudf
import time
import numpy
import pandas
import xgboost as xgb
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
  if model_name == "letters" or model_name == "covtype":
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

def RunSingleTest_Hummingbird(modelJSONPath, csvPath, num_repeats, regressor) -> float:
  num_batches, inputs = construct_inputs(csvPath, batchSize)
  test_batch = inputs[0:batchSize, :]
  if regressor:
    xgb_model = xgb.XGBRegressor()
  else:
    xgb_model = xgb.XGBClassifier()
  xgb_model.load_model(modelJSONPath)
  hb_model = convert(xgb_model, 'torch.jit', test_input=test_batch, device='cuda')
  # hb_model = convert(xgb_model, 'pytorch', test_input=test_batch, device='cuda')
  # hb_model = convert(xgb_model, 'tvm', test_input=test_batch, device='cuda')
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


def RunTestOnSingleModelTestInputs_RAPIDs(modelName : str) -> None:
  modelJSON = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json")
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
  return RunSingleTest_Hummingbird(modelJSON, csvPath, get_num_repeats(modelName), is_regressor(modelName))


if __name__ == "__main__":
  # batch_sizes = [512, 1024, 2048, 4096] 
  # benchmarks = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "higgs", "letters", "year_prediction_msd"]
  batch_sizes = [8192, 16384]
  benchmarks = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "higgs", "year_prediction_msd"]
  results = []
  for batchSize in batch_sizes:
    # print("Batch size: ", batchSize)
    for benchmark in benchmarks:
      # print("Benchmark: ", benchmark)
      hb_time = RunTestOnSingleModelTestInputs_RAPIDs(benchmark)
      print(f"{benchmark}\t{batchSize}\t", hb_time)
      result = {"benchmark": benchmark, "batchSize": batchSize, "hb_time": hb_time}
      results.append(result)
  # dataframe with the results
  df = pandas.DataFrame(results)
  print(df.to_string())
  
