import os
import math
import cudf
import time
import numpy
import pandas
import xgboost as xgb
from cuml import ForestInference

filepath = os.path.abspath(__file__)
treebeard_repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(filepath))))

num_repeats = 50
run_parallel = False
num_tiles = 4
batchSize = 1024

def RunSingleTest_RAPIDs(modelJSONPath, csvPath) -> float:
  booster = xgb.Booster(model_file=modelJSONPath)
  booster.set_param({"predictor": "gpu_predictor"})
  # cu_model = ForestInference.load(booster)
  cu_model = ForestInference.load(filename=modelJSONPath, output_class=True, model_type='xgboost_json')

  data_df = pandas.read_csv(csvPath, header=None)
  full_test_array = numpy.array(data_df, order='C') 
  full_test_array = numpy.tile(full_test_array, (num_tiles, 1))
  num_batches = int(math.floor(full_test_array.shape[0]/batchSize))
  numRows = int(math.floor(full_test_array.shape[0]/batchSize) * batchSize)

  inputs = numpy.array(full_test_array[0:numRows, :-1], numpy.float32, order='C')

  start = time.time()
  for i in range(num_repeats):
    for j in range(0, num_batches):
      start_index = j*batchSize
      stop_index = start_index + batchSize
      # batch = cudf.DataFrame(inputs[start_index:stop_index, :])
      # pred = cu_model.predict(batch)
      # predictions_array = pred.to_array()
      batch = inputs[start_index:stop_index, :]
      batch_dmatrix = xgb.DMatrix(batch)
      pred_xgboost = booster.predict(batch_dmatrix, validate_features=False)
      pred = cu_model.predict(batch)

  end = time.time()
  return (end-start)/(batchSize * num_batches * num_repeats)


def RunTestOnSingleModelTestInputs_RAPIDs(modelName : str) -> None:
  modelJSON = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json")
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
  return RunSingleTest_RAPIDs(modelJSON, csvPath)


if __name__ == "__main__":
  RunTestOnSingleModelTestInputs_RAPIDs("covtype")
  