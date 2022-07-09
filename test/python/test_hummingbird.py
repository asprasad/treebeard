import os
import numpy
import pandas
import math
import time
from hummingbird.ml import convert
import hummingbird.ml
import xgboost as xgb
from sklearn.metrics import mean_squared_error

run_parallel=True
batchSize=20000
num_repeats=1000
num_cores=16

def RunTestOnSingleModelTestInputs_Hummingbird_Torch(model, regressor, n_classes):
  if model=="letters":
      return -1;
  csvPath = "/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/xgb_models/" + model + "_xgb_model_save.json.test.sampled.csv"
  if regressor:
    xgb_model = xgb.XGBRegressor()
  else:
    xgb_model = xgb.XGBClassifier()
  
  xgb_model.load_model("/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/xgb_models/" + model + "_xgb_model_save.json")
  if not regressor:
    # xgb_model.get_booster().set_attr(n_classes=str(n_classes))
    xgb_model.n_classes_ = n_classes
  extra_config = dict()
  # extra_config[hummingbird.ml.hummingbird_constants.TREE_IMPLEMENTATION] = "perf_tree_trav"
  extra_config[hummingbird.ml.hummingbird_constants.BATCH_SIZE] = batchSize
  if not run_parallel:
    extra_config[hummingbird.ml.hummingbird_constants.N_THREADS] = 1
  else:
    extra_config[hummingbird.ml.hummingbird_constants.N_THREADS] = num_cores

  # pred = hb_model.predict([[-6.609820516539100410e+00,9.140594367590601621e+00,-8.452160840328915015e+00,-9.568673014533102261e+00,-6.162647346497802658e-01,-6.367367872963118458e+00,1.840899958894654631e+00,-5.351562295568266237e+00]])
  # print(pred)

  data_df = pandas.read_csv(csvPath, header=None)
  full_test_array = numpy.array(data_df) # numpy.genfromtxt(csvPath, ',')

  # x_numpy = full_test_array[:, 0:numFeatures]
  num_batches = int(math.floor(full_test_array.shape[0]/batchSize))
  numRows = int(math.floor(full_test_array.shape[0]/batchSize) * batchSize)
  
  inputs = numpy.array(full_test_array[0:numRows, :-1], numpy.float32)
  # expected_preds = full_test_array[0:numRows, full_test_array.shape[1]-1]
  # repeated_inputs = numpy.tile(inputs, (num_repeats, 1))
  hb_model = convert(xgb_model, 'pytorch', test_input=inputs[0:batchSize, :],  extra_config=extra_config)
  
  start = time.time()
  for i in range(num_repeats):
    # pred = booster.inplace_predict(inputs, validate_features=False)
    for j in range(0, num_batches):
      start_index = j*batchSize
      stop_index = start_index + batchSize
      batch = inputs[start_index:stop_index, :]
      pred = hb_model.predict(batch)
      # print(pred)
      # print(mean_squared_error(pred, expected_preds[start_index:stop_index])) 
  end = time.time()
  # print("(", end - start, "s )")
  print(model, " (Torch): ", ((end-start)/(batchSize * num_batches * num_repeats)))
  return ((end-start)/(batchSize * num_batches * num_repeats))

def RunTestOnSingleModelTestInputs_Hummingbird_TVM(model, regressor, n_classes):
  csvPath = "/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/xgb_models/" + model + "_xgb_model_save.json.test.sampled.csv"

  if regressor:
    xgb_model = xgb.XGBRegressor()
  else:
    xgb_model = xgb.XGBClassifier()
  
  xgb_model.load_model("/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/xgb_models/" + model + "_xgb_model_save.json")
  if not regressor:
    # xgb_model.get_booster().set_attr(n_classes=str(n_classes))
    xgb_model.n_classes_ = n_classes
  xgb_model.load_model("/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/xgb_models/" + model + "_xgb_model_save.json")

  extra_config = dict()
  # extra_config[hummingbird.ml.hummingbird_constants.TREE_IMPLEMENTATION] = "perf_tree_trav"
  extra_config[hummingbird.ml.hummingbird_constants.BATCH_SIZE] = batchSize
  if not run_parallel:
    extra_config[hummingbird.ml.hummingbird_constants.N_THREADS] = 1
  else:
    extra_config[hummingbird.ml.hummingbird_constants.N_THREADS] = num_cores

  # pred = hb_model.predict([[-6.609820516539100410e+00,9.140594367590601621e+00,-8.452160840328915015e+00,-9.568673014533102261e+00,-6.162647346497802658e-01,-6.367367872963118458e+00,1.840899958894654631e+00,-5.351562295568266237e+00]])
  # print(pred)

  data_df = pandas.read_csv(csvPath, header=None)
  full_test_array = numpy.array(data_df) # numpy.genfromtxt(csvPath, ',')

  # x_numpy = full_test_array[:, 0:numFeatures]
  if full_test_array.shape[0] >= batchSize:
    print("Not repeating : ", full_test_array.shape[0])
    num_batches = int(math.floor(full_test_array.shape[0]/batchSize))
    numRows = int(math.floor(full_test_array.shape[0]/batchSize) * batchSize)

    inputs = numpy.array(full_test_array[0:numRows, :-1], numpy.float32)
    # expected_preds = full_test_array[0:numRows, full_test_array.shape[1]-1]
    # repeated_inputs = numpy.tile(inputs, (num_repeats, 1))
  else:
    num_batches = 1
    num_repeats_array = math.ceil(batchSize/full_test_array.shape[0])
    print("Repeating input ", num_repeats_array)
    inputs = numpy.tile(full_test_array[:, :-1], (num_repeats_array, 1))
    print(inputs.shape)
  
  hb_model = convert(xgb_model, 'tvm', test_input=inputs[0:batchSize, :], extra_config=extra_config)
  print("Process ID : ", os.getpid())
  input("Attach profiler and press any key")
  start = time.time()
  for i in range(num_repeats):
    # pred = booster.inplace_predict(inputs, validate_features=False)
    for j in range(0, num_batches):
      start_index = j*batchSize
      stop_index = start_index + batchSize
      batch = inputs[start_index:stop_index, :]
      pred = hb_model.predict(batch)
      # print(pred)
      # print(mean_squared_error(pred, expected_preds[start_index:stop_index])) 

  end = time.time()
  input("Detach profiler and press any key")
  # print("(", end - start, "s )")
  print(model, " (TVM): ", ((end-start)/(batchSize * num_batches * num_repeats)))
  return ((end-start)/(batchSize * num_batches * num_repeats))

modelNames = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "letters", "higgs", "year_prediction_msd"]
regression = [True, False, False, False, False, False, False, True]
n_classes = [0, 2, 2, 8, 2, 26, 2, 0]

# modelNames = ["letters"]
# regression = [False]
# n_classes = [26]

hb_torch_times = []
hb_tvm_times = []

for model, reg, n_class in zip(modelNames, regression, n_classes):
  hb_tvm_time = RunTestOnSingleModelTestInputs_Hummingbird_TVM(model, reg, n_class)
  hb_tvm_times.append(hb_tvm_time)
  
  hb_torch_time = 1 # RunTestOnSingleModelTestInputs_Hummingbird_Torch(model, reg, n_class)
  hb_torch_times.append(hb_torch_time)

print (hb_tvm_times)
print (hb_torch_times)
