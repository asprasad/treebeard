import os
import sys
import pandas as pd
import numpy
import math
import time

filepath = os.path.abspath(__file__)
treebeard_repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(filepath)))
treebeard_runtime_dir = os.path.join(os.path.join(treebeard_repo_dir, "src"), "python")

sys.path.append(treebeard_runtime_dir)

import treebeard

NUM_REPEATS = 1000
REPRESENTATIONS = ["gpu-sparse"]
TILE_SIZES = [1,2,4,8]
BATCH_SIZES = [512, 1024]
XGB_MODELS = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "letters", "higgs", "year_prediction_msd"]
RUN_PARALLEL=True

def ConstructGPUCompilerOptions(batch_size, tile_size, model):
    compilerOptions = treebeard.CompilerOptions(batch_size, tile_size)
    compilerOptions.SetMakeAllLeavesSameDepth(1)

    compilerOptions.SetCompileToGPU()
    compilerOptions.SetPipelineWidth(8)
    compilerOptions.SetTilingType(0) # uniform tiling
    compilerOptions.SetTileShapeBitWidth(1 if tile_size == 1 else 16)
    compilerOptions.SetChildIndexBitWidth(16)
    compilerOptions.SetBasicGPUSchedule(32)
    compilerOptions.SetThresholdTypeWidth(32) # float type
    if model == "covtype" or model == "letters":
        compilerOptions.SetReturnTypeWidth(8)
        compilerOptions.SetReturnTypeIsFloatType(False)
        
    return compilerOptions

def RunSingleTestJIT_Multibatch(modelJSONPath, csvPath, options, returnType, representation) -> float:
  data_df = pd.read_csv(csvPath, header=None)
  data = numpy.array(data_df, order='C') # numpy.genfromtxt(csvPath, ',')

  inferenceRunner = treebeard.TreebeardInferenceRunner.FromModelFile(modelJSONPath, "", options)
  batch_size = inferenceRunner.batchSize
  num_batches = int(math.floor(data.shape[0]/inferenceRunner.batchSize))
  numRows = int(math.floor(data.shape[0]/inferenceRunner.batchSize) * inferenceRunner.batchSize)
  inputs = numpy.array(data[0:numRows, :-1], numpy.float32, order='C')
  repeated_inputs = numpy.tile(inputs, (NUM_REPEATS, 1))

  start = time.time()
  for i in range(NUM_REPEATS):
    for j in range(0, num_batches):
        start_index = j*batch_size
        stop_index = start_index + batch_size
        batch = inputs[start_index:stop_index, :]
        results = inferenceRunner.RunInference(batch, returnType)
  end = time.time()

  # print(batch_size, ',', data.shape)
  return (end - start)/(batch_size * num_batches * NUM_REPEATS)

treebeard.SetEnableSparseRepresentation(True)
print('tile_size,batch_size,representation,model,time_per_batch')
for model in XGB_MODELS:
    for batch_size in BATCH_SIZES:
        for tile_size in TILE_SIZES:
            for representation in REPRESENTATIONS:
                options = ConstructGPUCompilerOptions(batch_size, tile_size, model)
                model_path = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), model + "_xgb_model_save.json")
                csv_path = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), model + "_xgb_model_save.json.test.sampled.csv")
                time_per_batch = RunSingleTestJIT_Multibatch(model_path, csv_path, options, numpy.float32, representation)
                print(tile_size, batch_size, representation, model, time_per_batch, sep=",")