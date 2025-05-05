import os
import sys
import pandas
import numpy
import time 

filepath = os.path.abspath(__file__)
treebeard_repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(filepath)))
sys.path.append(os.path.join(treebeard_repo_dir, 'src', 'python'))

import treebeard

def CheckEqual(a, b) -> bool:
  if type(a) is numpy.float32:
    scaledThreshold = max(abs(a), abs(b))/1e8
    threshold = max(1e-6, scaledThreshold)
    return pow(a-b, 2) < threshold
  else:
    return a==b

def CheckArraysEqual(x, y) -> bool:
  assert len(x.shape) == 1
  assert len(y.shape) == 1
  if not x.shape[0] == y.shape[0]:
    return False
  for i in range(x.shape[0]):
    if not CheckEqual(x[i], y[i]):
      print(x[i], y[i])
      return False
  return True

# Setup treebeard options
BATCH_SIZE = 200
TREE_TILE_SIZE = 8
compiler_options = treebeard.CompilerOptions(BATCH_SIZE, TREE_TILE_SIZE)

compiler_options.SetNumberOfCores(1)
compiler_options.SetMakeAllLeavesSameDepth(1) # make all leaves same depth. Enables unrolling tree walks of same depth.
compiler_options.SetReorderTreesByDepth(True) # reorder trees by depth. Enables grouping of trees by depth
compiler_options.SetPipelineWidth(8) # set pipeline width. Enables jamming of unrolled loops. Should be less than batch size.

onnx_model_path = os.path.join(treebeard_repo_dir, "onnx_models", "abalone_onnx_model_save.onnx")

csv_path = os.path.join(treebeard_repo_dir, "onnx_models", "abalone_onnx_model_save.onnx.csv") 
data_df = pandas.read_csv(csv_path, header=None)
data = numpy.array(data_df, order='C') # numpy.genfromtxt(csvPath, ',')
inputs = numpy.array(data[:, :-1], numpy.float32, order='C')
expectedOutputs = data[:, data.shape[1]-1]

print("inputs shape: ", inputs.shape)
compiler_options.SetNumberOfFeatures(inputs.shape[1]) # set number of features, needed for ONNX models
  
globalsPath = onnx_model_path + ".treebeard-globals.json"

tbContext = treebeard.TreebeardContext(onnx_model_path, globalsPath, compiler_options)
tbContext.SetRepresentationType("sparse")
tbContext.SetInputFiletype("onnx_file")

inferenceRunner = treebeard.TreebeardInferenceRunner.FromTBContext(tbContext)

start = time.time()
for i in range(0, data.shape[0], 200):
    batch = inputs[i:i+200, :]
    results = inferenceRunner.RunInference(batch, numpy.float32)
    if not CheckArraysEqual(results, expectedOutputs[i:i+200]):
      print("Failed")
      assert False
end = time.time()
print("Passed (", end - start, "s )")


