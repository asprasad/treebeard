import ctypes
from treebeard_runtime_api import TreebeardAPI
import numpy
from typing import List
from enum import Enum

treebeardAPI = TreebeardAPI()

#### ---------------------------------------------------------------- ####
#### Compiler options
#### ---------------------------------------------------------------- ####
class CompilerOptions:
  def __init__(self, batchSize, tileSize) -> None:
    self.optionsPtr = treebeardAPI.runtime_lib.CreateCompilerOptions()
    treebeardAPI.runtime_lib.Set_batchSize(self.optionsPtr, batchSize)
    treebeardAPI.runtime_lib.Set_tileSize(self.optionsPtr, tileSize)

  def __del__(self):
    treebeardAPI.runtime_lib.DeleteCompilerOptions(self.optionsPtr)
  
  def SetThresholdTypeWidth(self, val : int):
    treebeardAPI.runtime_lib.Set_thresholdTypeWidth(self.optionsPtr, val)

  def SetReturnTypeWidth(self, val : int) :
    treebeardAPI.runtime_lib.Set_returnTypeWidth(self.optionsPtr, val);

  def SetReturnTypeIsFloatType(self, val):
    treebeardAPI.runtime_lib.Set_returnTypeFloatType(self.optionsPtr, 1 if val else 0)

  def SetFeatureIndexTypeWidth(self, val : int) :  
    treebeardAPI.runtime_lib.Set_featureIndexTypeWidth(self.optionsPtr, val)

  def SetNodeIndexTypeWidth(self, val : int) :
    treebeardAPI.runtime_lib.Set_nodeIndexTypeWidth(self.optionsPtr, val)
      
  def SetInputElementTypeWidth(self, val : int) :
    treebeardAPI.runtime_lib.Set_inputElementTypeWidth(self.optionsPtr, val)

  def SetTileShapeBitWidth(self, val : int) :  
    treebeardAPI.runtime_lib.Set_tileShapeBitWidth(self.optionsPtr, val)

  def SetChildIndexBitWidth(self, val : int) :  
    treebeardAPI.runtime_lib.Set_childIndexBitWidth(self.optionsPtr, val)
    
  def SetMakeAllLeavesSameDepth(self, val : int) :
    treebeardAPI.runtime_lib.Set_makeAllLeavesSameDepth(self.optionsPtr, val)

  def SetReorderTreesByDepth(self, val) :
    treebeardAPI.runtime_lib.Set_reorderTreesByDepth(self.optionsPtr, 1 if val else 0)

  def SetTilingType(self, val : int) :
    treebeardAPI.runtime_lib.Set_tilingType(self.optionsPtr, val)
  
  def SetPipelineWidth(self, val : int) :
    treebeardAPI.runtime_lib.Set_pipelineSize(self.optionsPtr, val)

  def SetNumberOfCores(self, val : int) :
    treebeardAPI.runtime_lib.Set_numberOfCores(self.optionsPtr, val)
  
  def SetStatsProfileCSVPath(self, val : str) :
    valStr = val.encode('ascii')
    treebeardAPI.runtime_lib.Set_statsProfileCSVPath(self.optionsPtr, valStr)
  
  def SetOneTreeAtATimeSchedule(self) :
    treebeardAPI.runtime_lib.SetOneTreeAtATimeSchedule(self.optionsPtr)


#### ---------------------------------------------------------------- ####
#### Index Variable
#### ---------------------------------------------------------------- ####
class GPUConstruct(Enum):
  Grid = 0
  ThreadBlock = 1
  Undef = 2

class Dimension(Enum):
  X = 0
  Y = 1
  Z = 2

class IndexVariable:
  def __init__(self, name, indexVarPtr) -> None:
     self.name = name
     self.indexVarPtr = indexVarPtr
  
  def set_gpu_dimension(self, gpuConstruct: GPUConstruct, dimension: Dimension):
    treebeardAPI.runtime_lib.IndexVariable_SetGPUThreadDim(ctypes.c_int64(self.indexVarPtr), ctypes.c_int32(gpuConstruct.value), ctypes.c_int32(dimension.value))

#### ---------------------------------------------------------------- ####
#### Schedule
#### ---------------------------------------------------------------- ####
class Schedule:
  def __init__(self, schedulePtr):
    self.schedulePtr = schedulePtr
  
  def NewIndexVariable(self, name):
      indexVarPtr = treebeardAPI.Schedule_NewIndexVariable(self.schedulePtr, name)
      return IndexVariable(name, indexVarPtr)

  def Tile(self, index, outerIndex, innerIndex, tileSize):
      treebeardAPI.Schedule_Tile(self.schedulePtr, index.indexVarPtr, outerIndex.indexVarPtr, innerIndex.indexVarPtr, tileSize)

  def Reorder(self, indices: List[IndexVariable]):
    indices_array = numpy.zeros((len(indices)), numpy.int64)
    for i in range(len(indices)):
      indices_array[i] = indices[i].indexVarPtr
    treebeardAPI.Schedule_Reorder(self.schedulePtr, indices_array.ctypes.data_as(ctypes.c_void_p), len(indices))

  # def Split(self, indexPtr, firstPtr, secondPtr, splitIteration, indexMapPtr):
  #     self.runtime_lib.Schedule_Split(self.schedulePtr, indexPtr, firstPtr, secondPtr, splitIteration, indexMapPtr)

  def Specialize(self, index):
    infoPtr = treebeardAPI.runtime_lib.Schedule_Specialize(self.schedulePtr, index.indexVarPtr)
    num_iters = treebeardAPI.runtime_lib.GetSpecializationInfoNumIterations(infoPtr)
    num_entries = treebeardAPI.runtime_lib.GetSpecializationInfoNumEntries(infoPtr)
    lengths = numpy.zeros((num_iters), numpy.int64)
    entries = numpy.zeros((num_entries), numpy.int64)
    treebeardAPI.runtime_lib.GetSpecializationInfoEntries(infoPtr, lengths.ctypes.data_as(ctypes.c_void_p), entries.ctypes.data_as(ctypes.c_void_p))
    iter_maps = []
    index  = 0
    first_length = lengths[0]
    for l in lengths:
      assert l == first_length
      iter_map = dict()
      for i in range(index, index+l, 2):
        index1 = IndexVariable("", entries[i])
        index2 = IndexVariable("", entries[i+1])
        iter_map[index1] = index2
      iter_maps.append(iter_map)
      index += l
    return iter_maps

  def Pipeline(self, index, stepSize):
      treebeardAPI.Schedule_Pipeline(self.schedulePtr, index.indexVarPtr, stepSize)
  
  def Simdize(self, index):
      treebeardAPI.Schedule_Simdize(self.schedulePtr, index.indexVarPtr)

  def Parallel(self, index):
      treebeardAPI.Schedule_Parallel(self.schedulePtr, index.indexVarPtr)

  def Unroll(self, index):
      treebeardAPI.Schedule_Unroll(self.schedulePtr, index.indexVarPtr)

  def PeelWalk(self, index, numberOfIterations):
      treebeardAPI.Schedule_PeelWalk(self.schedulePtr, index.indexVarPtr, ctypes.c_int32(numberOfIterations))

  def Cache(self, index):
      treebeardAPI.Schedule_Cache(self.schedulePtr, index.indexVarPtr)

  def GetRootIndex(self):
      return IndexVariable("root", treebeardAPI.Schedule_GetRootIndex(self.schedulePtr))

  def GetBatchIndex(self):
      return IndexVariable("batch", treebeardAPI.Schedule_GetBatchIndex(self.schedulePtr))

  def GetTreeIndex(self):
      return IndexVariable("batch", treebeardAPI.Schedule_GetTreeIndex(self.schedulePtr))

  # def Schedule_PrintToString(self, schedPtr, string, strLen):
  #     return self.runtime_lib.Schedule_PrintToString(schedPtr, string, strLen)

  def GetBatchSize(self):
      return treebeardAPI.Schedule_GetBatchSize(self.schedulePtr)

  def GetForestSize(self):
      return treebeardAPI.Schedule_GetForestSize(self.schedulePtr)

  def IsDefaultSchedule(self):
      return treebeardAPI.Schedule_IsDefaultSchedule(self.schedulePtr)

  def Finalize(self):
      treebeardAPI.Schedule_Finalize(self.schedulePtr)

#### ---------------------------------------------------------------- ####
#### Treebeard Context
#### ---------------------------------------------------------------- ####
class TreebeardContext:
  def __init__(self, model_filepath :str, globals_file_path : str, options : CompilerOptions) -> None:
     self.tbcontextPtr = treebeardAPI.ConstructTreebeardContext(model_filepath, globals_file_path, options.optionsPtr)
  
  def __del__(self):
    treebeardAPI.DestroyTreebeardContext(self.tbcontextPtr)

  def SetInputFiletype(self, file_type:str) -> None:
    # print("TBContext_SetType: ", self.tbcontextPtr, type(self.tbcontextPtr))
    treebeardAPI.SetForestCreatorType(self.tbcontextPtr, file_type)

  def SetRepresentationType(self, rep_type:str) -> None:
    treebeardAPI.SetRepresentationAndSerializer(self.tbcontextPtr, rep_type)

  def GetSchedule(self):
    schedulePtr = treebeardAPI.GetScheduleFromTBContext(self.tbcontextPtr)
    return Schedule(schedulePtr)

  def BuildHIRRepresentation(self):
    treebeardAPI.runtime_lib.BuildHIRRepresentation(self.tbcontextPtr)

  def DumpLLVMIR(self, path: str):
    self.BuildHIRRepresentation()
    treebeardAPI.LowerToLLVMAndDumpIR(self.tbcontextPtr, path)

  def ConstructInferenceRunnerFromHIR(self):
    inferenceRunner = TreebeardInferenceRunner()
    inferenceRunner.inferenceRunner = int(treebeardAPI.runtime_lib.ConstructInferenceRunnerFromHIR(self.tbcontextPtr))
    inferenceRunner.rowSize = treebeardAPI.GetRowSize(inferenceRunner.inferenceRunner)
    inferenceRunner.batchSize = treebeardAPI.GetBatchSize(inferenceRunner.inferenceRunner)
    return inferenceRunner
  
  def ConstructGPUInferenceRunnerFromHIR(self):
    inferenceRunner = TreebeardInferenceRunner()
    inferenceRunner.inferenceRunner = int(treebeardAPI.runtime_lib.ConstructGPUInferenceRunnerFromHIR(self.tbcontextPtr))
    inferenceRunner.rowSize = treebeardAPI.GetRowSize(inferenceRunner.inferenceRunner)
    inferenceRunner.batchSize = treebeardAPI.GetBatchSize(inferenceRunner.inferenceRunner)
    return inferenceRunner

#### ---------------------------------------------------------------- ####
#### Inference Runner
#### ---------------------------------------------------------------- ####
class TreebeardInferenceRunner:
  def __init__(self) -> None:
    self.treebeardAPI = treebeardAPI
    self.inferenceRunner = 0
    self.rowSize = -1
    self.batchSize = -1

  @classmethod
  def FromSOFile(self, modelSOPath : str, modelGlobalsJSONPath : str) -> None:
    inferenceRunner = TreebeardInferenceRunner()
    inferenceRunner.inferenceRunner = treebeardAPI.InitializeInferenceRunner(modelSOPath, modelGlobalsJSONPath)
    inferenceRunner.rowSize = treebeardAPI.GetRowSize(inferenceRunner.inferenceRunner)
    inferenceRunner.batchSize = treebeardAPI.GetBatchSize(inferenceRunner.inferenceRunner)
    return inferenceRunner

  @classmethod
  def FromModelFile(self, modelJSONPathStr : str, profileCSVPathStr : str, options : CompilerOptions) -> None:
    inferenceRunner = TreebeardInferenceRunner()
    modelJSONPath = modelJSONPathStr.encode('ascii')
    profileCSVPath = profileCSVPathStr.encode('ascii')
  
    inferenceRunner.inferenceRunner = treebeardAPI.runtime_lib.CreateInferenceRunner(modelJSONPath, profileCSVPath, options.optionsPtr)
    inferenceRunner.rowSize = treebeardAPI.GetRowSize(inferenceRunner.inferenceRunner)
    inferenceRunner.batchSize = treebeardAPI.GetBatchSize(inferenceRunner.inferenceRunner)
    return inferenceRunner
  
  @classmethod
  def FromTBContext(self, tbContext):
    inferenceRunner = TreebeardInferenceRunner()
    treebeardAPI.runtime_lib.BuildHIRRepresentation(tbContext.tbcontextPtr)
    inferenceRunner.inferenceRunner = int(treebeardAPI.runtime_lib.ConstructInferenceRunnerFromHIR(tbContext.tbcontextPtr))
    inferenceRunner.rowSize = treebeardAPI.GetRowSize(inferenceRunner.inferenceRunner)
    inferenceRunner.batchSize = treebeardAPI.GetBatchSize(inferenceRunner.inferenceRunner)
    return inferenceRunner

  def __del__(self):
    self.treebeardAPI.DeleteInferenceRunner(self.inferenceRunner)
  
  def RunInference(self, inputs, resultType=numpy.float32):
    assert type(inputs) is numpy.ndarray
    inputs_np = inputs
    results = numpy.zeros((self.batchSize), resultType)
    self.treebeardAPI.RunInference(self.inferenceRunner, inputs_np.ctypes.data_as(ctypes.c_void_p), results.ctypes.data_as(ctypes.c_void_p))
    return results

  def RunInferenceOnMultipleBatches(self, inputs, resultType=numpy.float32):
    assert type(inputs) is numpy.ndarray
    numRows = inputs.shape[0]
    results = numpy.zeros((numRows), resultType)
    self.treebeardAPI.RunInferenceOnMultipleBatches(self.inferenceRunner, inputs.ctypes.data_as(ctypes.c_void_p), results.ctypes.data_as(ctypes.c_void_p), numRows)
    return results

#### ---------------------------------------------------------------- ####
#### Treebeard API -- Do not use these!
#### ---------------------------------------------------------------- ####
def GenerateLLVMIRForXGBoostModel(modelJSONPathStr, llvmIRPathStr, modelGlobalsJSONPathStr, options):
  modelJSONPath = modelJSONPathStr.encode('ascii')
  llvmIRPath = llvmIRPathStr.encode('ascii')
  modelGlobalsJSONPath = modelGlobalsJSONPathStr.encode('ascii')
  treebeardAPI.runtime_lib.GenerateLLVMIRForXGBoostModel(ctypes.c_char_p(modelJSONPath), ctypes.c_char_p(llvmIRPath), ctypes.c_char_p(modelGlobalsJSONPath), options.optionsPtr)

def SetEnableSparseRepresentation(val):
  treebeardAPI.runtime_lib.SetEnableSparseRepresentation(1 if val else 0)

def IsSparseRepresentationEnabled():
  return treebeardAPI.runtime_lib.IsSparseRepresentationEnabled()

def SetPeeledCodeGenForProbabilityBasedTiling(val):
  treebeardAPI.runtime_lib.SetPeeledCodeGenForProbabilityBasedTiling(1 if val else 0)

def IsPeeledCodeGenForProbabilityBasedTilingEnabled():
  return treebeardAPI.runtime_lib.IsPeeledCodeGenForProbabilityBasedTilingEnabled()
