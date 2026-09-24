import ctypes
import os
import tempfile
try:
  from ._runtime_api import (  # pyright: ignore[reportMissingImports]
      TreebeardAPI, TreebeardErrorCode)
except ImportError:
  from treebeard_runtime_api import TreebeardAPI, TreebeardErrorCode
import numpy
from typing import Callable, List, Optional
from enum import Enum


class TreebeardError(RuntimeError):
  pass


class UnsupportedModelError(TreebeardError):
  pass


class UnsupportedConfigurationError(TreebeardError):
  pass


class CompilationError(TreebeardError):
  pass


class RuntimeUnavailableError(TreebeardError):
  pass


def _translate_native_error(error):
  error_types = {
    TreebeardErrorCode.UNSUPPORTED_MODEL: UnsupportedModelError,
    TreebeardErrorCode.UNSUPPORTED_CONFIGURATION:
      UnsupportedConfigurationError,
    TreebeardErrorCode.COMPILATION: CompilationError,
    TreebeardErrorCode.RUNTIME: RuntimeUnavailableError,
  }
  error_code = getattr(error, "code", TreebeardErrorCode.SUCCESS)
  return error_types.get(error_code, TreebeardError)(str(error))

treebeardAPI = TreebeardAPI()
#### ---------------------------------------------------------------- ####
#### GPU Autoschedule options
#### ---------------------------------------------------------------- ####
class GPUAutoScheduleOptions:
  def __init__(self) -> None:
      self.optionsPtr = treebeardAPI.runtime_lib.CreateGPUAutoScheduleOptions()
      self.unrollTreeWalks = False
  
  def __del__(self):
      treebeardAPI.runtime_lib.DeleteGPUAutoScheduleOptions(self.optionsPtr)
  
  def NumberOfRowsPerThreadBlock(self, val : int):
    treebeardAPI.runtime_lib.Set_numRowsPerTB(self.optionsPtr, ctypes.c_int32(val))

  def NumberOfRowsPerThread(self, val : int):
    treebeardAPI.runtime_lib.Set_numRowsPerThread(self.optionsPtr, ctypes.c_int32(val))  
 
  def RowTileSize(self, val : int):
    treebeardAPI.runtime_lib.Set_rowTileSize(self.optionsPtr, ctypes.c_int32(val))
  
  def NumberOfTreeThreads(self, val : int):
    treebeardAPI.runtime_lib.Set_numTreeThreads(self.optionsPtr, ctypes.c_int32(val))
  
  def NumberOfTreesAtATime(self, val : int):
    treebeardAPI.runtime_lib.Set_numTreesAtATime(self.optionsPtr, ctypes.c_int32(val))

  def CacheRows(self, val : bool):
    treebeardAPI.runtime_lib.Set_cacheRows(self.optionsPtr, ctypes.c_int32(1 if val else 0))
  
  def CacheTrees(self, val : bool):
    treebeardAPI.runtime_lib.Set_cacheTrees(self.optionsPtr, ctypes.c_int32(1 if val else 0))

  def UnrollTreeWalks(self, val : bool):
    self.unrollTreeWalks = val
    treebeardAPI.runtime_lib.Set_unrollTreeWalks(self.optionsPtr, ctypes.c_int32(1 if val else 0))

  def TreeWalkInterleaveFactor(self, val : int):
    treebeardAPI.runtime_lib.Set_treeWalkInterleaveFactor(self.optionsPtr, ctypes.c_int32(val))
  
  def SharedReduce(self, val : bool):
    treebeardAPI.runtime_lib.Set_sharedMemoryReduce(self.optionsPtr, ctypes.c_int32(1 if val else 0))
  
  @classmethod
  def Construct(cls, num_rows_per_TB : int, num_rows_per_thread : int = 1, num_tree_threads : int = 1, 
                cache_rows : bool = False, cache_trees : bool = False, unroll_tree_walks : bool = False,
                tree_walk_interleave_factor : int = -1, shared_reduce : bool = False):
    row_tile_size : int = 1
    num_trees_at_a_time : int = 1
    options = GPUAutoScheduleOptions()
    options.NumberOfRowsPerThreadBlock(num_rows_per_TB)
    options.NumberOfRowsPerThread(num_rows_per_thread)
    options.RowTileSize(row_tile_size)
    options.NumberOfTreeThreads(num_tree_threads)
    options.NumberOfTreesAtATime(num_trees_at_a_time)
    options.CacheRows(cache_rows)
    options.CacheTrees(cache_trees)
    options.UnrollTreeWalks(unroll_tree_walks)
    options.TreeWalkInterleaveFactor(tree_walk_interleave_factor)
    options.SharedReduce(shared_reduce)
    return options

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

  def SetNumberOfFeatures(self, val : int) :
    treebeardAPI.runtime_lib.Set_numberOfFeatures(self.optionsPtr, val)

  def SetNumberOfCores(self, val : int) :
    treebeardAPI.runtime_lib.Set_numberOfCores(self.optionsPtr, val)
  
  def SetStatsProfileCSVPath(self, val : str) :
    valStr = val.encode('ascii')
    treebeardAPI.runtime_lib.Set_statsProfileCSVPath(self.optionsPtr, valStr)
  
  def SetOneTreeAtATimeSchedule(self) :
    treebeardAPI.runtime_lib.SetOneTreeAtATimeSchedule(self.optionsPtr)

  def SetCompileToGPU(self):
    treebeardAPI.runtime_lib.Set_compileToGPU(self.optionsPtr, True)

  def SetBasicGPUSchedule(self, tileSize : int) :
    treebeardAPI.runtime_lib.SetBasicGPUSchedule(self.optionsPtr, tileSize)

  def SetNumberOfParallelTreeBatches(self, val : int) :
    treebeardAPI.runtime_lib.Set_numParallelTreeBatches(self.optionsPtr, val)

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
  
  def AtomicReduce(self, index):
      treebeardAPI.Schedule_AtomicReduce(self.schedulePtr, index.indexVarPtr)

  def VectorReduce(self, index, width):
      treebeardAPI.Schedule_VectorReduce(self.schedulePtr, index.indexVarPtr, width)

  def SharedReduce(self, index):
      treebeardAPI.Schedule_SharedReduce(self.schedulePtr, index.indexVarPtr)

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
  def __init__(self, model_filepath :str, globals_file_path : str, options : CompilerOptions, gpuAutoScheduleOptions : GPUAutoScheduleOptions = None) -> None:
    if gpuAutoScheduleOptions is None:
      self.tbcontextPtr = treebeardAPI.ConstructTreebeardContext(model_filepath, globals_file_path, options.optionsPtr)
    else:
      self.tbcontextPtr = treebeardAPI.ConstructTreebeardContextFromGPUAutoscheduleOptions(model_filepath, globals_file_path, options.optionsPtr, gpuAutoScheduleOptions.optionsPtr)
  
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
    try:
      treebeardAPI.BuildHIRRepresentation(self.tbcontextPtr)
    except RuntimeError as error:
      raise _translate_native_error(error) from error

  def DumpLLVMIR(self, path: str):
    treebeardAPI.LowerToLLVMAndDumpIR(self.tbcontextPtr, path)

  def ConstructInferenceRunnerFromHIR(self):
    inferenceRunner = TreebeardInferenceRunner()
    try:
      inferenceRunner.inferenceRunner = (
          treebeardAPI.ConstructInferenceRunnerFromHIR(self.tbcontextPtr))
    except RuntimeError as error:
      raise _translate_native_error(error) from error
    inferenceRunner.rowSize = treebeardAPI.GetRowSize(inferenceRunner.inferenceRunner)
    inferenceRunner.batchSize = treebeardAPI.GetBatchSize(inferenceRunner.inferenceRunner)
    return inferenceRunner
  
  def ConstructGPUInferenceRunnerFromHIR(self):
    inferenceRunner = TreebeardInferenceRunner()
    inferenceRunner.inferenceRunner = int(treebeardAPI.runtime_lib.ConstructGPUInferenceRunnerFromHIR(self.tbcontextPtr))
    inferenceRunner.rowSize = treebeardAPI.GetRowSize(inferenceRunner.inferenceRunner)
    inferenceRunner.batchSize = treebeardAPI.GetBatchSize(inferenceRunner.inferenceRunner)
    return inferenceRunner
  
  def ConstructGPUInferenceRunnerAutoSchedule(self):
    inferenceRunner = TreebeardInferenceRunner()
    inferenceRunner.inferenceRunner = int(treebeardAPI.runtime_lib.findBestGPUScheduleAndCompileModel(self.tbcontextPtr))
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
  def FromONNXModelFile(self, onnx_model_path: str, options: CompilerOptions):
     inferenceRunner = TreebeardInferenceRunner()
     onnx_model_path = onnx_model_path.encode('ascii')
     
     inferenceRunner.inferenceRunner = treebeardAPI.runtime_lib.CreateInferenceRunnerForONNXModel(onnx_model_path, options.optionsPtr)
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

  @classmethod
  def GPUInferenceRunnerFromTBContext(self, tbContext):
    inferenceRunner = TreebeardInferenceRunner()
    inferenceRunnerPtr = treebeardAPI.runtime_lib.ConstructGPUInferenceRunnerFromTBContext(tbContext.tbcontextPtr)
    inferenceRunner.inferenceRunner = int(inferenceRunnerPtr)
    inferenceRunner.rowSize = treebeardAPI.GetRowSize(inferenceRunner.inferenceRunner)
    inferenceRunner.batchSize = treebeardAPI.GetBatchSize(inferenceRunner.inferenceRunner)
    return inferenceRunner
  
  @classmethod
  def AutoScheduleGPUInferenceRunnerFromTBContext(self, tbContext):
    inferenceRunner = tbContext.ConstructGPUInferenceRunnerAutoSchedule()
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
    try:
      self.treebeardAPI.RunInferenceOnMultipleBatches(
          self.inferenceRunner, inputs.ctypes.data_as(ctypes.c_void_p),
          results.ctypes.data_as(ctypes.c_void_p), numRows)
    except RuntimeError as error:
      raise _translate_native_error(error) from error
    return results

#### ---------------------------------------------------------------- ####
#### High-level compilation API
#### ---------------------------------------------------------------- ####
class TreebeardKernel:
  def __init__(self, inference_runner, compilation_context, model_artifacts,
               input_type, result_type) -> None:
    self._inference_runner = inference_runner
    self._compilation_context = compilation_context
    self._model_artifacts = model_artifacts
    self.input_type = numpy.dtype(input_type)
    self.result_type = numpy.dtype(result_type)
    self.batch_size = inference_runner.batchSize
    self.num_features = inference_runner.rowSize

  def close(self):
    self._inference_runner = None
    self._compilation_context = None
    if self._model_artifacts is not None:
      self._model_artifacts.cleanup()
      self._model_artifacts = None

  def __enter__(self):
    return self

  def __exit__(self, exception_type, exception_value, traceback):
    self.close()

  def __del__(self):
    self.close()

  def run(self, inputs):
    if self._inference_runner is None:
      raise RuntimeError("kernel is closed")
    inputs = numpy.asarray(inputs, dtype=self.input_type, order="C")
    if inputs.ndim != 2:
      raise ValueError("inputs must be a two-dimensional array")
    if inputs.shape[1] != self.num_features:
      raise ValueError(
          f"expected {self.num_features} features, got {inputs.shape[1]}")
    if inputs.shape[0] == 0:
      return numpy.empty((0,), dtype=self.result_type)

    num_rows = inputs.shape[0]
    padded_rows = ((num_rows + self.batch_size - 1) //
                   self.batch_size) * self.batch_size
    if padded_rows != num_rows:
      padded_inputs = numpy.zeros(
          (padded_rows, self.num_features), dtype=self.input_type)
      padded_inputs[:num_rows] = inputs
      inputs = padded_inputs

    results = self._inference_runner.RunInferenceOnMultipleBatches(
        inputs, self.result_type)
    return results[:num_rows]


def _materialize_model(model, artifacts):
  if isinstance(model, (str, os.PathLike)):
    model_path = os.fspath(model)
    if not os.path.isfile(model_path):
      raise FileNotFoundError(model_path)
    extension = os.path.splitext(model_path)[1].lower()
    return (model_path, "onnx_file" if extension == ".onnx" else
            "xgboost_json", None)

  if hasattr(model, "get_booster"):
    model = model.get_booster()
  if hasattr(model, "save_model"):
    model_path = os.path.join(artifacts.name, "model.json")
    model.save_model(model_path)
    return model_path, "xgboost_json", None
  if hasattr(model, "SerializeToString"):
    model_path = os.path.join(artifacts.name, "model.onnx")
    with open(model_path, "wb") as model_file:
      model_file.write(model.SerializeToString())
    try:
      num_features = model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
    except (AttributeError, IndexError):
      num_features = 0
    return model_path, "onnx_file", num_features if num_features > 0 else None

  raise TypeError(
      "model must be an XGBoost estimator/booster, an ONNX model, or a model path")


def compile(model, options: Optional[CompilerOptions] = None, *,
            batch_size: int = 1024, tile_size: int = 1,
            target: str = "cpu",
            schedule: Optional[Callable[[Schedule], None]] = None,
            representation: str = "sparse",
            gpu_auto_schedule_options: Optional[GPUAutoScheduleOptions] = None,
            input_type=numpy.float32, result_type=numpy.float32):
  if target not in ("cpu", "gpu"):
    raise ValueError("target must be 'cpu' or 'gpu'")
  if batch_size <= 0:
    raise ValueError("batch_size must be positive")
  if gpu_auto_schedule_options is not None and target != "gpu":
    raise ValueError("gpu_auto_schedule_options requires target='gpu'")
  if gpu_auto_schedule_options is not None and schedule is not None:
    raise ValueError("GPU auto-scheduling cannot be combined with a schedule")
  if options is None:
    options = CompilerOptions(batch_size, tile_size)

  artifacts = tempfile.TemporaryDirectory(prefix="treebeard-")
  try:
    model_path, input_filetype, num_features = _materialize_model(
        model, artifacts)
    if num_features is not None:
      options.SetNumberOfFeatures(num_features)
    globals_path = os.path.join(artifacts.name, "model.treebeard-globals.json")
    context = TreebeardContext(
        model_path, globals_path, options, gpu_auto_schedule_options)
    context.SetRepresentationType(representation)
    context.SetInputFiletype(input_filetype)

    if gpu_auto_schedule_options is not None:
      runner = TreebeardInferenceRunner.GPUInferenceRunnerFromTBContext(context)
    else:
      context.BuildHIRRepresentation()
      if schedule is not None:
        schedule(context.GetSchedule())
      if target == "gpu":
        runner = context.ConstructGPUInferenceRunnerFromHIR()
      else:
        runner = context.ConstructInferenceRunnerFromHIR()
    return TreebeardKernel(
        runner, context, artifacts, input_type, result_type)
  except Exception:
    artifacts.cleanup()
    raise

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

def SetEnableMeasureGpuKernelTime(val):
  treebeardAPI.runtime_lib.SetEnableMeasureGpuKernelTime(val)

def SetNumberOfKernelRuns(val):
  treebeardAPI.runtime_lib.SetNumberOfKernelRuns(val)

def GetGPUKernelExecutionTime(inferenceRunner):
  return treebeardAPI.runtime_lib.GetGPUKernelExecutionTime(inferenceRunner.inferenceRunner)
