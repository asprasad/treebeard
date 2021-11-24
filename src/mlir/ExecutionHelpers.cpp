#include <iostream>
#include <unistd.h>
#include <libgen.h>
#include <climits>
#include <dlfcn.h>
#include <set>
#include "ExecutionHelpers.h"
#include "Dialect.h"

namespace 
{

std::string GetDebugSOPath() { 
  char exePath[PATH_MAX];
  memset(exePath, 0, sizeof(exePath)); 
  if (readlink("/proc/self/exe", exePath, PATH_MAX) == -1)
    return std::string("");
  // std::cout << "Calculated executable path : " << exePath << std::endl;
  char *execDir = dirname(exePath);
  char *buildDir = dirname(execDir);
  std::string debugSOPath = std::string(buildDir) + "/src/debug-helpers/libtreebearddebug.so";
  return debugSOPath;
}

}

namespace mlir
{
namespace decisionforest
{
llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> InferenceRunner::CreateExecutionEngine(mlir::ModuleOp module) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::registerLLVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(/*optLevel=*/ 0, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

  // Libraries that we'll pass to the ExecutionEngine for loading.
  SmallVector<StringRef, 4> executionEngineLibs;

  std::string debugSOPath;
  if (decisionforest::InsertDebugHelpers) {
    debugSOPath = GetDebugSOPath();
    // std::cout << "Calculated debug SO path : " << debugSOPath << std::endl;
    executionEngineLibs.push_back(debugSOPath.data());
  }

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(module, /*llvmModuleBuilder=*/nullptr, optPipeline, llvm::None, executionEngineLibs);
  assert(maybeEngine && "failed to construct an execution engine");
  return maybeEngine;
}

InferenceRunner::InferenceRunner(mlir::ModuleOp module, int32_t tileSize, int32_t thresholdSize, int32_t featureIndexSize) 
  :m_maybeEngine(CreateExecutionEngine(module)), m_engine(m_maybeEngine.get()), m_module(module), 
   m_tileSize(tileSize), m_thresholdSize(thresholdSize), m_featureIndexSize(featureIndexSize)
{
  InitializeLengthsArray();
  InitializeOffsetsArray();
  InitializeModelArray();
  assert(tileSize > 0);
  if (tileSize != 1)
    InitializeLUT();
}

int32_t InferenceRunner::InitializeLengthsArray() {
  // TODO The ForestJSONReader class needs to provide an interface to iterate over tile sizes and bit widths
  // We need to construct the name of the getter function based on those. 
  auto& engine = m_engine;
  LengthMemrefType lengthMemref;
  void *args[] = { &lengthMemref };
  auto invocationResult = engine->invokePacked("Get_lengths", args);
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }
  // std::cout << "Length memref length : " << lengthMemref.lengths[0] << std::endl;

  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(lengthMemref.alignedPtr, m_tileSize, m_thresholdSize, m_featureIndexSize); 
  return 0;
}

void InferenceRunner::PrintLengthsArray() {
  // TODO The ForestJSONReader class needs to provide an interface to iterate over tile sizes and bit widths
  // We need to construct the name of the getter function based on those. 
  auto& engine = m_engine;
  LengthMemrefType lengthMemref;
  void *args[] = { &lengthMemref };
  auto invocationResult = engine->invokePacked("Get_lengths", args);
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return;
  }
  std::cout << "Length memref (size : " << lengthMemref.lengths[0] << ", ";
  std::cout << "elements : {";
  for (int64_t i=0; i<lengthMemref.lengths[0]; ++i)
    std::cout << " " << lengthMemref.alignedPtr[i];
  std::cout << " })\n";

  return;
}

// TODO All the initialize methods are doing the same thing, except that the getter they're calling are different. 
// Refactor them into a shared method.

int32_t InferenceRunner::InitializeOffsetsArray() {
  // TODO The ForestJSONReader class needs to provide an interface to iterate over tile sizes and bit widths
  // We need to construct the name of the getter function based on those. 
  auto& engine = m_engine;
  OffsetMemrefType offsetMemref;
  void *args[] = { &offsetMemref };
  auto invocationResult = engine->invokePacked("Get_offsets", args);
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }
  // std::cout << "Offset memref length : " << offsetMemref.lengths[0] << std::endl;

  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(offsetMemref.alignedPtr, m_tileSize, m_thresholdSize, m_featureIndexSize);
  return 0;
}

void InferenceRunner::PrintOffsetsArray() {
  // TODO The ForestJSONReader class needs to provide an interface to iterate over tile sizes and bit widths
  // We need to construct the name of the getter function based on those. 
  auto& engine = m_engine;
  OffsetMemrefType offsetMemref;
  void *args[] = { &offsetMemref };
  auto invocationResult = engine->invokePacked("Get_offsets", args);
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return;
  }
  std::cout << "Offset memref (size : " << offsetMemref.lengths[0] << ", ";
  std::cout << "elements : {";
  for (int64_t i=0; i<offsetMemref.lengths[0]; ++i)
    std::cout << " " << offsetMemref.alignedPtr[i];
  std::cout << " })\n";

  return;
}

template<typename ThresholdType, typename FeatureIndexType>
int32_t InferenceRunner::CallInitMethod() {
  std::vector<ThresholdType> thresholds;
  std::vector<FeatureIndexType> featureIndices;
  std::vector<int32_t> tileShapeIDs;
  mlir::decisionforest::ForestJSONReader::GetInstance().GetModelValues(m_tileSize, m_thresholdSize, m_featureIndexSize, thresholds, featureIndices, tileShapeIDs);

  Memref<ThresholdType, 1> thresholdsMemref{thresholds.data(), thresholds.data(), 0, {(int64_t)thresholds.size()}, 1};
  Memref<FeatureIndexType, 1> featureIndexMemref{featureIndices.data(), featureIndices.data(), 0, {(int64_t)featureIndices.size()}, 1};
  Memref<int32_t, 1> tileShapeIDMemref{tileShapeIDs.data(), tileShapeIDs.data(), 0, {(int64_t)tileShapeIDs.size()}, 1};
  int32_t returnValue = -1;

  void* args[] = {
    &thresholdsMemref.bufferPtr, &thresholdsMemref.alignedPtr, &thresholdsMemref.offset, &thresholdsMemref.lengths[0], &thresholdsMemref.strides[0],
    &featureIndexMemref.bufferPtr, &featureIndexMemref.alignedPtr, &featureIndexMemref.offset, &featureIndexMemref.lengths[0], &featureIndexMemref.strides[0],
    &tileShapeIDMemref.bufferPtr, &tileShapeIDMemref.alignedPtr, &tileShapeIDMemref.offset, &tileShapeIDMemref.lengths[0], &tileShapeIDMemref.strides[0],
    &returnValue
  };
  auto& engine = m_engine;
  auto invocationResult = engine->invokePacked("Init_model", args);
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }
  assert(returnValue == 0);
  return 0;
}

int32_t InferenceRunner::InitializeModelArray() {
  // TODO The ForestJSONReader class needs to provide an interface to iterate over tile sizes and bit widths
  // We need to construct the name of the getter function based on those. 
  if (false) {
    auto& engine = m_engine;
    // TODO this type doesn't really matter, but its not right. Maybe just use some dummy type here?
    Memref<TileType_Packed<double, int32_t, 1>, 1> modelMemref;
    void *args[] = { &modelMemref };
    auto invocationResult = engine->invokePacked("Get_model", args);
    if (invocationResult) {
      llvm::errs() << "JIT invocation failed\n";
      return -1;
    }
    // std::cout << "Model memref length : " << modelMemref.lengths[0] << std::endl;

    std::vector<int32_t> offsets(mlir::decisionforest::ForestJSONReader::GetInstance().GetNumberOfTrees(), -1);
    mlir::decisionforest::ForestJSONReader::GetInstance().InitializeBuffer(modelMemref.alignedPtr, m_tileSize, m_thresholdSize, m_featureIndexSize, offsets);
  }
  else {
    if (m_thresholdSize == 64) {
      if (m_featureIndexSize == 32) {
        return CallInitMethod<double, int32_t>();
      }
      else if (m_featureIndexSize == 16) {
        return CallInitMethod<double, int16_t>();
      }
      else if (m_featureIndexSize == 8) {
        return CallInitMethod<double, int8_t>();
      }
      else {
        assert (false);
      }
    }
    else if (m_thresholdSize == 32) {
      if (m_featureIndexSize == 32) {
        return CallInitMethod<float, int32_t>();
      }
      else if (m_featureIndexSize == 16) {
        return CallInitMethod<float, int16_t>();
      }
      else if (m_featureIndexSize == 8) {
        return CallInitMethod<float, int8_t>();
      }
      else {
        assert (false);
      }
    }
    else {
      assert (false);
    }
  }
  return 0;
}

int32_t InferenceRunner::InitializeLUT() {
  auto& engine = m_engine;
  // TODO this type doesn't really matter, but its not right. Maybe just use some dummy type here?
  Memref<int8_t, 2> lutMemref;
  void *args[] = { &lutMemref };
  auto invocationResult = engine->invokePacked("Get_lookupTable", args);
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLookUpTable(lutMemref.alignedPtr, m_tileSize, 8); 
  return 0;
}

SharedObjectInferenceRunner::SharedObjectInferenceRunner(const std::string& soPath, int32_t tileSize, int32_t thresholdSize, int32_t featureIndexSize)
  : m_tileSize(tileSize), m_thresholdSize(thresholdSize), m_featureIndexSize(featureIndexSize)
{
  m_so = dlopen(soPath.c_str(), RTLD_NOW);
  InitializeLengthsArray();
  InitializeOffsetsArray();
  InitializeModelArray();
  assert(tileSize > 0);
  if (tileSize != 1)
    InitializeLUT();
  m_inferenceFuncPtr = dlsym(m_so, "Prediction_Function");
}

SharedObjectInferenceRunner::~SharedObjectInferenceRunner() {
  dlclose(m_so);
}

int32_t SharedObjectInferenceRunner::InitializeLengthsArray() {
  typedef LengthMemrefType (*GetLengthFunc_t)();
  auto getLengthPtr = reinterpret_cast<GetLengthFunc_t>(dlsym(m_so, "Get_lengths"));
  LengthMemrefType lengthMemref = getLengthPtr();
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(lengthMemref.alignedPtr, m_tileSize, m_thresholdSize, m_featureIndexSize); 
  return 0;
}

int32_t SharedObjectInferenceRunner::InitializeOffsetsArray() {
  typedef OffsetMemrefType (*GetOffsetsFunc_t)();
  auto getOffsetPtr = reinterpret_cast<GetOffsetsFunc_t>(dlsym(m_so, "Get_offsets"));
  auto offsetMemref = getOffsetPtr();
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(offsetMemref.alignedPtr, m_tileSize, m_thresholdSize, m_featureIndexSize);
  return 0;
}

template<typename ThresholdType, typename FeatureIndexType>
int32_t SharedObjectInferenceRunner::CallInitMethod() {
  std::vector<ThresholdType> thresholds;
  std::vector<FeatureIndexType> featureIndices;
  std::vector<int32_t> tileShapeIDs;
  mlir::decisionforest::ForestJSONReader::GetInstance().GetModelValues(m_tileSize, m_thresholdSize, m_featureIndexSize, thresholds, featureIndices, tileShapeIDs);

  std::set<int32_t> tileShapes(tileShapeIDs.begin(), tileShapeIDs.end());
  std::cout << "Number of unique tile shapes : " << tileShapes.size() << std::endl;

  typedef int32_t (*InitModelPtr_t)(ThresholdType*, ThresholdType*, int64_t, int64_t, int64_t, FeatureIndexType*, FeatureIndexType*, int64_t, int64_t, int64_t,
                                    int32_t*, int32_t*, int64_t, int64_t, int64_t);
  auto initModelPtr = reinterpret_cast<InitModelPtr_t>(dlsym(m_so, "Init_model"));

  Memref<ThresholdType, 1> thresholdsMemref{thresholds.data(), thresholds.data(), 0, {(int64_t)thresholds.size()}, 1};
  Memref<FeatureIndexType, 1> featureIndexMemref{featureIndices.data(), featureIndices.data(), 0, {(int64_t)featureIndices.size()}, 1};
  Memref<int32_t, 1> tileShapeIDMemref{tileShapeIDs.data(), tileShapeIDs.data(), 0, {(int64_t)tileShapeIDs.size()}, 1};
  initModelPtr(thresholdsMemref.bufferPtr, thresholdsMemref.alignedPtr, thresholdsMemref.offset, thresholdsMemref.lengths[0], thresholdsMemref.strides[0],
               featureIndexMemref.bufferPtr, featureIndexMemref.alignedPtr, featureIndexMemref.offset, featureIndexMemref.lengths[0], featureIndexMemref.strides[0],
               tileShapeIDMemref.bufferPtr, tileShapeIDMemref.alignedPtr, tileShapeIDMemref.offset, tileShapeIDMemref.lengths[0], tileShapeIDMemref.strides[0]);
  return 0;
}

int32_t SharedObjectInferenceRunner::InitializeModelArray() {
  if (m_thresholdSize == 64) {
    if (m_featureIndexSize == 32) {
      return CallInitMethod<double, int32_t>();
    }
    else if (m_featureIndexSize == 16) {
      return CallInitMethod<double, int16_t>();
    }
    else if (m_featureIndexSize == 8) {
      return CallInitMethod<double, int8_t>();
    }
    else {
      assert (false);
    }
  }
  else if (m_thresholdSize == 32) {
    if (m_featureIndexSize == 32) {
      return CallInitMethod<float, int32_t>();
    }
    else if (m_featureIndexSize == 16) {
      return CallInitMethod<float, int16_t>();
    }
    else if (m_featureIndexSize == 8) {
      return CallInitMethod<float, int8_t>();
    }
    else {
      assert (false);
    }
  }
  else {
    assert (false);
  }
  return 0;
}

int32_t SharedObjectInferenceRunner::InitializeLUT() {
  using LUTMemrefType = Memref<int8_t, 2>;
  typedef LUTMemrefType (*GetLUTFunc_t)();
  
  auto getLUTPtr = reinterpret_cast<GetLUTFunc_t>(dlsym(m_so, "Get_lookupTable"));
  LUTMemrefType lutMemref = getLUTPtr();
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLookUpTable(lutMemref.alignedPtr, m_tileSize, 8); 
  return 0;
}

} // decisionforest
} // mlir
