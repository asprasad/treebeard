#include <iostream>
#include <unistd.h>
#include <libgen.h>
#include <climits>
#include <dlfcn.h>
#include <set>
#include "GPUExecutionHelper.h"
#include "Dialect.h"
#include "Logger.h"

namespace 
{

bool FileExists(const std::string& filename) {
  std::ifstream fin(filename);
  return fin.good();
}

} // anonymous namespace

namespace mlir
{
namespace decisionforest
{
void GPUInferenceRunner::Init() {
  InitializeLengthsArray();
  InitializeOffsetsArray();
  InitializeModelArray();
  InitializeClassInformation();
  assert(m_tileSize > 0);
  if (m_tileSize != 1) {
    InitializeLUT();
    if (decisionforest::UseSparseTreeRepresentation)
      InitializeLeafArrays();
  }
  m_inferenceFuncPtr = GetFunctionAddress("Prediction_Function");
}

int32_t GPUInferenceRunner::InitializeLengthsArray() {
  typedef LengthMemrefType (*InitLengthFunc_t)(int64_t*, int64_t*, int64_t, int64_t, int64_t);
  auto initLengthPtr = reinterpret_cast<InitLengthFunc_t>(GetFunctionAddress("Init_Lengths"));
  
  std::vector<int64_t> lengths(mlir::decisionforest::ForestJSONReader::GetInstance().GetNumberOfTrees(), -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(lengths.data(), m_tileSize, m_thresholdSize, m_featureIndexSize);
  LengthMemrefType lengthsMemref{lengths.data(), lengths.data(), 0, {static_cast<int64_t>(lengths.size())}, {1}};
  m_lengthsMemref = initLengthPtr(lengthsMemref.bufferPtr, lengthsMemref.alignedPtr, lengthsMemref.offset, lengthsMemref.lengths[0], lengthsMemref.strides[0]);

  return 0;
}

int32_t GPUInferenceRunner::InitializeOffsetsArray() {
  typedef LengthMemrefType (*InitLengthFunc_t)(int64_t*, int64_t*, int64_t, int64_t, int64_t);
  auto initOffsetPtr = reinterpret_cast<InitLengthFunc_t>(GetFunctionAddress("Init_Offsets"));
  
  std::vector<int64_t> offsets(mlir::decisionforest::ForestJSONReader::GetInstance().GetNumberOfTrees(), -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(offsets.data(), m_tileSize, m_thresholdSize, m_featureIndexSize);
  LengthMemrefType offsetsMemref{offsets.data(), offsets.data(), 0, {static_cast<int64_t>(offsets.size())}, {1}};
  m_offsetsMemref = initOffsetPtr(offsetsMemref.bufferPtr, offsetsMemref.alignedPtr, offsetsMemref.offset, offsetsMemref.lengths[0], offsetsMemref.strides[0]);

  return 0;
}

template<typename ThresholdType, typename FeatureIndexType, typename TileShapeType>
int32_t GPUInferenceRunner::ResolveChildIndexType() {
  if (!decisionforest::UseSparseTreeRepresentation)
    return CallInitMethod<ThresholdType, FeatureIndexType, TileShapeType, int32_t>();
  else {
    auto childIndexBitWidth = mlir::decisionforest::ForestJSONReader::GetInstance().GetChildIndexBitWidth();
    assert (childIndexBitWidth > 0);
    if (childIndexBitWidth == 8)
      return CallInitMethod<ThresholdType, FeatureIndexType, TileShapeType, int8_t>();
    else if (childIndexBitWidth == 16)
      return CallInitMethod<ThresholdType, FeatureIndexType, TileShapeType, int16_t>();
    else if (childIndexBitWidth == 32)
      return CallInitMethod<ThresholdType, FeatureIndexType, TileShapeType, int32_t>();
    else
      assert (false && "Unsupported child index bit width");
    return -1;
  }
}

template<typename ThresholdType, typename FeatureIndexType>
int32_t GPUInferenceRunner::ResolveTileShapeType() {
  auto tileShapeBitWidth = mlir::decisionforest::ForestJSONReader::GetInstance().GetTileShapeBitWidth();
  if (tileShapeBitWidth == 8)
    return ResolveChildIndexType<ThresholdType, FeatureIndexType, int8_t>();
  else if (tileShapeBitWidth == 16)
    return ResolveChildIndexType<ThresholdType, FeatureIndexType, int16_t>();
  else if (tileShapeBitWidth == 32)
    return ResolveChildIndexType<ThresholdType, FeatureIndexType, int32_t>();
  else
    assert (false && "Unsupported tile shape bit width");
  return -1;
}


template<typename ThresholdType, typename FeatureIndexType, typename TileShapeType, typename ChildIndexType>
int32_t GPUInferenceRunner::CallInitMethod() {
  std::vector<ThresholdType> thresholds;
  std::vector<FeatureIndexType> featureIndices;
  std::vector<TileShapeType> tileShapeIDs;
  std::vector<ChildIndexType> childIndices;

  if (!decisionforest::UseSparseTreeRepresentation)
    mlir::decisionforest::ForestJSONReader::GetInstance().GetModelValues(m_tileSize, m_thresholdSize, m_featureIndexSize, thresholds, featureIndices, tileShapeIDs);
  else {
    assert (decisionforest::UseSparseTreeRepresentation);
    mlir::decisionforest::ForestJSONReader::GetInstance().GetModelValues(m_tileSize, m_thresholdSize, m_featureIndexSize, 
                                                                         thresholds, featureIndices, tileShapeIDs, childIndices);
  }

  Memref<ThresholdType, 1> thresholdsMemref{thresholds.data(), thresholds.data(), 0, {(int64_t)thresholds.size()}, 1};
  Memref<FeatureIndexType, 1> featureIndexMemref{featureIndices.data(), featureIndices.data(), 0, {(int64_t)featureIndices.size()}, 1};
  Memref<TileShapeType, 1> tileShapeIDMemref{tileShapeIDs.data(), tileShapeIDs.data(), 0, {(int64_t)tileShapeIDs.size()}, 1};
  
  if (!decisionforest::UseSparseTreeRepresentation) {
    typedef ModelMemrefType (*InitModelPtr_t)(ThresholdType*, ThresholdType*, int64_t, int64_t, int64_t, FeatureIndexType*, FeatureIndexType*, int64_t, int64_t, int64_t,
                                      TileShapeType*, TileShapeType*, int64_t, int64_t, int64_t);
    auto initModelPtr = reinterpret_cast<InitModelPtr_t>(GetFunctionAddress("Init_Model"));

    m_modelMemref = initModelPtr(thresholdsMemref.bufferPtr, thresholdsMemref.alignedPtr, thresholdsMemref.offset, thresholdsMemref.lengths[0], thresholdsMemref.strides[0],
                featureIndexMemref.bufferPtr, featureIndexMemref.alignedPtr, featureIndexMemref.offset, featureIndexMemref.lengths[0], featureIndexMemref.strides[0],
                tileShapeIDMemref.bufferPtr, tileShapeIDMemref.alignedPtr, tileShapeIDMemref.offset, tileShapeIDMemref.lengths[0], tileShapeIDMemref.strides[0]);
  }
  else {
    assert (decisionforest::UseSparseTreeRepresentation);
    Memref<ChildIndexType, 1> childIndexMemref{childIndices.data(), childIndices.data(), 0, {(int64_t)childIndices.size()}, 1};
    typedef ModelMemrefType (*InitModelPtr_t)(ThresholdType*, ThresholdType*, int64_t, int64_t, int64_t, FeatureIndexType*, FeatureIndexType*, int64_t, int64_t, int64_t,
                                      TileShapeType*, TileShapeType*, int64_t, int64_t, int64_t, ChildIndexType*, ChildIndexType*, int64_t, int64_t, int64_t);
    auto initModelPtr = reinterpret_cast<InitModelPtr_t>(GetFunctionAddress("Init_Model"));

    m_modelMemref = initModelPtr(thresholdsMemref.bufferPtr, thresholdsMemref.alignedPtr, thresholdsMemref.offset, thresholdsMemref.lengths[0], thresholdsMemref.strides[0],
                featureIndexMemref.bufferPtr, featureIndexMemref.alignedPtr, featureIndexMemref.offset, featureIndexMemref.lengths[0], featureIndexMemref.strides[0],
                tileShapeIDMemref.bufferPtr, tileShapeIDMemref.alignedPtr, tileShapeIDMemref.offset, tileShapeIDMemref.lengths[0], tileShapeIDMemref.strides[0],
                childIndexMemref.bufferPtr, childIndexMemref.alignedPtr, childIndexMemref.offset, childIndexMemref.lengths[0], childIndexMemref.strides[0]);
  }
  // TODO We expect to log the size of model memref etc here. But we can't do it yet 
  // for GPU code because we can't 
  // if (TreeBeard::Logging::loggingOptions.logGenCodeStats) {
  //   TreeBeard::Logging::Log("Model memref size : " + std::to_string(returnValue));
  //   std::set<int32_t> tileShapes(tileShapeIDs.begin(), tileShapeIDs.end());
  //   TreeBeard::Logging::Log("Number of unique tile shapes : " + std::to_string(tileShapes.size()));
  // }
  return 0;
}

int32_t GPUInferenceRunner::InitializeModelArray() {
  // TODO The ForestJSONReader class needs to provide an interface to iterate over tile sizes and bit widths
  // We need to construct the name of the getter function based on those. 
  if (m_thresholdSize == 64) {
    if (m_featureIndexSize == 32) {
      return ResolveTileShapeType<double, int32_t>();
    }
    else if (m_featureIndexSize == 16) {
      return ResolveTileShapeType<double, int16_t>();
    }
    else if (m_featureIndexSize == 8) {
      return ResolveTileShapeType<double, int8_t>();
    }
    else {
      assert (false);
    }
  }
  else if (m_thresholdSize == 32) {
    if (m_featureIndexSize == 32) {
      return ResolveTileShapeType<float, int32_t>();
    }
    else if (m_featureIndexSize == 16) {
      return ResolveTileShapeType<float, int16_t>();
    }
    else if (m_featureIndexSize == 8) {
      return ResolveTileShapeType<float, int8_t>();
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

int32_t GPUInferenceRunner::InitializeLUT() {
  // using LUTMemrefType = Memref<int8_t, 2>;
  // typedef LUTMemrefType (*GetLUTFunc_t)();
  
  // auto getLUTPtr = reinterpret_cast<GetLUTFunc_t>(GetFunctionAddress("Get_lookupTable"));
  // LUTMemrefType lutMemref = getLUTPtr();
  // mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLookUpTable(lutMemref.alignedPtr, m_tileSize, 8); 
  return 0;
}

int32_t GPUInferenceRunner::InitializeLeafArrays() {
  // {
  //   // Initialize the leaf values
  //   typedef Memref<double, 1> (*GetLeavesFunc_t)();
  //   auto getLeavesPtr = reinterpret_cast<GetLeavesFunc_t>(GetFunctionAddress("Get_leaves"));
  //   auto leavesMemref = getLeavesPtr();
  //   mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLeaves(leavesMemref.alignedPtr, m_tileSize, m_thresholdSize, m_featureIndexSize); 
  // }
  // {
  //   // Initialize leaf offsets
  //   typedef OffsetMemrefType (*GetOffsetsFunc_t)();
  //   auto getOffsetPtr = reinterpret_cast<GetOffsetsFunc_t>(GetFunctionAddress("Get_leavesOffsets"));
  //   auto offsetMemref = getOffsetPtr();
  //   mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLeavesOffsetBuffer(offsetMemref.alignedPtr, m_tileSize, m_thresholdSize, m_featureIndexSize);
  // }
  // {
  //   typedef LengthMemrefType (*GetLengthFunc_t)();
  //   auto getLengthPtr = reinterpret_cast<GetLengthFunc_t>(GetFunctionAddress("Get_leavesLengths"));
  //   LengthMemrefType lengthMemref = getLengthPtr();
  //   mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLeavesLengthBuffer(lengthMemref.alignedPtr, m_tileSize, 
  //                                                                                      m_thresholdSize, m_featureIndexSize); 
  // }
  return 0;
}

void GPUInferenceRunner::InitializeClassInformation() {
  
  // if (mlir::decisionforest::ForestJSONReader::GetInstance().GetNumberOfClasses() == 0) return;
   
  // typedef ClassMemrefType (*GetClassMemref_t)();
  // auto getClassInfoPtr = reinterpret_cast<GetClassMemref_t>(GetFunctionAddress("Get_treeClassInfo"));
  // ClassMemrefType treeClassInfo = getClassInfoPtr();

  // mlir::decisionforest::ForestJSONReader::GetInstance().InitializeClassInformation(treeClassInfo.alignedPtr, m_tileSize,
  //                                                                                  m_thresholdSize, m_featureIndexSize); 
}

llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> GPUInferenceRunner::CreateExecutionEngine(mlir::ModuleOp module) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::registerLLVMDialectTranslation(*module->getContext());
  mlir::registerOpenMPDialectTranslation(*module->getContext());
  
  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(/*optLevel=*/ 0, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

  // Libraries that we'll pass to the ExecutionEngine for loading.
  SmallVector<StringRef, 4> executionEngineLibs;

#ifdef OMP_SUPPORT
  std::string libompPath = std::string(LLVM_LIB_DIR) + std::string("lib/libomp.so");
  assert (FileExists(libompPath)); 
  executionEngineLibs.push_back(libompPath);
   
  std::string libCudaRuntimePath = std::string(LLVM_LIB_DIR) + std::string("lib/libmlir_cuda_runtime.so");
  assert (FileExists(libCudaRuntimePath)); 
  executionEngineLibs.push_back(libCudaRuntimePath);

  std::string libMLIRRunnerUtilsPath = std::string(LLVM_LIB_DIR) + std::string("lib/libmlir_runner_utils.so");
  assert (FileExists(libMLIRRunnerUtilsPath)); 
  executionEngineLibs.push_back(libMLIRRunnerUtilsPath);

#endif
  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions options{nullptr, {}, std::nullopt, executionEngineLibs};
  options.enablePerfNotificationListener = EnablePerfNotificationListener;
  auto maybeEngine = mlir::ExecutionEngine::create(module, options);
  assert(maybeEngine && "failed to construct an execution engine");
  return maybeEngine;
}

void *GPUInferenceRunner::GetFunctionAddress(const std::string& functionName) {
  auto expectedFptr = m_engine->lookup(functionName);
  if (!expectedFptr)
    return nullptr;
  auto fptr = *expectedFptr;
  return fptr;
}

} // decisionforest
} // mlir