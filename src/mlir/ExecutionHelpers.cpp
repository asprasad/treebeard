#include <iostream>
#include <unistd.h>
#include <libgen.h>
#include <climits>
#include <dlfcn.h>
#include <set>
#include "ExecutionHelpers.h"
#include "Dialect.h"
#include "Logger.h"

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

bool FileExists(const std::string& filename) {
  std::ifstream fin(filename);
  return fin.good();
}

}

namespace mlir
{
namespace decisionforest
{
void InferenceRunnerBase::Init() {
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

int32_t InferenceRunnerBase::InitializeLengthsArray() {
  // TODO The ForestJSONReader class needs to provide an interface to iterate over tile sizes and bit widths
  // We need to construct the name of the getter function based on those. 
  typedef LengthMemrefType (*GetLengthFunc_t)();
  auto getLengthPtr = reinterpret_cast<GetLengthFunc_t>(GetFunctionAddress("Get_lengths"));
  LengthMemrefType lengthMemref = getLengthPtr();
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(lengthMemref.alignedPtr, m_tileSize, m_thresholdSize, m_featureIndexSize); 

  return 0;
}

void InferenceRunnerBase::PrintLengthsArray() {
  // TODO The ForestJSONReader class needs to provide an interface to iterate over tile sizes and bit widths
  // We need to construct the name of the getter function based on those. 
  typedef LengthMemrefType (*GetLengthFunc_t)();
  auto getLengthPtr = reinterpret_cast<GetLengthFunc_t>(GetFunctionAddress("Get_lengths"));
  LengthMemrefType lengthMemref = getLengthPtr();
  std::cout << "Length memref (size : " << lengthMemref.lengths[0] << ", ";
  std::cout << "elements : {";
  for (int64_t i=0; i<lengthMemref.lengths[0]; ++i)
    std::cout << " " << lengthMemref.alignedPtr[i];
  std::cout << " })\n";

  return;
}

// TODO All the initialize methods are doing the same thing, except that the getter they're calling are different. 
// Refactor them into a shared method.

int32_t InferenceRunnerBase::InitializeOffsetsArray() {
  // TODO The ForestJSONReader class needs to provide an interface to iterate over tile sizes and bit widths
  // We need to construct the name of the getter function based on those. 
  typedef OffsetMemrefType (*GetOffsetsFunc_t)();
  auto getOffsetPtr = reinterpret_cast<GetOffsetsFunc_t>(GetFunctionAddress("Get_offsets"));
  auto offsetMemref = getOffsetPtr();
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(offsetMemref.alignedPtr, m_tileSize, m_thresholdSize, m_featureIndexSize);
  return 0;
}

void InferenceRunnerBase::PrintOffsetsArray() {
  // TODO The ForestJSONReader class needs to provide an interface to iterate over tile sizes and bit widths
  // We need to construct the name of the getter function based on those. 
  typedef OffsetMemrefType (*GetOffsetsFunc_t)();
  auto getOffsetPtr = reinterpret_cast<GetOffsetsFunc_t>(GetFunctionAddress("Get_offsets"));
  auto offsetMemref = getOffsetPtr();
  std::cout << "Offset memref (size : " << offsetMemref.lengths[0] << ", ";
  std::cout << "elements : {";
  for (int64_t i=0; i<offsetMemref.lengths[0]; ++i)
    std::cout << " " << offsetMemref.alignedPtr[i];
  std::cout << " })\n";

  return;
}

template<typename ThresholdType, typename FeatureIndexType, typename TileShapeType>
int32_t InferenceRunnerBase::ResolveChildIndexType() {
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
int32_t InferenceRunnerBase::ResolveTileShapeType() {
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
int32_t InferenceRunnerBase::CallInitMethod() {
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
  int32_t returnValue = -1;
  
  if (!decisionforest::UseSparseTreeRepresentation) {
    typedef int32_t (*InitModelPtr_t)(ThresholdType*, ThresholdType*, int64_t, int64_t, int64_t, FeatureIndexType*, FeatureIndexType*, int64_t, int64_t, int64_t,
                                      TileShapeType*, TileShapeType*, int64_t, int64_t, int64_t);
    auto initModelPtr = reinterpret_cast<InitModelPtr_t>(GetFunctionAddress("Init_model"));

    returnValue = initModelPtr(thresholdsMemref.bufferPtr, thresholdsMemref.alignedPtr, thresholdsMemref.offset, thresholdsMemref.lengths[0], thresholdsMemref.strides[0],
                featureIndexMemref.bufferPtr, featureIndexMemref.alignedPtr, featureIndexMemref.offset, featureIndexMemref.lengths[0], featureIndexMemref.strides[0],
                tileShapeIDMemref.bufferPtr, tileShapeIDMemref.alignedPtr, tileShapeIDMemref.offset, tileShapeIDMemref.lengths[0], tileShapeIDMemref.strides[0]);
  }
  else {
    assert (decisionforest::UseSparseTreeRepresentation);
    Memref<ChildIndexType, 1> childIndexMemref{childIndices.data(), childIndices.data(), 0, {(int64_t)childIndices.size()}, 1};
    typedef int32_t (*InitModelPtr_t)(ThresholdType*, ThresholdType*, int64_t, int64_t, int64_t, FeatureIndexType*, FeatureIndexType*, int64_t, int64_t, int64_t,
                                      TileShapeType*, TileShapeType*, int64_t, int64_t, int64_t, ChildIndexType*, ChildIndexType*, int64_t, int64_t, int64_t);
    auto initModelPtr = reinterpret_cast<InitModelPtr_t>(GetFunctionAddress("Init_model"));

    returnValue = initModelPtr(thresholdsMemref.bufferPtr, thresholdsMemref.alignedPtr, thresholdsMemref.offset, thresholdsMemref.lengths[0], thresholdsMemref.strides[0],
                featureIndexMemref.bufferPtr, featureIndexMemref.alignedPtr, featureIndexMemref.offset, featureIndexMemref.lengths[0], featureIndexMemref.strides[0],
                tileShapeIDMemref.bufferPtr, tileShapeIDMemref.alignedPtr, tileShapeIDMemref.offset, tileShapeIDMemref.lengths[0], tileShapeIDMemref.strides[0],
                childIndexMemref.bufferPtr, childIndexMemref.alignedPtr, childIndexMemref.offset, childIndexMemref.lengths[0], childIndexMemref.strides[0]);
  }
  if (TreeBeard::Logging::loggingOptions.logGenCodeStats) {
    TreeBeard::Logging::Log("Model memref size : " + std::to_string(returnValue));
    std::set<int32_t> tileShapes(tileShapeIDs.begin(), tileShapeIDs.end());
    TreeBeard::Logging::Log("Number of unique tile shapes : " + std::to_string(tileShapes.size()));
  }
  assert(returnValue != -1);
  return returnValue;
}

int32_t InferenceRunnerBase::InitializeModelArray() {
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

int32_t InferenceRunnerBase::InitializeLUT() {
  using LUTMemrefType = Memref<int8_t, 2>;
  typedef LUTMemrefType (*GetLUTFunc_t)();
  
  auto getLUTPtr = reinterpret_cast<GetLUTFunc_t>(GetFunctionAddress("Get_lookupTable"));
  LUTMemrefType lutMemref = getLUTPtr();
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLookUpTable(lutMemref.alignedPtr, m_tileSize, 8); 
  return 0;
}

int32_t InferenceRunnerBase::InitializeLeafArrays() {
  {
    // Initialize the leaf values
    typedef Memref<double, 1> (*GetLeavesFunc_t)();
    auto getLeavesPtr = reinterpret_cast<GetLeavesFunc_t>(GetFunctionAddress("Get_leaves"));
    auto leavesMemref = getLeavesPtr();
    mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLeaves(leavesMemref.alignedPtr, m_tileSize, m_thresholdSize, m_featureIndexSize); 
  }
  {
    // Initialize leaf offsets
    typedef OffsetMemrefType (*GetOffsetsFunc_t)();
    auto getOffsetPtr = reinterpret_cast<GetOffsetsFunc_t>(GetFunctionAddress("Get_leavesOffsets"));
    auto offsetMemref = getOffsetPtr();
    mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLeavesOffsetBuffer(offsetMemref.alignedPtr, m_tileSize, m_thresholdSize, m_featureIndexSize);
  }
  {
    typedef LengthMemrefType (*GetLengthFunc_t)();
    auto getLengthPtr = reinterpret_cast<GetLengthFunc_t>(GetFunctionAddress("Get_leavesLengths"));
    LengthMemrefType lengthMemref = getLengthPtr();
    mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLeavesLengthBuffer(lengthMemref.alignedPtr, m_tileSize, 
                                                                                       m_thresholdSize, m_featureIndexSize); 
  }
  return 0;
}

void InferenceRunnerBase::InitializeClassInformation() {
  
  if (mlir::decisionforest::ForestJSONReader::GetInstance().GetNumberOfClasses() == 0) return;
   
  typedef ClassMemrefType (*GetClassMemref_t)();
  auto getClassInfoPtr = reinterpret_cast<GetClassMemref_t>(GetFunctionAddress("Get_treeClassInfo"));
  ClassMemrefType treeClassInfo = getClassInfoPtr();

  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeClassInformation(treeClassInfo.alignedPtr, m_tileSize,
                                                                                   m_thresholdSize, m_featureIndexSize); 
}

// ===------------------------------------------------------=== //
// Shared object inference runner 
// ===------------------------------------------------------=== //

SharedObjectInferenceRunner::SharedObjectInferenceRunner(const std::string& modelGlobalsJSONFilePath, const std::string& soPath, int32_t tileSize, int32_t thresholdSize, int32_t featureIndexSize)
  : InferenceRunnerBase(modelGlobalsJSONFilePath, tileSize, thresholdSize, featureIndexSize)
{
  m_so = dlopen(soPath.c_str(), RTLD_NOW);
  Init();
}

SharedObjectInferenceRunner::~SharedObjectInferenceRunner() {
  dlclose(m_so);
}

void* SharedObjectInferenceRunner::GetFunctionAddress(const std::string& functionName) {
  auto fptr = dlsym(m_so, functionName.c_str());
  assert (fptr);
  return fptr;
}

// ===------------------------------------------------------=== //
// JIT inference runner 
// ===------------------------------------------------------=== //

llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> InferenceRunner::CreateExecutionEngine(mlir::ModuleOp module) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::registerLLVMDialectTranslation(*module->getContext());
  mlir::registerOpenMPDialectTranslation(*module->getContext());
  
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
#ifdef OMP_SUPPORT
  std::string libompPath = std::string(LLVM_LIB_DIR) + std::string("lib/libomp.so");
  if (!FileExists(libompPath)) {
    libompPath = "/usr/lib/llvm-10/lib/libomp.so";
  }
  executionEngineLibs.push_back(libompPath);
#endif
  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions options{nullptr, {}, llvm::None, executionEngineLibs};
  options.enablePerfNotificationListener = EnablePerfNotificationListener;
  auto maybeEngine = mlir::ExecutionEngine::create(module, options);
  assert(maybeEngine && "failed to construct an execution engine");
  return maybeEngine;
}

InferenceRunner::InferenceRunner(const std::string& modelGlobalsJSONFilePath, mlir::ModuleOp module, int32_t tileSize, int32_t thresholdSize, int32_t featureIndexSize) 
  :InferenceRunnerBase(modelGlobalsJSONFilePath, tileSize, thresholdSize, featureIndexSize),
   m_maybeEngine(CreateExecutionEngine(module)), m_engine(m_maybeEngine.get()), m_module(module)
{
  Init();
}

void *InferenceRunner::GetFunctionAddress(const std::string& functionName) {
  auto expectedFptr = m_engine->lookup(functionName);
  if (!expectedFptr)
    return nullptr;
  auto fptr = *expectedFptr;
  return fptr;
}

} // decisionforest
} // mlir
