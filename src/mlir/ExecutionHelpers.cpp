#include <iostream>
#include <unistd.h>
#include <libgen.h>
#include <climits>

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

int32_t InferenceRunner::InitializeModelArray() {
  // TODO The ForestJSONReader class needs to provide an interface to iterate over tile sizes and bit widths
  // We need to construct the name of the getter function based on those. 
  auto& engine = m_engine;
  Memref<TileType<double, int32_t, 1>, 1> modelMemref;
  void *args[] = { &modelMemref };
  auto invocationResult = engine->invokePacked("Get_model", args);
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }
  // std::cout << "Model memref length : " << modelMemref.lengths[0] << std::endl;
  std::vector<int32_t> offsets(mlir::decisionforest::ForestJSONReader::GetInstance().GetNumberOfTrees(), -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeBuffer(modelMemref.alignedPtr, m_tileSize, m_thresholdSize, m_featureIndexSize, offsets); 
  return 0;
}

} // decisionforest
} // mlir