#include <iostream>

#include "ExecutionHelpers.h"

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

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(module, /*llvmModuleBuilder=*/nullptr, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  return maybeEngine;
}

InferenceRunner::InferenceRunner(mlir::ModuleOp module) 
  :m_maybeEngine(CreateExecutionEngine(module)), m_engine(m_maybeEngine.get()), m_module(module)
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

  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(lengthMemref.alignedPtr, 1, 64, 32); 
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

  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(offsetMemref.alignedPtr, 1, 64, 32); 
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
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeBuffer(modelMemref.alignedPtr, 1, 64, 32, offsets); 
  return 0;
}

} // decisionforest
} // mlir