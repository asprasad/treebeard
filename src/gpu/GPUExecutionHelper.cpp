#ifdef TREEBEARD_GPU_SUPPORT

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

GPUInferenceRunner::~GPUInferenceRunner() { 
  m_serializer->CleanupBuffers();
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
#endif
   
  std::string libCudaRuntimePath = std::string(LLVM_LIB_DIR) + std::string("lib/libmlir_cuda_runtime.so");
  assert (FileExists(libCudaRuntimePath)); 
  executionEngineLibs.push_back(libCudaRuntimePath);

  std::string libMLIRRunnerUtilsPath = std::string(LLVM_LIB_DIR) + std::string("lib/libmlir_runner_utils.so");
  assert (FileExists(libMLIRRunnerUtilsPath)); 
  executionEngineLibs.push_back(libMLIRRunnerUtilsPath);

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

#endif // TREEBEARD_GPU_SUPPORT