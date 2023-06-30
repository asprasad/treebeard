#include <iostream>
#include <unistd.h>
#include <libgen.h>
#include <climits>
#include <dlfcn.h>
#include <set>
#include "ExecutionHelpers.h"
#include "Dialect.h"
#include "Logger.h"
#include "TreebeardContext.h"
#include "TiledTree.h"

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

InferenceRunnerBase::InferenceRunnerBase(std::shared_ptr<IModelSerializer> serializer,
                                         int32_t tileSize,
                                         int32_t thresholdSize,
                                         int32_t featureIndexSize)
  : m_serializer(serializer),
    m_tileSize(tileSize),
    m_thresholdSize(thresholdSize),
    m_featureIndexSize(featureIndexSize)
{ 
}

void InferenceRunnerBase::InitIntegerField(const std::string& functionName, int32_t& field) {
  using GetFunc_t = int32_t(*)();
  auto get = reinterpret_cast<GetFunc_t>(GetFunctionAddress(functionName));
  field = get();
}

void InferenceRunnerBase::Init() {
  m_serializer->InitializeBuffers(this);
  m_inferenceFuncPtr = GetFunctionAddress("Prediction_Function");
  InitIntegerField("GetBatchSize", m_batchSize);
  InitIntegerField("GetRowSize", m_rowSize);
  InitIntegerField("GetInputTypeBitWidth", m_inputElementBitWidth);
  InitIntegerField("GetReturnTypeBitWidth", m_returnTypeBitWidth);
}

int32_t InferenceRunnerBase::RunInference_CustomImpl(double *input, double *returnValue) {
  Memref<double, 2> inputs{reinterpret_cast<double*>(input),
                            reinterpret_cast<double*>(input),
                            0,
                            {m_batchSize, m_rowSize}, // lengths
                            {m_rowSize, 1} // strides
                          };
  Memref<double, 1> resultMemref{reinterpret_cast<double*>(returnValue),
                                  reinterpret_cast<double*>(returnValue),
                                  0,
                                  {m_batchSize}, //length
                                  {1}
                                };
  m_serializer->CallPredictionMethod(m_inferenceFuncPtr, inputs, resultMemref);
  return 0;
}

bool InferenceRunnerBase::SerializerHasCustomPredictionMethod() {
  return m_serializer->HasCustomPredictionMethod();
}
// ===------------------------------------------------------=== //
// Shared object inference runner 
// ===------------------------------------------------------=== //

SharedObjectInferenceRunner::SharedObjectInferenceRunner(std::shared_ptr<IModelSerializer> serializer,
                                                         const std::string& soPath,
                                                         int32_t tileSize,
                                                         int32_t thresholdSize,
                                                         int32_t featureIndexSize)
  : InferenceRunnerBase(serializer, tileSize, thresholdSize, featureIndexSize)
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
  mlir::ExecutionEngineOptions options{nullptr, {}, std::nullopt, executionEngineLibs};
  options.enablePerfNotificationListener = EnablePerfNotificationListener;
  auto maybeEngine = mlir::ExecutionEngine::create(module, options);
  assert(maybeEngine && "failed to construct an execution engine");
  return maybeEngine;
}

InferenceRunner::InferenceRunner(std::shared_ptr<IModelSerializer> serializer,
                                 mlir::ModuleOp module,
                                 int32_t tileSize,
                                 int32_t thresholdSize,
                                 int32_t featureIndexSize) 
  :InferenceRunnerBase(serializer, tileSize, thresholdSize, featureIndexSize),
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
