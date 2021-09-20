#include <iostream>

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/Support/TargetSelect.h"

namespace {

template<typename T, int32_t Rank>
struct Memref {
  T *bufferPtr;
  T *alignedPtr;
  int64_t offset;
  int64_t lengths[Rank];
  int64_t strides[Rank];
};

using LengthMemrefType = Memref<int64_t, 1>;
using OffsetMemrefType = Memref<int64_t, 1>;
using ResultMemrefType = Memref<double, 1>;

class InferenceRunner {
  llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> m_maybeEngine;
  std::unique_ptr<mlir::ExecutionEngine>& m_engine;
  mlir::ModuleOp m_module;

  static llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> CreateEngine(mlir::ModuleOp module) {
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
public:
  InferenceRunner(mlir::ModuleOp module) 
    :m_maybeEngine(CreateEngine(module)), m_engine(m_maybeEngine.get()), m_module(module)
  {   }

  int32_t InitializeLengthsArray() {
    auto& engine = m_engine;
    Memref<int64_t, 1> lengthMemref;
    void *args[] = { &lengthMemref };
    auto invocationResult = engine->invokePacked("Get_lengths", args);
    if (invocationResult) {
      llvm::errs() << "JIT invocation failed\n";
      return -1;
    }
    std::cout << lengthMemref.lengths[0] << std::endl;
    return 0;
  }
};

} //anonymous

namespace mlir
{
namespace decisionforest
{

int runJIT(mlir::ModuleOp module) {
  InferenceRunner inferenceRunner(module);
  inferenceRunner.InitializeLengthsArray();
  return 0;
}

} // decisionforest
} // mlir