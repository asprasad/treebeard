#ifndef _GPUEXECUTIONHELPER_H_
#define _GPUEXECUTIONHELPER_H_

#ifdef TREEBEARD_GPU_SUPPORT

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/Support/TargetSelect.h"

#include "TreeTilingUtils.h"
#include "TypeDefinitions.h"
#include "TreebeardContext.h"
#include "ExecutionHelpers.h"

namespace mlir
{
namespace decisionforest
{

class GPUInferenceRunner : public InferenceRunnerBase {
protected:
  using super = InferenceRunnerBase;
  llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> m_maybeEngine;
  std::unique_ptr<mlir::ExecutionEngine>& m_engine;
  mlir::ModuleOp m_module;

  void* GetFunctionAddress(const std::string& functionName) override;
  void Init()  final;
  int32_t initializeGpuLut();
  llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> CreateExecutionEngine(mlir::ModuleOp module);
public:
  GPUInferenceRunner(std::shared_ptr<IModelSerializer> serializer, 
                     mlir::ModuleOp module,
                     int32_t tileSize,
                     int32_t thresholdSize,
                     int32_t featureIndexSize)
    : InferenceRunnerBase(serializer, tileSize, thresholdSize, featureIndexSize),
      m_maybeEngine(CreateExecutionEngine(module)),
      m_engine(m_maybeEngine.get()), m_module(module) 
  {
    m_serializer->ReadData(); 
    // Okay to call virtual fn here, it's final.
    Init();
  }
  virtual ~GPUInferenceRunner();
};

} // decisionforest
} // mlir

#endif // TREEBEARD_GPU_SUPPORT

#endif // _GPUEXECUTIONHELPER_H_