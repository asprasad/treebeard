#ifndef _EXECUTIONHELPERS_H_
#define _EXECUTIONHELPERS_H_

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

namespace mlir
{
namespace decisionforest
{

class IModelSerializer;

// This is just a place holder type for documentation.
// We don't really want to intrepret this type on the CPU. 
struct Tile {

};

using LengthMemrefType = Memref<int64_t, 1>;
using OffsetMemrefType = Memref<int64_t, 1>;
// #TODO Tree-Beard#19
using ClassMemrefType = Memref<int8_t, 1>;
using ModelMemrefType = Memref<Tile, 1>;

using LUTEntryType = int8_t;
using LUTMemrefType = Memref<LUTEntryType, 2>;

// using ResultMemrefType = Memref<double, 1>;

class InferenceRunnerBase {
  friend class IModelSerializer;
protected:
  std::shared_ptr<IModelSerializer> m_serializer;
  int32_t m_inputElementBitWidth;
  int32_t m_returnTypeBitWidth;
  int32_t m_tileSize;
  int32_t m_thresholdSize;
  int32_t m_featureIndexSize;
  int32_t m_batchSize;
  int32_t m_rowSize;
  void *m_inferenceFuncPtr;
  LUTMemrefType m_lutMemref;

  virtual void* GetFunctionAddress(const std::string& functionName) = 0;
  void InitIntegerField(const std::string& functionName, int32_t& field);
  
  virtual void Init();
  
  template<typename InputElementType, typename ReturnType>
  int32_t RunInference_Default(InputElementType *input, ReturnType *returnValue) {
    
    typedef Memref<ReturnType, 1> (*InferenceFunc_t)(InputElementType*, InputElementType*, int64_t, int64_t, int64_t, int64_t, int64_t, 
                                                     ReturnType*, ReturnType*, int64_t, int64_t, int64_t);
    auto inferenceFuncPtr = reinterpret_cast<InferenceFunc_t>(m_inferenceFuncPtr);
    InputElementType *ptr = input, *alignedPtr = input;
    int64_t rowSize = m_rowSize, offset = 0, stride = 1;
    ReturnType *resultPtr = returnValue, *resultAlignedPtr = returnValue;
    int64_t resultLen = m_batchSize;
    inferenceFuncPtr(ptr, alignedPtr, offset, m_batchSize, rowSize, rowSize /*stride by rowSize to move from one row to the next*/, stride, 
                     resultPtr, resultAlignedPtr, offset, resultLen, stride);
    return 0;
  }

  bool SerializerHasCustomPredictionMethod();
  int32_t RunInference_CustomImpl(double *input, double* returnValue);

  template<typename InputElementType, typename ReturnType>
  int32_t RunInference_Custom(InputElementType *input, ReturnType *returnValue) {
    RunInference_CustomImpl(reinterpret_cast<double*>(input),
                            reinterpret_cast<double*>(returnValue));
    return 0;
  }
  
public:
  InferenceRunnerBase(std::shared_ptr<IModelSerializer> serializer,
                      int32_t tileSize,
                      int32_t thresholdSize,
                      int32_t featureIndexSize);
  virtual ~InferenceRunnerBase() { }
  
  int32_t GetBatchSize() { return m_batchSize; }
  int32_t GetTileSize() { return m_tileSize; }
  int32_t GetRowSize() { return m_rowSize; }
  int32_t GetThresholdWidth() { return m_thresholdSize; }
  int32_t GetFeatureIndexWidth() { return m_featureIndexSize; }
  int32_t GetInputElementBitWidth() { return m_inputElementBitWidth; }
  int32_t GetReturnTypeBitWidth() { return m_returnTypeBitWidth; }
  LUTMemrefType GetLUTMemref() { return m_lutMemref; }
  template<typename InputElementType, typename ReturnType>
  int32_t RunInference(InputElementType *input, ReturnType *returnValue) {
    if (SerializerHasCustomPredictionMethod()) {
      return RunInference_Custom(input, returnValue);
    }
    else {
      return RunInference_Default(input, returnValue);
    }
    return 0;
  }
};

class InferenceRunner : public InferenceRunnerBase {
protected:
  llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> m_maybeEngine;
  std::unique_ptr<mlir::ExecutionEngine>& m_engine;
  mlir::ModuleOp m_module;

  void* GetFunctionAddress(const std::string& functionName) override;
public:
  static llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> CreateExecutionEngine(mlir::ModuleOp module);
  InferenceRunner(std::shared_ptr<IModelSerializer> serializer, 
                  mlir::ModuleOp module,
                  int32_t tileSize, 
                  int32_t thresholdSize,
                  int32_t featureIndexSize);
};

class SharedObjectInferenceRunner : public InferenceRunnerBase{
  void *m_so;
protected:
  void* GetFunctionAddress(const std::string& functionName) override;
public:
  SharedObjectInferenceRunner(std::shared_ptr<IModelSerializer> serializer,
                              const std::string& soPath,
                              int32_t tileSize,
                              int32_t thresholdSize,
                              int32_t featureIndexSize);
  ~SharedObjectInferenceRunner();
};

} // decision forest
} // mlir

#endif // _EXECUTIONHELPERS_H_