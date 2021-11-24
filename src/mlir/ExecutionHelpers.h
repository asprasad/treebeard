#ifndef _EXECUTIONHELPERS_H_
#define _EXECUTIONHELPERS_H_

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/Support/TargetSelect.h"

#include "TreeTilingUtils.h"

namespace mlir
{
namespace decisionforest
{

#pragma pack(push, 1)
template<typename ThresholdType, typename FeatureIndexType, int32_t TileSize>
struct TileType_Packed {
  ThresholdType thresholds[TileSize];
  FeatureIndexType featureIndices[TileSize];
  bool operator==(const TileType_Packed<ThresholdType, IndexType, TileSize>& other) const {
    for (int32_t i=0 ; i<TileSize ; ++i)
      if (thresholds[i] != other.thresholds[i])
        return false;
    return index==other.index;
  }
};
#pragma pack(pop)

template<typename ThresholdType, typename FeatureIndexType, int32_t TileSize>
struct TileType_Natural {
  ThresholdType thresholds[TileSize];
  FeatureIndexType featureIndices[TileSize];
};

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
// using ResultMemrefType = Memref<double, 1>;

class InferenceRunner {
protected:
  llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> m_maybeEngine;
  std::unique_ptr<mlir::ExecutionEngine>& m_engine;
  mlir::ModuleOp m_module;
  int32_t m_tileSize;
  int32_t m_thresholdSize;
  int32_t m_featureIndexSize;

  template<typename ThresholdType, typename FeatureIndexType>
  int32_t CallInitMethod();
public:
  static llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> CreateExecutionEngine(mlir::ModuleOp module);
  InferenceRunner(mlir::ModuleOp module, int32_t tileSize, int32_t thresholdSize, int32_t featureIndexSize);

  int32_t InitializeLengthsArray();
  int32_t InitializeOffsetsArray();
  int32_t InitializeModelArray();
  int32_t InitializeLUT();

  void PrintLengthsArray();
  void PrintOffsetsArray();
  void PrintModelArray();

  template<typename InputElementType, typename ReturnType>
  int32_t RunInference(InputElementType *input, ReturnType *returnValue, int32_t inputRowSize, int32_t batchSize) {
    auto& engine = m_engine;
    Memref<ReturnType, 1> resultMemref;
    // Memref<InputElementType, 2> resultMemref;
    InputElementType *ptr = input, *alignedPtr = input;
    int64_t rowSize = inputRowSize, offset = 0, stride = 1;
    ReturnType *resultPtr = returnValue, *resultAlignedPtr = returnValue;
    int64_t resultLen = batchSize;
    void *args[] = { &ptr, &alignedPtr, &offset, &batchSize, &rowSize, &stride, &stride, // Input memref fields
                     &resultPtr, &resultAlignedPtr, &offset, &resultLen, &stride, // Result memref fields 
                     &resultMemref };
    auto invocationResult = engine->invokePacked("Prediction_Function", args);
    if (invocationResult) {
      llvm::errs() << "JIT invocation failed\n";
      return -1;
    }
    // std::cout << "Result memref length : " << resultMemref.lengths[0] << std::endl;
    return 0;
  }
};

class SharedObjectInferenceRunner {
  int32_t m_tileSize;
  int32_t m_thresholdSize;
  int32_t m_featureIndexSize;

  void *m_so;
  void *m_inferenceFuncPtr;

  template<typename ThresholdType, typename FeatureIndexType>
  int32_t CallInitMethod();
public:
  SharedObjectInferenceRunner(const std::string& soPath, int32_t tileSize, int32_t thresholdSize, int32_t featureIndexSize);
  ~SharedObjectInferenceRunner();
  int32_t InitializeLengthsArray();
  int32_t InitializeOffsetsArray();
  int32_t InitializeModelArray();
  int32_t InitializeLUT();

  template<typename InputElementType, typename ReturnType>
  int32_t RunInference(InputElementType *input, ReturnType *returnValue, int32_t inputRowSize, int32_t batchSize) {
    typedef Memref<ReturnType, 1> (*InferenceFunc_t)(InputElementType*, InputElementType*, int64_t, int64_t, int64_t, int64_t, int64_t, 
                                                     ReturnType*, ReturnType*, int64_t, int64_t, int64_t);
    auto inferenceFuncPtr = reinterpret_cast<InferenceFunc_t>(m_inferenceFuncPtr);
    InputElementType *ptr = input, *alignedPtr = input;
    int64_t rowSize = inputRowSize, offset = 0, stride = 1;
    ReturnType *resultPtr = returnValue, *resultAlignedPtr = returnValue;
    int64_t resultLen = batchSize;
    inferenceFuncPtr(ptr, alignedPtr, offset, batchSize, rowSize, stride, stride, 
                     resultPtr, resultAlignedPtr, offset, resultLen, stride);
    return 0;
  }

};

} // decision forest
} // mlir

#endif // _EXECUTIONHELPERS_H_