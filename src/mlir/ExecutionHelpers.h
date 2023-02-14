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
#include "TreebeardContext.h"

namespace mlir
{
namespace decisionforest
{

using LengthMemrefType = Memref<int64_t, 1>;
using OffsetMemrefType = Memref<int64_t, 1>;
// #TODO Tree-Beard#19
using ClassMemrefType = Memref<int8_t, 1>;

// using ResultMemrefType = Memref<double, 1>;

class InferenceRunnerBase {
protected:
  std::shared_ptr<IModelSerializer> m_serializer;
  std::string m_modelGlobalsJSONFilePath;
  int32_t m_inputElementBitWidth;
  int32_t m_returnTypeBitWidth;
  int32_t m_tileSize;
  int32_t m_thresholdSize;
  int32_t m_featureIndexSize;
  int32_t m_batchSize;
  int32_t m_rowSize;
  void *m_inferenceFuncPtr;
  
  template<typename ThresholdType, typename FeatureIndexType, typename TileShapeType, typename ChildIndexType>
  int32_t CallInitMethod();
  
  template<typename ThresholdType, typename FeatureIndexType>
  int32_t ResolveTileShapeType();

  template<typename ThresholdType, typename FeatureIndexType, typename TileShapeType>
  int32_t ResolveChildIndexType();

  virtual void* GetFunctionAddress(const std::string& functionName) = 0;
  
  void Init();
public:
  InferenceRunnerBase(std::shared_ptr<IModelSerializer> serializer,
                     int32_t tileSize,
                     int32_t thresholdSize,
                     int32_t featureIndexSize)
    : m_serializer(serializer),
      m_modelGlobalsJSONFilePath(serializer->GetFilePath()),
      m_tileSize(tileSize),
      m_thresholdSize(thresholdSize),
      m_featureIndexSize(featureIndexSize) 
  { 
    decisionforest::ForestJSONReader::GetInstance().SetFilePath(m_modelGlobalsJSONFilePath);
    decisionforest::ForestJSONReader::GetInstance().ParseJSONFile();
    // TODO read the thresholdSize and featureIndexSize from the JSON!
    m_batchSize = decisionforest::ForestJSONReader::GetInstance().GetBatchSize();
    m_rowSize = decisionforest::ForestJSONReader::GetInstance().GetRowSize();
    m_inputElementBitWidth = decisionforest::ForestJSONReader::GetInstance().GetInputElementBitWidth();
    m_returnTypeBitWidth = decisionforest::ForestJSONReader::GetInstance().GetReturnTypeBitWidth();
  }
  virtual ~InferenceRunnerBase() { }
  
  int32_t InitializeLengthsArray();
  int32_t InitializeOffsetsArray();
  int32_t InitializeModelArray();
  int32_t InitializeLUT();
  int32_t InitializeLeafArrays();
  void InitializeClassInformation();

  int32_t GetBatchSize() { return m_batchSize; }
  int32_t GetRowSize() { return m_rowSize; }
  int32_t GetThresholdWidth() { return m_thresholdSize; }
  int32_t GetInputElementBitWidth() { return m_inputElementBitWidth; }
  int32_t GetReturnTypeBitWidth() { return m_returnTypeBitWidth; }
  
  void PrintLengthsArray();
  void PrintOffsetsArray();
  void PrintModelArray();

  template<typename InputElementType, typename ReturnType>
  int32_t RunInference(InputElementType *input, ReturnType *returnValue, int32_t inputRowSize, int32_t batchSize) {
    assert (batchSize == m_batchSize);
    assert (inputRowSize == m_rowSize);

    typedef Memref<ReturnType, 1> (*InferenceFunc_t)(InputElementType*, InputElementType*, int64_t, int64_t, int64_t, int64_t, int64_t, 
                                                     ReturnType*, ReturnType*, int64_t, int64_t, int64_t);
    auto inferenceFuncPtr = reinterpret_cast<InferenceFunc_t>(m_inferenceFuncPtr);
    InputElementType *ptr = input, *alignedPtr = input;
    int64_t rowSize = inputRowSize, offset = 0, stride = 1;
    ReturnType *resultPtr = returnValue, *resultAlignedPtr = returnValue;
    int64_t resultLen = batchSize;
    inferenceFuncPtr(ptr, alignedPtr, offset, batchSize, rowSize, rowSize /*stride by rowSize to move from one row to the next*/, stride, 
                     resultPtr, resultAlignedPtr, offset, resultLen, stride);
    return 0;
  }

  template<typename InputElementType, typename ReturnType>
  int32_t RunInference(InputElementType *input, ReturnType *returnValue) {
    return RunInference<InputElementType, ReturnType>(input, returnValue, m_rowSize, m_batchSize);
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