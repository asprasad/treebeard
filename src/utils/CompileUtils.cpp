#include <vector>
#include <sstream>
#include <chrono>
#include <iostream>
#include "Dialect.h"
#include "TestUtilsCommon.h"

#include "mlir/IR/MLIRContext.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/STLExtras.h"

#include "CompileUtils.h"
#include "ExecutionHelpers.h"
#include "TestUtilsCommon.h"

namespace TreeBeard
{

template<typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
mlir::ModuleOp SpecializeInputElementType(mlir::MLIRContext& context, const std::string&modelJsonPath, int32_t inputElementTypeWidth, int32_t batchSize, int32_t tileSize) {
  if (inputElementTypeWidth == 32) {
    return ConstructLLVMDialectModuleFromXGBoostJSON<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType, float>(context, modelJsonPath, batchSize, tileSize);
  }
  else if (inputElementTypeWidth == 64) {
    return ConstructLLVMDialectModuleFromXGBoostJSON<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType, double>(context, modelJsonPath, batchSize, tileSize);
  }
  else {
    assert (false && "Unknown input element type");
  }
  return mlir::ModuleOp();  
}

template<typename ThresholdType, typename ReturnType, typename FeatureIndexType>
mlir::ModuleOp SpecializeNodeIndexType(mlir::MLIRContext& context, const std::string&modelJsonPath,
                                       int32_t nodeIndexTypeWidth, int32_t inputElementTypeWidth, int32_t batchSize, int32_t tileSize) {
  if (nodeIndexTypeWidth == 8) {
    return SpecializeInputElementType<ThresholdType, ReturnType, FeatureIndexType, int8_t>(context, modelJsonPath, inputElementTypeWidth, batchSize, tileSize);
  } 
  else if (nodeIndexTypeWidth == 16) {
    return SpecializeInputElementType<ThresholdType, ReturnType, FeatureIndexType, int16_t>(context, modelJsonPath, inputElementTypeWidth, batchSize, tileSize);
  } 
  else if (nodeIndexTypeWidth == 32) {
    return SpecializeInputElementType<ThresholdType, ReturnType, FeatureIndexType, int32_t>(context, modelJsonPath, inputElementTypeWidth, batchSize, tileSize);
  }
  else if (nodeIndexTypeWidth == 64) {
    return SpecializeInputElementType<ThresholdType, ReturnType, FeatureIndexType, int64_t>(context, modelJsonPath, inputElementTypeWidth, batchSize, tileSize);
  } 
  else {
    assert (false && "Unknown feature index type");
  }
  return mlir::ModuleOp();
}


template<typename ThresholdType, typename ReturnType>
mlir::ModuleOp SpecializeFeatureIndexType(mlir::MLIRContext& context, const std::string&modelJsonPath,
                                          int32_t featureIndexTypeWidth, int32_t nodeIndexTypeWidth, int32_t inputElementTypeWidth,
                                          int32_t batchSize, int32_t tileSize) {
  if (featureIndexTypeWidth == 8) {
    return SpecializeNodeIndexType<ThresholdType, ReturnType, int8_t>(context, modelJsonPath, nodeIndexTypeWidth, inputElementTypeWidth, batchSize, tileSize);
  } 
  else if (featureIndexTypeWidth == 16) {
    return SpecializeNodeIndexType<ThresholdType, ReturnType, int16_t>(context, modelJsonPath, nodeIndexTypeWidth, inputElementTypeWidth, batchSize, tileSize);
  } 
  else if (featureIndexTypeWidth == 32) {
    return SpecializeNodeIndexType<ThresholdType, ReturnType, int32_t>(context, modelJsonPath, nodeIndexTypeWidth, inputElementTypeWidth, batchSize, tileSize);
  }
  else if (featureIndexTypeWidth == 64) {
    return SpecializeNodeIndexType<ThresholdType, ReturnType, int64_t>(context, modelJsonPath, nodeIndexTypeWidth, inputElementTypeWidth, batchSize, tileSize);
  } 
  else {
    assert (false && "Unknown feature index type");
  }
  return mlir::ModuleOp();
}

template<typename ThresholdType>
mlir::ModuleOp SpecializeReturnType(mlir::MLIRContext& context, const std::string&modelJsonPath,
                                    int32_t returnTypeWidth, int32_t featureIndexTypeWidth, 
                                    int32_t nodeIndexTypeWidth, int32_t inputElementTypeWidth, int32_t batchSize, int32_t tileSize) {
  if (returnTypeWidth == 32) {
    return SpecializeFeatureIndexType<ThresholdType, float>(context, modelJsonPath, featureIndexTypeWidth, 
                                                            nodeIndexTypeWidth, inputElementTypeWidth, batchSize, tileSize);
  }
  else if (returnTypeWidth == 64) {
    return SpecializeFeatureIndexType<ThresholdType, double>(context, modelJsonPath, featureIndexTypeWidth, 
                                                             nodeIndexTypeWidth, inputElementTypeWidth, batchSize, tileSize);
  } 
  else {
    assert (false && "Unknown return type");
  }
  return mlir::ModuleOp();
}


mlir::ModuleOp ConstructLLVMDialectModuleFromXGBoostJSON(mlir::MLIRContext& context, const std::string&modelJsonPath,
                                                         int32_t thresholdTypeWidth, int32_t returnTypeWidth, int32_t featureIndexTypeWidth, 
                                                         int32_t nodeIndexTypeWidth, int32_t inputElementTypeWidth, int32_t batchSize, int32_t tileSize) {
  if (thresholdTypeWidth == 32) {
    return SpecializeReturnType<float>(context, modelJsonPath, returnTypeWidth, featureIndexTypeWidth, nodeIndexTypeWidth, inputElementTypeWidth, batchSize, tileSize);
  }
  else if (thresholdTypeWidth == 64) {
    return SpecializeReturnType<double>(context, modelJsonPath, returnTypeWidth, featureIndexTypeWidth, nodeIndexTypeWidth, inputElementTypeWidth, batchSize, tileSize);
  }
  else {
    assert (false && "Unknown threshold type");
  }
  return mlir::ModuleOp();
}

void InitializeMLIRContext(mlir::MLIRContext& context) {
  context.getOrLoadDialect<mlir::decisionforest::DecisionForestDialect>();
  context.getOrLoadDialect<mlir::StandardOpsDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::vector::VectorDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
}

void ConvertXGBoostJSONToLLVMIR(const std::string&modelJsonPath, const std::string& llvmIRFilePath,
                                int32_t thresholdTypeWidth, int32_t returnTypeWidth, int32_t featureIndexTypeWidth, 
                                int32_t nodeIndexTypeWidth, int32_t inputElementTypeWidth, int32_t batchSize, int32_t tileSize) {
  mlir::MLIRContext context;
  InitializeMLIRContext(context);
  auto module = ConstructLLVMDialectModuleFromXGBoostJSON(context, modelJsonPath, thresholdTypeWidth, returnTypeWidth, featureIndexTypeWidth, 
                                                          nodeIndexTypeWidth, inputElementTypeWidth, batchSize, tileSize);
  mlir::decisionforest::dumpLLVMIRToFile(module, llvmIRFilePath);
}

template<typename FloatType>
int64_t RunXGBoostInferenceOnCSVInput(const std::string& csvPath, mlir::decisionforest::SharedObjectInferenceRunner& inferenceRunner, int32_t batchSize) {
  TreeBeard::test::TestCSVReader csvReader(csvPath);
  std::vector<std::vector<FloatType>> inputData;
  for (size_t i=batchSize  ; i<csvReader.NumberOfRows()-1 ; i += batchSize) {
    std::vector<FloatType> batch, preds;
    for (int32_t j=0 ; j<batchSize ; ++j) {
      auto rowIndex = (i-batchSize) + j;
      auto row = csvReader.GetRowOfType<FloatType>(rowIndex);
      row.pop_back();
      batch.insert(batch.end(), row.begin(), row.end());
    }
    inputData.push_back(batch);
  }

  char ch;
  std::cout << "Attach profiler and press any key...";
  std::cin >> ch;

  constexpr int32_t NUM_RUNS=50;
  size_t rowSize = csvReader.GetRow(0).size() - 1; // The last entry is the xgboost prediction
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  for (int32_t trial=0 ; trial<NUM_RUNS ; ++trial) {
    for(auto& batch : inputData) {
      assert (batch.size() % batchSize == 0);
      std::vector<FloatType> result(batchSize, -1);
      inferenceRunner.RunInference<FloatType, FloatType>(batch.data(), result.data(), rowSize, batchSize);
    }
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::cout << "Detach profiler and press any key...";
  std::cin >> ch;

  return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
}

void RunInferenceUsingSO(const std::string&modelJsonPath, const std::string& soPath, const std::string& csvPath,
                         int32_t thresholdTypeWidth, int32_t returnTypeWidth, int32_t featureIndexTypeWidth, 
                         int32_t nodeIndexTypeWidth, int32_t inputElementTypeWidth, int32_t batchSize, int32_t tileSize) {
  mlir::MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);
  TreeBeard::ConstructLLVMDialectModuleFromXGBoostJSON(context, modelJsonPath, thresholdTypeWidth, returnTypeWidth, featureIndexTypeWidth, 
                                                       nodeIndexTypeWidth, inputElementTypeWidth, batchSize, tileSize);
  
  mlir::decisionforest::SharedObjectInferenceRunner inferenceRunner(soPath, tileSize, thresholdTypeWidth, featureIndexTypeWidth);
  assert (inputElementTypeWidth == returnTypeWidth);
  int64_t time;
  if (inputElementTypeWidth == 32)
    time = RunXGBoostInferenceOnCSVInput<float>(csvPath, inferenceRunner, batchSize);
  else if (inputElementTypeWidth == 64)
    time = RunXGBoostInferenceOnCSVInput<double>(csvPath, inferenceRunner, batchSize);
  else
    assert(false && "Unknow floating point type");
  std::cout << "Execution time (us) : " << time << std::endl;
}


} // TreeBeard