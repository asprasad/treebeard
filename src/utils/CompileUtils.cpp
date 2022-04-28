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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLExtras.h"

#include "CompileUtils.h"
#include "ExecutionHelpers.h"
#include "TestUtilsCommon.h"
#include "Logger.h"

namespace TreeBeard
{

namespace Logging
{

LoggingOptions::LoggingOptions() 
  : logGenCodeStats(false), logTreeStats(false)
{ }

bool LoggingOptions::ShouldEnableLogging() {
  return logGenCodeStats || logTreeStats;
}

LoggingOptions loggingOptions;

bool InitLoggingOptions() {
  loggingOptions.logGenCodeStats = false;
  loggingOptions.logTreeStats = false;
  return loggingOptions.ShouldEnableLogging();
}

bool loggingEnabled = InitLoggingOptions();

} // Logging

template<typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
mlir::ModuleOp SpecializeInputElementType(mlir::MLIRContext& context, const std::string&modelJsonPath,
                                          const std::string& modelGlobalsJSONPath, const CompilerOptions& options) {
  if (options.inputElementTypeWidth == 32) {
    return ConstructLLVMDialectModuleFromXGBoostJSON<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType, float>(context, modelJsonPath, modelGlobalsJSONPath, options);
  }
  else if (options.inputElementTypeWidth == 64) {
    return ConstructLLVMDialectModuleFromXGBoostJSON<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType, double>(context, modelJsonPath, modelGlobalsJSONPath, options);
  }
  else {
    assert (false && "Unknown input element type");
  }
  return mlir::ModuleOp();  
}

template<typename ThresholdType, typename ReturnType, typename FeatureIndexType>
mlir::ModuleOp SpecializeNodeIndexType(mlir::MLIRContext& context, const std::string&modelJsonPath, 
                                       const std::string& modelGlobalsJSONPath, const CompilerOptions& options) {
  if (options.nodeIndexTypeWidth == 8) {
    return SpecializeInputElementType<ThresholdType, ReturnType, FeatureIndexType, int8_t>(context, modelJsonPath, modelGlobalsJSONPath, options);
  } 
  else if (options.nodeIndexTypeWidth == 16) {
    return SpecializeInputElementType<ThresholdType, ReturnType, FeatureIndexType, int16_t>(context, modelJsonPath, modelGlobalsJSONPath, options);
  } 
  else if (options.nodeIndexTypeWidth == 32) {
    return SpecializeInputElementType<ThresholdType, ReturnType, FeatureIndexType, int32_t>(context, modelJsonPath, modelGlobalsJSONPath, options);
  }
  else if (options.nodeIndexTypeWidth == 64) {
    return SpecializeInputElementType<ThresholdType, ReturnType, FeatureIndexType, int64_t>(context, modelJsonPath, modelGlobalsJSONPath, options);
  } 
  else {
    assert (false && "Unknown feature index type");
  }
  return mlir::ModuleOp();
}


template<typename ThresholdType, typename ReturnType>
mlir::ModuleOp SpecializeFeatureIndexType(mlir::MLIRContext& context, const std::string&modelJsonPath, 
                                          const std::string& modelGlobalsJSONPath, const CompilerOptions& options) {
  if (options.featureIndexTypeWidth == 8) {
    return SpecializeNodeIndexType<ThresholdType, ReturnType, int8_t>(context, modelJsonPath, modelGlobalsJSONPath, options);
  } 
  else if (options.featureIndexTypeWidth == 16) {
    return SpecializeNodeIndexType<ThresholdType, ReturnType, int16_t>(context, modelJsonPath, modelGlobalsJSONPath, options);
  } 
  else if (options.featureIndexTypeWidth == 32) {
    return SpecializeNodeIndexType<ThresholdType, ReturnType, int32_t>(context, modelJsonPath, modelGlobalsJSONPath, options);
  }
  else if (options.featureIndexTypeWidth == 64) {
    return SpecializeNodeIndexType<ThresholdType, ReturnType, int64_t>(context, modelJsonPath, modelGlobalsJSONPath, options);
  } 
  else {
    assert (false && "Unknown feature index type");
  }
  return mlir::ModuleOp();
}

template<typename ThresholdType>
mlir::ModuleOp SpecializeReturnType(mlir::MLIRContext& context, const std::string&modelJsonPath, 
                                    const std::string& modelGlobalsJSONPath, const CompilerOptions& options) {
  if (options.returnTypeFloatType) {
    if (options.returnTypeWidth == 32) {
      return SpecializeFeatureIndexType<ThresholdType, float>(context, modelJsonPath, modelGlobalsJSONPath, options);
    }
    else if (options.returnTypeWidth == 64) {
      return SpecializeFeatureIndexType<ThresholdType, double>(context, modelJsonPath, modelGlobalsJSONPath, options);
    } 
    else {
      assert (false && "Unknown return type");
    }
  }
  else {
    if (options.returnTypeWidth == 8) {
      return SpecializeFeatureIndexType<ThresholdType, int8_t>(context, modelJsonPath, modelGlobalsJSONPath, options);
    }
    else {
      assert (false && "Unknown return type");
    }
  }
  return mlir::ModuleOp();
}


mlir::ModuleOp ConstructLLVMDialectModuleFromXGBoostJSON(mlir::MLIRContext& context, const std::string&modelJsonPath, 
                                                         const std::string& modelGlobalsJSONPath, const CompilerOptions& options) {
  if (options.thresholdTypeWidth == 32) {
    return SpecializeReturnType<float>(context, modelJsonPath, modelGlobalsJSONPath, options);
  }
  else if (options.thresholdTypeWidth == 64) {
    return SpecializeReturnType<double>(context, modelJsonPath, modelGlobalsJSONPath, options);
  }
  else {
    assert (false && "Unknown threshold type");
  }
  return mlir::ModuleOp();
}

void InitializeMLIRContext(mlir::MLIRContext& context) {
  context.getOrLoadDialect<mlir::decisionforest::DecisionForestDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::vector::VectorDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::arith::ArithmeticDialect>();
  context.getOrLoadDialect<mlir::omp::OpenMPDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
}

void ConvertXGBoostJSONToLLVMIR(const std::string&modelJsonPath, const std::string& llvmIRFilePath, const std::string& modelGlobalsJSONPath,
                                const CompilerOptions& options) {
  mlir::MLIRContext context;
  InitializeMLIRContext(context);
  auto module = ConstructLLVMDialectModuleFromXGBoostJSON(context, modelJsonPath, modelGlobalsJSONPath, options);
  mlir::decisionforest::dumpLLVMIRToFile(module, llvmIRFilePath);
}

template<typename FloatType, typename ReturnType=FloatType>
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

  constexpr int32_t NUM_RUNS=1000;
  size_t rowSize = csvReader.GetRow(0).size() - 1; // The last entry is the xgboost prediction
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  for (int32_t trial=0 ; trial<NUM_RUNS ; ++trial) {
    for(auto& batch : inputData) {
      assert (batch.size() % batchSize == 0);
      std::vector<ReturnType> result(batchSize, -1);
      inferenceRunner.RunInference<FloatType, ReturnType>(batch.data(), result.data(), rowSize, batchSize);
    }
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::cout << "Detach profiler and press any key...";
  std::cin >> ch;

  return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
}

void RunInferenceUsingSO(const std::string&modelJsonPath, const std::string& soPath, const std::string& modelGlobalsJSONPath, 
                         const std::string& csvPath, const CompilerOptions& options) {
  mlir::decisionforest::SharedObjectInferenceRunner inferenceRunner(modelGlobalsJSONPath, soPath, options.tileSize, options.thresholdTypeWidth, options.featureIndexTypeWidth);
  int64_t time;
  if (options.returnTypeFloatType){ 
    assert (options.inputElementTypeWidth == options.returnTypeWidth);
    if (options.inputElementTypeWidth == 32)
      time = RunXGBoostInferenceOnCSVInput<float>(csvPath, inferenceRunner, options.batchSize);
    else if (options.inputElementTypeWidth == 64)
      time = RunXGBoostInferenceOnCSVInput<double>(csvPath, inferenceRunner, options.batchSize);
    else
      assert(false && "Unknown floating point type");
  }
  else {
    assert (options.returnTypeWidth == 8);
    if (options.inputElementTypeWidth == 32)
      time = RunXGBoostInferenceOnCSVInput<float, int8_t>(csvPath, inferenceRunner, options.batchSize);
    else if (options.inputElementTypeWidth == 64)
      time = RunXGBoostInferenceOnCSVInput<double, int8_t>(csvPath, inferenceRunner, options.batchSize);
    else
      assert(false && "Unknown integer type");
  }
  TreeBeard::Logging::Log("Execution time (us) : "  +  std::to_string(time));
}


} // TreeBeard