#include <vector>
#include <sstream>
#include <chrono>
#include "Dialect.h"
#include "TestUtilsCommon.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "llvm/ADT/STLExtras.h"

#include "CompileUtils.h"
#include "ExecutionHelpers.h"
#include "TreeTilingDescriptor.h"
#include "TreeTilingUtils.h"
#include "ForestTestUtils.h"

using namespace mlir;

namespace TreeBeard
{
namespace test
{

std::string GetTypeName(double d) {
  return "double";
}

std::string GetTypeName(float f) {
  return "float";
}

// #define PROFILE_MODE

#ifndef PROFILE_MODE
constexpr int32_t NUM_RUNS = 1000;
#else
constexpr int32_t NUM_RUNS = 1000;
#endif

template<typename FloatType, typename ReturnType=FloatType>
int64_t Test_CodeGenForJSON_VariableBatchSize(int64_t batchSize, const std::string& modelJsonPath, int32_t tileSize, int32_t tileShapeBitWidth, 
                                              int32_t childIndexBitWidth, mlir::decisionforest::ScheduleManipulator *scheduleManipulator) {
  // TODO consider changing this so that you use the smallest possible type possible (need to make it a parameter)
  using FeatureIndexType = int16_t;
  using NodeIndexType = int16_t;

  mlir::MLIRContext context;
  int32_t floatTypeBitWidth = sizeof(FloatType)*8;
  TreeBeard::CompilerOptions options(floatTypeBitWidth, sizeof(ReturnType)*8, IsFloatType(ReturnType()), sizeof(FeatureIndexType)*8, sizeof(NodeIndexType)*8,
                                     floatTypeBitWidth, batchSize, tileSize, tileShapeBitWidth, childIndexBitWidth, false, scheduleManipulator);
  TreeBeard::InitializeMLIRContext(context);
  auto modelGlobalsJSONFilePath = TreeBeard::ModelJSONParser<FloatType, ReturnType, int32_t, int32_t, FloatType>::ModelGlobalJSONFilePathFromJSONFilePath(modelJsonPath);
  auto module = TreeBeard::ConstructLLVMDialectModuleFromXGBoostJSON<FloatType, ReturnType, FeatureIndexType, int32_t, FloatType>(context, modelJsonPath, modelGlobalsJSONFilePath, options);

  decisionforest::InferenceRunner inferenceRunner(modelGlobalsJSONFilePath, module, tileSize, floatTypeBitWidth, sizeof(FeatureIndexType)*8);
  
  TestCSVReader csvReader(modelJsonPath + ".test.sampled.csv");
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

#ifdef PROFILE_MODE
  char ch;
  std::cout << "Attach profiler and press any key...";
  std::cin >> ch;
#endif

  size_t rowSize = csvReader.GetRow(0).size() - 1; // The last entry is the xgboost prediction
  std::vector<FloatType> result(batchSize, -1);
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  for (int32_t trial=0 ; trial<NUM_RUNS ; ++trial) {
    for(auto& batch : inputData) {
      // assert (batch.size() % batchSize == 0);
      inferenceRunner.RunInference<FloatType, FloatType>(batch.data(), result.data(), rowSize, batchSize);
    }
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

#ifdef PROFILE_MODE
  std::cout << "Detach profiler and press any key...";
  std::cin >> ch;
#endif
  return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
}

template<typename FloatType, typename ReturnType=FloatType>
int64_t Test_CodeGenForJSON_ProbabilityBasedTiling(int64_t batchSize, const std::string& modelJsonPath, 
                                              const std::string& statsProfileCSV,
                                              int32_t tileSize, int32_t tileShapeBitWidth, 
                                              int32_t childIndexBitWidth, mlir::decisionforest::ScheduleManipulator *scheduleManipulator, bool probTiling) {
  // TODO consider changing this so that you use the smallest possible type possible (need to make it a parameter)
  using FeatureIndexType = int16_t;
  using NodeIndexType = int16_t;

  mlir::MLIRContext context;
  int32_t floatTypeBitWidth = sizeof(FloatType)*8;
  TreeBeard::CompilerOptions options(floatTypeBitWidth, sizeof(ReturnType)*8, IsFloatType(ReturnType()), sizeof(FeatureIndexType)*8, sizeof(NodeIndexType)*8,
                                     floatTypeBitWidth, batchSize, tileSize, tileShapeBitWidth, childIndexBitWidth, false, scheduleManipulator);
  TreeBeard::InitializeMLIRContext(context);
  auto modelGlobalsJSONFilePath = TreeBeard::ModelJSONParser<FloatType, ReturnType, int32_t, int32_t, FloatType>::ModelGlobalJSONFilePathFromJSONFilePath(modelJsonPath);
  // auto module = TreeBeard::ConstructLLVMDialectModuleFromXGBoostJSON<FloatType, ReturnType, FeatureIndexType, int32_t, FloatType>(context, modelJsonPath, modelGlobalsJSONFilePath, options);

  TreeBeard::XGBoostJSONParser<FloatType, ReturnType, FeatureIndexType, NodeIndexType, FloatType> xgBoostParser(context, modelJsonPath, 
                                                                                                                modelGlobalsJSONFilePath, statsProfileCSV, options.batchSize );
  xgBoostParser.Parse();
  xgBoostParser.SetChildIndexBitWidth(options.childIndexBitWidth);
  auto module = xgBoostParser.GetEvaluationFunction();

  if (options.scheduleManipulator) {
    auto schedule = xgBoostParser.GetSchedule();
    options.scheduleManipulator->Run(schedule);
  }

  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(context, module);
  // mlir::decisionforest::DoProbabilityBasedTiling(context, module, options.tileSize, options.tileShapeBitWidth);
  if (probTiling)
    mlir::decisionforest::DoHybridTiling(context, module, options.tileSize, options.tileShapeBitWidth);
  else
    mlir::decisionforest::DoUniformTiling(context, module, options.tileSize, options.tileShapeBitWidth, false);
  mlir::decisionforest::LowerEnsembleToMemrefs(context, module);
  mlir::decisionforest::ConvertNodeTypeToIndexType(context, module);
  // module->dump();
  mlir::decisionforest::LowerToLLVM(context, module);
  // mlir::decisionforest::dumpLLVMIR(module, false);

  decisionforest::InferenceRunner inferenceRunner(modelGlobalsJSONFilePath, module, tileSize, floatTypeBitWidth, sizeof(FeatureIndexType)*8);
  
  TestCSVReader csvReader(modelJsonPath + ".test.sampled.csv");
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

#ifdef PROFILE_MODE
  char ch;
  std::cout << "Attach profiler and press any key...";
  std::cin >> ch;
#endif

  size_t rowSize = csvReader.GetRow(0).size() - 1; // The last entry is the xgboost prediction
  std::vector<FloatType> result(batchSize, -1);
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  for (int32_t trial=0 ; trial<NUM_RUNS ; ++trial) {
    for(auto& batch : inputData) {
      // assert (batch.size() % batchSize == 0);
      inferenceRunner.RunInference<FloatType, FloatType>(batch.data(), result.data(), rowSize, batchSize);
    }
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

#ifdef PROFILE_MODE
  std::cout << "Detach profiler and press any key...";
  std::cin >> ch;
#endif
  return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
}

template<typename FPType, typename ReturnType, int32_t TileSize, int32_t BatchSize>
void RunSingleBenchmark_SingleConfig(const std::string& modelName, mlir::decisionforest::ScheduleManipulator *scheduleManipulator, bool probTiling) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/" + modelName + "_xgb_model_save.json";
  // if (!probTiling) {
  //   auto time = Test_CodeGenForJSON_VariableBatchSize<FPType, ReturnType>(BatchSize, modelJSONPath, TileSize, 16, 16, scheduleManipulator);
  //   std::cout << "\t" + modelName << "\t" << time << std::endl;
  // }
  // else {
    std::string statsProfileCSV = testModelsDir + "/profiles/" + modelName + ".test.csv";
    auto time = Test_CodeGenForJSON_ProbabilityBasedTiling<FPType, ReturnType>(BatchSize, modelJSONPath, statsProfileCSV, TileSize, 16, 16, scheduleManipulator, probTiling);
    std::cout << "\t" + modelName << "\t" << time << std::endl;
  // }
}


template<typename FPType, int32_t TileSize, int32_t BatchSize>
void RunBenchmark_SingleConfig(mlir::decisionforest::ScheduleManipulator *scheduleManipulator, bool probTiling) {
  std::cout << "Type\t" << GetTypeName(FPType()) << "\tBatch\t" << BatchSize << " \tTile\t" << TileSize << std::endl;
  RunSingleBenchmark_SingleConfig<FPType, FPType, TileSize, BatchSize>("abalone", scheduleManipulator, probTiling);
  RunSingleBenchmark_SingleConfig<FPType, FPType, TileSize, BatchSize>("airline", scheduleManipulator, probTiling);
  RunSingleBenchmark_SingleConfig<FPType, FPType, TileSize, BatchSize>("airline-ohe", scheduleManipulator, probTiling);
  // // RunSingleBenchmark_SingleConfig<FPType, FPType, TileSize, BatchSize>("bosch", scheduleManipulator);
  RunSingleBenchmark_SingleConfig<FPType, int8_t, TileSize, BatchSize>("covtype", scheduleManipulator, probTiling);
  RunSingleBenchmark_SingleConfig<FPType, FPType, TileSize, BatchSize>("epsilon", nullptr, probTiling);
  RunSingleBenchmark_SingleConfig<FPType, FPType, TileSize, BatchSize>("higgs", scheduleManipulator, probTiling);
  RunSingleBenchmark_SingleConfig<FPType, FPType, TileSize, BatchSize>("year_prediction_msd", scheduleManipulator, probTiling);
}

void RunAllBenchmarks(mlir::decisionforest::ScheduleManipulator *scheduleManipulator, bool probTiling) {
  constexpr int32_t batchSize = 500;
  // {
  //   using FPType = double;
  //   RunBenchmark_SingleConfig<FPType, 1, 1>();
  //   RunBenchmark_SingleConfig<FPType, 1, 2>();
  //   RunBenchmark_SingleConfig<FPType, 1, 4>();

  //   RunBenchmark_SingleConfig<FPType, 2, 1>();
  //   RunBenchmark_SingleConfig<FPType, 2, 2>();
  //   RunBenchmark_SingleConfig<FPType, 2, 4>();

  //   RunBenchmark_SingleConfig<FPType, 3, 1>();
  //   RunBenchmark_SingleConfig<FPType, 3, 2>();
  //   RunBenchmark_SingleConfig<FPType, 3, 4>();

  //   RunBenchmark_SingleConfig<FPType, 4, 1>();
  //   RunBenchmark_SingleConfig<FPType, 4, 2>();
  //   RunBenchmark_SingleConfig<FPType, 4, 4>();
  // }
  {
    using FPType = float;
    // RunBenchmark_SingleConfig<FPType, 1, 1>();
    // RunBenchmark_SingleConfig<FPType, 1, 2>();
    // RunBenchmark_SingleConfig<FPType, 1, batchSize>(scheduleManipulator);

    // RunBenchmark_SingleConfig<FPType, 2, 1>();
    // RunBenchmark_SingleConfig<FPType, 2, 2>();
    // RunBenchmark_SingleConfig<FPType, 2, 4>();

    // RunBenchmark_SingleConfig<FPType, 3, 1>();
    // RunBenchmark_SingleConfig<FPType, 3, 2>();
    // RunBenchmark_SingleConfig<FPType, 3, 4>();

    // RunBenchmark_SingleConfig<FPType, 4, 1>();
    // RunBenchmark_SingleConfig<FPType, 4, 2>();
    // RunBenchmark_SingleConfig<FPType, 4, batchSize>(scheduleManipulator);
    // RunBenchmark_SingleConfig<FPType, 5, 4>();
    // RunBenchmark_SingleConfig<FPType, 6, 4>();
    RunBenchmark_SingleConfig<FPType, 8, batchSize>(scheduleManipulator, probTiling);
  }
}

void RunSparseXGBoostBenchmarks(mlir::decisionforest::ScheduleManipulator *scheduleManipulator, bool probTiling=false) {
  decisionforest::UseSparseTreeRepresentation = true;
  std::cout << "\nRunning sparse benchmarks ... \n\n";
  RunAllBenchmarks(scheduleManipulator, probTiling);
}

void RunXGBoostBenchmarks() {
  // RunAllBenchmarks(nullptr);
  // RunSparseXGBoostBenchmarks(nullptr);
  
  // {
  //   decisionforest::UseSparseTreeRepresentation = false;
  //   std::cout << "\n\nOne Tree at a Time Schedule\n";
  //   TreeBeard::test::ScheduleManipulationFunctionWrapper scheduleManipulator(OneTreeAtATimeSchedule);
  //   // RunAllBenchmarks(&scheduleManipulator);
  //   RunSparseXGBoostBenchmarks(&scheduleManipulator, false);
  // }

  {
    decisionforest::UseSparseTreeRepresentation = false;
    std::cout << "\n\nProbabilistic tiling, One Tree at a Time Schedule\n";
    TreeBeard::test::ScheduleManipulationFunctionWrapper scheduleManipulator(OneTreeAtATimeSchedule);
    // RunAllBenchmarks(&scheduleManipulator);
    RunSparseXGBoostBenchmarks(&scheduleManipulator, true);
  }

  // {
  //   decisionforest::UseSparseTreeRepresentation = false;
  //   std::cout << "\n\n\nMultiple Trees at a Time Schedule\n\n";
  //   TreeBeard::test::ScheduleManipulationFunctionWrapper scheduleManipulator(TileTreeDimensionSchedule<20>);
  //   RunAllBenchmarks(&scheduleManipulator);
  //   RunSparseXGBoostBenchmarks(&scheduleManipulator);
  // }
}


} // test
} // TreeBeard
