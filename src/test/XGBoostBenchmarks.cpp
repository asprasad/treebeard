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
using namespace mlir::decisionforest;

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
constexpr int32_t NUM_RUNS = 100;
#else
constexpr int32_t NUM_RUNS = 1000;
#endif

int32_t GetNumberOfCores() { 
  return 8;
}

template<typename FloatType, typename ReturnType=FloatType>
int64_t Test_CodeGenForJSON_ProbabilityBasedTiling(int64_t batchSize, const std::string& modelJsonPath, 
                                              const std::string& statsProfileCSV,
                                              int32_t tileSize, int32_t tileShapeBitWidth, 
                                              int32_t childIndexBitWidth, mlir::decisionforest::ScheduleManipulator *scheduleManipulator, 
                                              bool probTiling, bool parallel,
                                              int32_t pipelineSize) {
  // TODO consider changing this so that you use the smallest possible type possible (need to make it a parameter)
  using FeatureIndexType = int16_t;
  using NodeIndexType = int16_t;

  mlir::MLIRContext context;
  int32_t floatTypeBitWidth = sizeof(FloatType)*8;
  bool reorderTrees = probTiling || parallel || pipelineSize > 1;
  TreeBeard::CompilerOptions options(floatTypeBitWidth, sizeof(ReturnType)*8, IsFloatType(ReturnType()), sizeof(FeatureIndexType)*8, sizeof(NodeIndexType)*8,
                                     floatTypeBitWidth, batchSize, tileSize, tileShapeBitWidth, childIndexBitWidth,
                                     (probTiling ? TreeBeard::TilingType::kHybrid : TreeBeard::TilingType::kUniform), 
                                     pipelineSize > 1, // make all leaves same depth
                                     reorderTrees, // reorder trees
                                     reorderTrees ? nullptr : scheduleManipulator);

  options.statsProfileCSVPath = statsProfileCSV;
  options.SetPipelineSize(pipelineSize);

  if (parallel)
    options.numberOfCores = GetNumberOfCores();
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

template<typename FPType, typename ReturnType, int32_t TileSize, int32_t BatchSize>
int64_t RunSingleBenchmark_SingleConfig(const std::string& modelName, mlir::decisionforest::ScheduleManipulator *scheduleManipulator,
                                        bool probTiling, bool parallel, int32_t pipelineSize) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/" + modelName + "_xgb_model_save.json";
  std::string statsProfileCSV = testModelsDir + "/profiles/" + modelName + ".test.csv";
  auto time = Test_CodeGenForJSON_ProbabilityBasedTiling<FPType, ReturnType>(BatchSize, modelJSONPath, statsProfileCSV, 
                                                                            TileSize, 16, 16, scheduleManipulator,
                                                                            probTiling, parallel, pipelineSize);
  return time;
}

template<typename FPType, int32_t TileSize, int32_t BatchSize>
void RunBenchmark_SingleConfig(mlir::decisionforest::ScheduleManipulator *scheduleManipulator, bool probTiling, const std::string& config, bool parallel, int32_t pipelineSize) {
  std::cout << config << ", ";
  std::cout << GetTypeName(FPType()) << ", " << BatchSize << " , " << TileSize;
  std::cout << ", " << RunSingleBenchmark_SingleConfig<FPType, FPType, TileSize, BatchSize>("abalone", scheduleManipulator, probTiling, parallel, pipelineSize) << std::flush;
  std::cout << ", " << RunSingleBenchmark_SingleConfig<FPType, FPType, TileSize, BatchSize>("airline", scheduleManipulator, probTiling, parallel, pipelineSize) << std::flush;
  std::cout << ", " << RunSingleBenchmark_SingleConfig<FPType, FPType, TileSize, BatchSize>("airline-ohe", scheduleManipulator, probTiling, parallel, pipelineSize) << std::flush;
  // // RunSingleBenchmark_SingleConfig<FPType, FPType, TileSize, BatchSize>("bosch", scheduleManipulator) << std::flush;
  std::cout << ", " << RunSingleBenchmark_SingleConfig<FPType, int8_t, TileSize, BatchSize>("covtype", scheduleManipulator, probTiling, parallel, pipelineSize) << std::flush;
  std::cout << ", " << RunSingleBenchmark_SingleConfig<FPType, FPType, TileSize, BatchSize>("epsilon", nullptr, probTiling, parallel, pipelineSize) << std::flush;
  std::cout << ", " << RunSingleBenchmark_SingleConfig<FPType, int8_t, TileSize, BatchSize>("letters", scheduleManipulator, probTiling, parallel, pipelineSize) << std::flush;
  std::cout << ", " << RunSingleBenchmark_SingleConfig<FPType, FPType, TileSize, BatchSize>("higgs", scheduleManipulator, probTiling, parallel, pipelineSize) << std::flush;
  std::cout << ", " << RunSingleBenchmark_SingleConfig<FPType, FPType, TileSize, BatchSize>("year_prediction_msd", scheduleManipulator, probTiling, parallel, pipelineSize) << std::flush;
  std::cout << std::endl;
}

void RunAllBenchmarks(mlir::decisionforest::ScheduleManipulator *scheduleManipulator, 
                      bool probTiling, const std::string& config, bool parallel=false, int32_t pipelineSize = -1) {
  constexpr int32_t batchSize = 500;
  {
    using FPType = float;
    RunBenchmark_SingleConfig<FPType, 1, batchSize>(scheduleManipulator, false, config, parallel, pipelineSize);
    RunBenchmark_SingleConfig<FPType, 4, batchSize>(scheduleManipulator, probTiling, config, parallel, pipelineSize);
    RunBenchmark_SingleConfig<FPType, 8, batchSize>(scheduleManipulator, probTiling, config, parallel, pipelineSize);
  }
}

void RunAllPipelinedBenchmarks(const std::string& config, bool parallel=false) {
  constexpr int32_t batchSize = 500;
  {
    using FPType = float;
    RunBenchmark_SingleConfig<FPType, 4, batchSize>(nullptr, false, config, parallel, 2);
    RunBenchmark_SingleConfig<FPType, 8, batchSize>(nullptr, false, config, parallel, 4);
    RunBenchmark_SingleConfig<FPType, 8, batchSize>(nullptr, false, config, parallel, 8);
  }
}

void RunDefaultScheduleXGBoostBenchmarks() {
  RunAllBenchmarks(nullptr, false, "array-default_sched");
  
  decisionforest::UseSparseTreeRepresentation = true;
  RunAllBenchmarks(nullptr, false, "sparse-default_sched");
  decisionforest::UseSparseTreeRepresentation = false;
}

void RunOneTreeAtATimeScheduleXGBoostBenchmarks() {
  mlir::decisionforest::ScheduleManipulationFunctionWrapper scheduleManipulator(OneTreeAtATimeSchedule);
  RunAllBenchmarks(&scheduleManipulator, false, "array-one_tree");
  
  decisionforest::UseSparseTreeRepresentation = true;
  RunAllBenchmarks(&scheduleManipulator, false, "sparse-one_tree");
  decisionforest::UseSparseTreeRepresentation = false;
}

void RunProbabilisticOneTreeAtATimeScheduleXGBoostBenchmarks() {
  // There is too much of a blow up of model size when we use array + prob tiling. 
  // So not measuring that.
  mlir::decisionforest::ScheduleManipulationFunctionWrapper scheduleManipulator(OneTreeAtATimeSchedule);
  
  decisionforest::UseSparseTreeRepresentation = true;
  RunAllBenchmarks(&scheduleManipulator, true, "sparse-prob-one_tree");
  decisionforest::UseSparseTreeRepresentation = false;
}

void RunProbabilisticOneTreeAtATimeSchedule_RemoveExtraHop_XGBoostBenchmarks() {
  // There is too much of a blow up of model size when we use array + prob tiling. 
  // So not measuring that.
  mlir::decisionforest::ScheduleManipulationFunctionWrapper scheduleManipulator(OneTreeAtATimeSchedule);
  
  decisionforest::UseSparseTreeRepresentation = true;
  decisionforest::PeeledCodeGenForProbabiltyBasedTiling = true;
  RunAllBenchmarks(&scheduleManipulator, true, "sparse_remove_extra_hop-prob-one_tree");
  decisionforest::UseSparseTreeRepresentation = false;
  decisionforest::PeeledCodeGenForProbabiltyBasedTiling = false;
}

void RunOneTreeAtATimeParallelScheduleXGBoostBenchmarks() {
  RunAllBenchmarks(nullptr, false, "array-one_tree-par", true);
  
  decisionforest::UseSparseTreeRepresentation = true;
  RunAllBenchmarks(nullptr, false, "sparse-one_tree-par", true);
  decisionforest::UseSparseTreeRepresentation = false;
}

void RunPipeliningBenchmarks() {
  RunAllPipelinedBenchmarks("array-pipelined_sched");
  
  decisionforest::UseSparseTreeRepresentation = true;
  RunAllPipelinedBenchmarks("sparse-pipelined_sched");
  decisionforest::UseSparseTreeRepresentation = false;
}

void RunXGBoostBenchmarks() {
  RunDefaultScheduleXGBoostBenchmarks();
  // RunOneTreeAtATimeScheduleXGBoostBenchmarks();
  // RunProbabilisticOneTreeAtATimeScheduleXGBoostBenchmarks();
  // RunProbabilisticOneTreeAtATimeSchedule_RemoveExtraHop_XGBoostBenchmarks();
  // RunOneTreeAtATimeParallelScheduleXGBoostBenchmarks();
  RunPipeliningBenchmarks();

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
