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
constexpr int32_t NUM_RUNS = 500;
#else
constexpr int32_t NUM_RUNS = 1000;
#endif

// int32_t GetNumberOfCores() { 
//   return 16;
// }

template<typename FloatType, typename ReturnType=FloatType>
double Test_CodeGenForJSON_ProbabilityBasedTiling(int64_t batchSize, const std::string& modelJsonPath, 
                                              const std::string& statsProfileCSV,
                                              int32_t tileSize, int32_t tileShapeBitWidth, 
                                              int32_t childIndexBitWidth, mlir::decisionforest::ScheduleManipulator *scheduleManipulator, 
                                              bool probTiling, int32_t numberOfCores,
                                              int32_t pipelineSize) {
  // TODO consider changing this so that you use the smallest possible type possible (need to make it a parameter)
  using FeatureIndexType = int16_t;
  using NodeIndexType = int16_t;

  mlir::MLIRContext context;
  int32_t floatTypeBitWidth = sizeof(FloatType)*8;
  bool reorderTrees = probTiling || (numberOfCores!=-1) || pipelineSize > 1;
  TreeBeard::CompilerOptions options(floatTypeBitWidth, sizeof(ReturnType)*8, IsFloatType(ReturnType()), sizeof(FeatureIndexType)*8, sizeof(NodeIndexType)*8,
                                     floatTypeBitWidth, batchSize, tileSize, tileShapeBitWidth, childIndexBitWidth,
                                     (probTiling ? TreeBeard::TilingType::kHybrid : TreeBeard::TilingType::kUniform), 
                                     pipelineSize > 1, // make all leaves same depth
                                     reorderTrees, // reorder trees
                                     reorderTrees ? nullptr : scheduleManipulator);

  options.statsProfileCSVPath = statsProfileCSV;
  options.SetPipelineSize(pipelineSize);

  if (numberOfCores != -1)
    options.numberOfCores = numberOfCores;
  TreeBeard::InitializeMLIRContext(context);
  auto modelGlobalsJSONFilePath = TreeBeard::ModelJSONParser<FloatType, ReturnType, int32_t, int32_t, FloatType>::ModelGlobalJSONFilePathFromJSONFilePath(modelJsonPath);
  auto module = TreeBeard::ConstructLLVMDialectModuleFromXGBoostJSON<FloatType, ReturnType, FeatureIndexType, int32_t, FloatType>(context, modelJsonPath, modelGlobalsJSONFilePath, options);

  decisionforest::InferenceRunner inferenceRunner(modelGlobalsJSONFilePath, module, tileSize, floatTypeBitWidth, sizeof(FeatureIndexType)*8);
  
  TestCSVReader csvReader(modelJsonPath + ".test.sampled.csv", 2000 /*num lines*/);
  assert (csvReader.NumberOfRows() == 2000);

  std::vector<std::vector<FloatType>> inputData;
  for (size_t i=batchSize  ; i<=csvReader.NumberOfRows() ; i += batchSize) {
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
  std::vector<ReturnType> result(batchSize, -1);
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  for (int32_t trial=0 ; trial<NUM_RUNS ; ++trial) {
    for(auto& batch : inputData) {
      // assert (batch.size() % batchSize == 0);
      inferenceRunner.RunInference<FloatType, ReturnType>(batch.data(), result.data(), rowSize, batchSize);
    }
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

#ifdef PROFILE_MODE
  std::cout << "Detach profiler and press any key...";
  std::cin >> ch;
#endif
  int64_t numSamples = NUM_RUNS * inputData.size() * batchSize;
  int64_t timeTaken = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
  auto timePerSample = (double)timeTaken/(double)numSamples;
  return timePerSample;
}

template<typename FPType, typename ReturnType, int32_t TileSize>
double RunSingleBenchmark_SingleConfig(const std::string& modelName, mlir::decisionforest::ScheduleManipulator *scheduleManipulator,
                                        bool probTiling, int32_t numCores, int32_t pipelineSize, int32_t BatchSize) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/" + modelName + "_xgb_model_save.json";
  std::string statsProfileCSV = testModelsDir + "/profiles/" + modelName + ".test.csv";
  auto time = Test_CodeGenForJSON_ProbabilityBasedTiling<FPType, ReturnType>(BatchSize, modelJSONPath, statsProfileCSV, 
                                                                            TileSize, 16, 16, scheduleManipulator,
                                                                            probTiling, numCores, pipelineSize);
  return time;
}

template<typename FPType, int32_t TileSize>
void RunBenchmark_SingleConfig(mlir::decisionforest::ScheduleManipulator *scheduleManipulator, bool probTiling,
                               const std::string& config, int32_t numCores, int32_t pipelineSize, int32_t BatchSize) {
  std::cout << config << ", ";
  std::cout << GetTypeName(FPType()) << ", " << BatchSize << " , " << TileSize;
  std::cout << ", " << RunSingleBenchmark_SingleConfig<FPType, FPType, TileSize>("abalone", scheduleManipulator, probTiling, numCores, pipelineSize, BatchSize) << std::flush;
  std::cout << ", " << RunSingleBenchmark_SingleConfig<FPType, FPType, TileSize>("airline", scheduleManipulator, probTiling, numCores, pipelineSize, BatchSize) << std::flush;
  std::cout << ", " << RunSingleBenchmark_SingleConfig<FPType, FPType, TileSize>("airline-ohe", scheduleManipulator, probTiling, numCores, pipelineSize, BatchSize) << std::flush;
  std::cout << ", " << RunSingleBenchmark_SingleConfig<FPType, int8_t, TileSize>("covtype", scheduleManipulator, probTiling, numCores, pipelineSize, BatchSize) << std::flush;
  std::cout << ", " << RunSingleBenchmark_SingleConfig<FPType, FPType, TileSize>("epsilon", scheduleManipulator, probTiling, numCores, pipelineSize, BatchSize) << std::flush;
  std::cout << ", " << RunSingleBenchmark_SingleConfig<FPType, int8_t, TileSize>("letters", scheduleManipulator, probTiling, numCores, pipelineSize, BatchSize) << std::flush;
  std::cout << ", " << RunSingleBenchmark_SingleConfig<FPType, FPType, TileSize>("higgs", scheduleManipulator, probTiling, numCores, pipelineSize, BatchSize) << std::flush;
  std::cout << ", " << RunSingleBenchmark_SingleConfig<FPType, FPType, TileSize>("year_prediction_msd", scheduleManipulator, probTiling, numCores, pipelineSize, BatchSize) << std::flush;
  std::cout << std::endl;
}

void RunAllBenchmarks(mlir::decisionforest::ScheduleManipulator *scheduleManipulator, int32_t batchSize,
                      bool probTiling, const std::string& config, int32_t numCores=-1, 
                      int32_t pipelineSize = -1) {
  {
    using FPType = float;
    RunBenchmark_SingleConfig<FPType, 1>(scheduleManipulator, false, config, numCores, pipelineSize, batchSize);
    RunBenchmark_SingleConfig<FPType, 4>(scheduleManipulator, probTiling, config, numCores, pipelineSize, batchSize);
    RunBenchmark_SingleConfig<FPType, 8>(scheduleManipulator, probTiling, config, numCores, pipelineSize, batchSize);
  }
}

void RunAllPipelinedBenchmarks(int32_t batchSize, const std::string& config, int32_t numCores=-1) {
  {
    using FPType = float;
    RunBenchmark_SingleConfig<FPType, 8>(nullptr, false, config + std::string("-2pipeline"), numCores, 2, batchSize);
    RunBenchmark_SingleConfig<FPType, 8>(nullptr, false, config + std::string("-4pipeline"), numCores, 4, batchSize);
    RunBenchmark_SingleConfig<FPType, 8>(nullptr, false, config + std::string("-8pipeline"), numCores, 8, batchSize);
  }
}

void RunDefaultScheduleXGBoostBenchmarks(int32_t batchSize) {
  RunAllBenchmarks(nullptr, batchSize, false, "array-default_sched");
  
  decisionforest::UseSparseTreeRepresentation = true;
  RunAllBenchmarks(nullptr, batchSize, false, "sparse-default_sched");
  decisionforest::UseSparseTreeRepresentation = false;
}

void RunOneTreeAtATimeScheduleXGBoostBenchmarks(int32_t batchSize) {
  mlir::decisionforest::ScheduleManipulationFunctionWrapper scheduleManipulator(OneTreeAtATimeSchedule);
  RunAllBenchmarks(&scheduleManipulator, batchSize, false, "array-one_tree");
  
  decisionforest::UseSparseTreeRepresentation = true;
  RunAllBenchmarks(&scheduleManipulator, batchSize, false, "sparse-one_tree");
  decisionforest::UseSparseTreeRepresentation = false;
}

void RunProbabilisticOneTreeAtATimeScheduleXGBoostBenchmarks(int32_t batchSize) {
  // There is too much of a blow up of model size when we use array + prob tiling. 
  // So not measuring that.
  mlir::decisionforest::ScheduleManipulationFunctionWrapper scheduleManipulator(OneTreeAtATimeSchedule);
  
  decisionforest::UseSparseTreeRepresentation = true;
  RunAllBenchmarks(&scheduleManipulator, batchSize, true, "sparse-prob-one_tree");
  decisionforest::UseSparseTreeRepresentation = false;
}

void RunProbabilisticOneTreeAtATimeSchedule_RemoveExtraHop_XGBoostBenchmarks(int32_t batchSize) {
  // There is too much of a blow up of model size when we use array + prob tiling. 
  // So not measuring that.
  mlir::decisionforest::ScheduleManipulationFunctionWrapper scheduleManipulator(OneTreeAtATimeSchedule);
  
  decisionforest::UseSparseTreeRepresentation = true;
  decisionforest::PeeledCodeGenForProbabiltyBasedTiling = true;
  RunAllBenchmarks(&scheduleManipulator, batchSize, true, "sparse_remove_extra_hop-prob-one_tree");
  decisionforest::UseSparseTreeRepresentation = false;
  decisionforest::PeeledCodeGenForProbabiltyBasedTiling = false;
}

void RunPipeliningBenchmarks(int32_t batchSize) {
  RunAllPipelinedBenchmarks(batchSize, "array-pipelined_sched");
  
  decisionforest::UseSparseTreeRepresentation = true;
  RunAllPipelinedBenchmarks(batchSize, "sparse-pipelined_sched");
  decisionforest::UseSparseTreeRepresentation = false;
}

void RunParallelPipeliningBenchmarks(int32_t batchSize, int32_t numCores) {
  RunAllPipelinedBenchmarks(batchSize, "array-pipelined_sched-par", numCores);
  
  decisionforest::UseSparseTreeRepresentation = true;
  RunAllPipelinedBenchmarks(batchSize, "sparse-pipelined_sched-par", numCores);
  decisionforest::UseSparseTreeRepresentation = false;
}

void RunSparseParallelPipeliningBenchmarks(int32_t batchSize, int32_t numCores) {
  decisionforest::UseSparseTreeRepresentation = true;
  RunAllPipelinedBenchmarks(batchSize, "sparse-pipelined_sched-par", numCores);
  decisionforest::UseSparseTreeRepresentation = false;
}

void RunBenchmarksOverDifferentNumCores(int32_t batchSize) {
  std::vector<int32_t> numberOfCores{2, 4, 8, 16};
  for (auto numCores : numberOfCores)
    RunSparseParallelPipeliningBenchmarks(batchSize, numCores);
}

void RunXGBoostBenchmarks() {
  std::vector<int32_t> batchSizes{64, 128, 256, 512, 1024, 2000};
  for (auto batchSize : batchSizes) {
    RunOneTreeAtATimeScheduleXGBoostBenchmarks(batchSize);
    RunProbabilisticOneTreeAtATimeSchedule_RemoveExtraHop_XGBoostBenchmarks(batchSize);
    RunPipeliningBenchmarks(batchSize);
    // RunParallelPipeliningBenchmarks(batchSize, 16);
    // RunBenchmarksOverDifferentNumCores(batchSize);
  }
}

void RunXGBoostParallelBenchmarks() {
  std::vector<int32_t> batchSizes{64, 128, 256, 512, 1024, 2000};
  for (auto batchSize : batchSizes) {
    RunParallelPipeliningBenchmarks(batchSize, 16 /*numCores*/);
  }
}

} // test
} // TreeBeard
