#ifdef TREEBEARD_GPU_SUPPORT

#include <chrono>
#include <cstdint>
#include <dlfcn.h>
#include <filesystem>
#include <sstream>
#include <thread>
#include <vector>

#include "ExecutionHelpers.h"
#include "TiledTree.h"
#include "TreeTilingUtils.h"
#include "forestcreator.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "xgboostparser.h"
#include "llvm/ADT/STLExtras.h"

#include "CompileUtils.h"
#include "ForestTestUtils.h"
#include "GPUCompileUtils.h"
#include "GPUExecutionHelper.h"
#include "GPUModelSerializers.h"
#include "GPUSchedules.h"
#include "GPUSupportUtils.h"
#include "GPUTestUtils.h"
#include "LowerReduceOps.h"
#include "ModelSerializers.h"
#include "ReorgForestRepresentation.h"
#include "Representations.h"
#include "TestUtilsCommon.h"
#include <cassert>

#include "GPUBenchmarkUtils.h"

using namespace mlir;

using mlir::decisionforest::GPUBasicSchedule;
using mlir::decisionforest::TahoeSharedDataStrategy;
using mlir::decisionforest::TahoeSharedDataStrategy_Modified;
using mlir::decisionforest::TahoeSharedForestStrategy;
using mlir::decisionforest::TahoeSharedPartialForestStrategy;

// std::vector<int32_t> batchSizes{32,   64,   256,  512, 1024, 2048, 4096,
//                                 8192, 16384};
// std::vector<int32_t> rowsPerTB{8, 16, 32, 64, 128, 256, 512, 1024, 2048};
// std::vector<int32_t> rowsPerThread{1, 2, 4, 8, 16, 32, 64, 128, 256};

std::vector<int32_t> batchSizes{4096, 8192, 16384};
std::vector<int32_t> numTreeThreads{2, 10, 20, 50};
std::vector<int32_t> rowsPerTB{2, 8, 32, 64, 256}; //, 512};
std::vector<int32_t> rowsPerThread{1, 2};          //, 4};
std::vector<int32_t> interleaveDepth{2, 4};

// TODO_Ashwin the compiler is not currently equiped to handle tile size 16!
// there are 35357670 tile shapes. We need to change
// to only compute LUT, tile shape IDs etc for required tile shapes.
std::vector<int32_t> tileSizes{4, 8}; //, 16};
std::vector<int32_t> treeInterleaveDepths{-1, 2, 4};

// abalone
// std::vector<int32_t> numTreeThreads{2, 10, 25};
// std::vector<int32_t> rowsPerTB{32, 64, 256}; //, 512};
// std::vector<int32_t> rowsPerThread{1, 2, 4};
// std::vector<int32_t> interleaveDepth{2, 4};

// airline, higgs
// std::vector<int32_t> numTreeThreads{2, 10, 25};
// std::vector<int32_t> rowsPerTB{32, 64, 256}; //, 512};
// std::vector<int32_t> rowsPerThread{1, 2, 4};
// std::vector<int32_t> interleaveDepth{2};

// airline-ohe
// std::vector<int32_t> numTreeThreads{10, 25, 50}; //, 20, 25};
// std::vector<int32_t> rowsPerTB{4, 8, 16};
// std::vector<int32_t> rowsPerThread{1};
// std::vector<int32_t> interleaveDepth{2, 4};

// epsilon
// std::vector<int32_t> numTreeThreads{25, 50}; //, 20, 25};
// std::vector<int32_t> rowsPerTB{4};
// std::vector<int32_t> rowsPerThread{1};
// std::vector<int32_t> interleaveDepth{2};

// std::vector<int32_t> batchSizes{1024};
// std::vector<int32_t> numTreeThreads{2, 4, 10, 20, 50, 100};
// std::vector<int32_t> rowsPerTB{8, 16, 32, 64};
// std::vector<int32_t> rowsPerThread{1};
// std::vector<int32_t> interleaveDepth{4};
// std::vector<int32_t> treeInterleaveDepths{-1};

namespace TreeBeard {
namespace test {

template <typename ThresholdType, typename ReturnType, bool unrollTreeWalk>
void RunAutoScheduleBenchmarks(const std::string &modelName,
                               const std::string &representationName,
                               int32_t batchSize, int32_t numRowsPerTB,
                               int32_t numRowsPerThread, int32_t numTreeThreads,
                               int32_t treeInterleaveDepth) {

  // Interleaving is a no-op if unrolling is disabled
  if (!unrollTreeWalk && treeInterleaveDepth != -1)
    return;

  if (numRowsPerThread >= numRowsPerTB)
    return;

  BenchmarkIfNoSharedMemOverflow<ThresholdType, ReturnType, false, false,
                                 unrollTreeWalk>(
      modelName, representationName, batchSize, numRowsPerTB, numRowsPerThread,
      numTreeThreads, treeInterleaveDepth);

  BenchmarkIfNoSharedMemOverflow<ThresholdType, ReturnType, true, false,
                                 unrollTreeWalk>(
      modelName, representationName, batchSize, numRowsPerTB, numRowsPerThread,
      numTreeThreads, treeInterleaveDepth);

  // TODO_Ashwin: There is a bug with tree interleaving and tree caching
  // We'll worry about it later
  if (treeInterleaveDepth != -1)
    return;

  // Commenting out the following benchmarks for now. They take too long to run
  // and are never faster than the above benchmarks.
  // BenchmarkIfNoSharedMemOverflow<ThresholdType, ReturnType, false, true,
  //                                unrollTreeWalk>(
  //     modelName, representationName, batchSize, numRowsPerTB,
  //     numRowsPerThread, numTreeThreads, treeInterleaveDepth);

  // BenchmarkIfNoSharedMemOverflow<ThresholdType, ReturnType, true, true,
  //                                unrollTreeWalk>(
  //     modelName, representationName, batchSize, numRowsPerTB,
  //     numRowsPerThread, numTreeThreads, treeInterleaveDepth);
}

bool skipAirlineOHE(int32_t numRowsPerTB, int32_t numRowsPerThread,
                    int32_t numTreeThreads, int32_t interleaveDepth) {
  if (numRowsPerTB == 32 && numRowsPerThread == 2 && numTreeThreads == 50 &&
      interleaveDepth == 4)
    return true;
  return false;
}

bool skipCovtype(int32_t numRowsPerTB, int32_t numRowsPerThread,
                 int32_t numTreeThreads, int32_t interleaveDepth,
                 const std::string &rep) {
  if (numRowsPerTB == 32 && numRowsPerThread == 2 && numTreeThreads == 50 &&
      interleaveDepth == 4 && rep == "gpu_reorg")
    return true;
  return false;
}

bool skipLetters(int32_t numRowsPerTB, int32_t numRowsPerThread,
                 int32_t numTreeThreads, int32_t interleaveDepth,
                 const std::string &rep) {
  // if (numTreeThreads >= 20 && interleaveDepth == 4)
  //   return true;
  if (numRowsPerTB == 32 && numRowsPerThread == 2 && numTreeThreads == 50 &&
      interleaveDepth == 2 && rep == "gpu_reorg")
    return true;
  if (numRowsPerTB == 64 && numRowsPerThread == 1 && numTreeThreads == 10 &&
      interleaveDepth == 4 && rep == "gpu_reorg")
    return true;
  return false;
}

void RunAllAutoScheduleXGBoostGPUBenchmarks() {
  std::cout
      << "model,representation,batch_size,rowsPerTB,rowsPerT,"
         "numTreeThreads,treeInterleaveDepth,cache_rows,cache_trees,unroll,"
         "total,kernel"
      << std::endl;

  for (auto batchSize : batchSizes) {
    for (auto numRowsPerTB : rowsPerTB) {

      auto gridSize = batchSize / numRowsPerTB;
      if (gridSize <= 1)
        break;

      for (auto numRowsPerThread : rowsPerThread) {
        for (auto numTreeThreads : numTreeThreads) {
          for (auto treeInterleaveDepth : treeInterleaveDepths) {
            for (auto rep : {"gpu_array", "gpu_sparse", "gpu_reorg"}) {

              auto tbSize = numRowsPerTB / numRowsPerThread * numTreeThreads;
              if (tbSize < MIN_TB_SIZE)
                continue;
              if (tbSize > MAX_TB_SIZE)
                break;

              RunAutoScheduleBenchmarks<float, float, false>(
                  "abalone", rep, batchSize, numRowsPerTB, numRowsPerThread,
                  numTreeThreads, treeInterleaveDepth);
              RunAutoScheduleBenchmarks<float, float, false>(
                  "airline", rep, batchSize, numRowsPerTB, numRowsPerThread,
                  numTreeThreads, treeInterleaveDepth);
              RunAutoScheduleBenchmarks<float, float, false>(
                  "airline-ohe", rep, batchSize, numRowsPerTB, numRowsPerThread,
                  numTreeThreads, treeInterleaveDepth);
              RunAutoScheduleBenchmarks<float, int8_t, false>(
                  "covtype", rep, batchSize, numRowsPerTB, numRowsPerThread,
                  numTreeThreads, treeInterleaveDepth);
              RunAutoScheduleBenchmarks<float, float, false>(
                  "epsilon", rep, batchSize, numRowsPerTB, numRowsPerThread,
                  numTreeThreads, treeInterleaveDepth);
              RunAutoScheduleBenchmarks<float, float, false>(
                  "higgs", rep, batchSize, numRowsPerTB, numRowsPerThread,
                  numTreeThreads, treeInterleaveDepth);
              RunAutoScheduleBenchmarks<float, int8_t, false>(
                  "letters", rep, batchSize, numRowsPerTB, numRowsPerThread,
                  numTreeThreads, treeInterleaveDepth);
              RunAutoScheduleBenchmarks<float, float, false>(
                  "year_prediction_msd", rep, batchSize, numRowsPerTB,
                  numRowsPerThread, numTreeThreads, treeInterleaveDepth);

              if (PAD_TREES == false)
                continue;

              RunAutoScheduleBenchmarks<float, float, true>(
                  "abalone", rep, batchSize, numRowsPerTB, numRowsPerThread,
                  numTreeThreads, treeInterleaveDepth);
              RunAutoScheduleBenchmarks<float, float, true>(
                  "airline", rep, batchSize, numRowsPerTB, numRowsPerThread,
                  numTreeThreads, treeInterleaveDepth);
              if (!skipAirlineOHE(numRowsPerTB, numRowsPerThread,
                                  numTreeThreads, treeInterleaveDepth))
                RunAutoScheduleBenchmarks<float, float, true>(
                    "airline-ohe", rep, batchSize, numRowsPerTB,
                    numRowsPerThread, numTreeThreads, treeInterleaveDepth);
              if (!skipCovtype(numRowsPerTB, numRowsPerThread, numTreeThreads,
                               treeInterleaveDepth, rep))
                RunAutoScheduleBenchmarks<float, int8_t, true>(
                    "covtype", rep, batchSize, numRowsPerTB, numRowsPerThread,
                    numTreeThreads, treeInterleaveDepth);
              RunAutoScheduleBenchmarks<float, float, true>(
                  "epsilon", rep, batchSize, numRowsPerTB, numRowsPerThread,
                  numTreeThreads, treeInterleaveDepth);
              RunAutoScheduleBenchmarks<float, float, true>(
                  "higgs", rep, batchSize, numRowsPerTB, numRowsPerThread,
                  numTreeThreads, treeInterleaveDepth);
              if (!skipLetters(numRowsPerTB, numRowsPerThread, numTreeThreads,
                               treeInterleaveDepth, rep))
                RunAutoScheduleBenchmarks<float, int8_t, true>(
                    "letters", rep, batchSize, numRowsPerTB, numRowsPerThread,
                    numTreeThreads, treeInterleaveDepth);
              RunAutoScheduleBenchmarks<float, float, true>(
                  "year_prediction_msd", rep, batchSize, numRowsPerTB,
                  numRowsPerThread, numTreeThreads, treeInterleaveDepth);
            }
          }
        }
      }
    }
  }

  mlir::decisionforest::measureGpuKernelTime = false;
}

// ===---------------------------------------------------=== //
// GPU Custom Schedule Benchmarks - Helpers
// ===---------------------------------------------------=== //

template <typename ThresholdType, typename IndexType, typename ReturnType>
GPUTimes BenchmarkCustomScheduleCodeGeneration(
    const int32_t batchSize, const int32_t tileSize,
    ForestCreator &forestCreator, const std::string &modelPath,
    const std::string &modelGlobalsJSONPath,
    std::shared_ptr<decisionforest::IModelSerializer> serializer,
    std::shared_ptr<decisionforest::IRepresentation> representation,
    std::function<void(decisionforest::Schedule &)> scheduleManipulator) {

  const int32_t tileShapeBitwidth = 16;
  const int32_t childIndexBitWidth = 16;

  decisionforest::ScheduleManipulationFunctionWrapper
      scheduleManipulatorWrapper(scheduleManipulator);

  TreeBeard::CompilerOptions options(
      sizeof(FloatType) * 8, sizeof(FloatType) * 8, true, sizeof(IndexType) * 8,
      sizeof(IndexType) * 8, sizeof(FloatType) * 8, batchSize, tileSize,
      tileShapeBitwidth, childIndexBitWidth, TreeBeard::TilingType::kUniform,
      PAD_TREES, false, &scheduleManipulatorWrapper);

  // [HACK!] Create a shared pointer that points to forestCreator
  std::shared_ptr<ForestCreator> forestCreatorPtr(&forestCreator,
                                                  NoOpDeleter());

  TreeBeard::TreebeardContext tbContext(
      forestCreator.GetContext(), modelPath, modelGlobalsJSONPath, options,
      representation, serializer, forestCreatorPtr);

  auto module = ConstructGPUModuleFromTreebeardContext(tbContext);

  if (tbContext.gpuCompileInfo.sharedMemoryInBytes >= MAX_SHMEM_SIZE) {
    return GPUTimes{-1, -1};
  }

  return BenchmarkGPUCodeGeneration<ThresholdType, IndexType, ReturnType>(
      module, tbContext);
}

template <typename ThresholdType, typename ReturnType>
void RunCustomScheduleBenchmarks(
    const std::string &configName, const std::string &xgboostModelFile,
    const std::string representationName, int32_t batchSize, int32_t tileSize,
    std::function<void(decisionforest::Schedule &)> scheduleManipulator) {
  using IndexType = int16_t;

  auto xgboostModelPath = GetXGBoostModelPath(xgboostModelFile);
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          xgboostModelPath);

  auto serializer =
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          representationName, modelGlobalsJSONPath);
  auto representation =
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          representationName);

  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);
  TreeBeard::XGBoostJSONParser<ThresholdType, ReturnType, IndexType, IndexType,
                               ThresholdType>
      xgBoostParser(context, xgboostModelPath, serializer, batchSize);

  auto time = BenchmarkCustomScheduleCodeGeneration<ThresholdType, IndexType,
                                                    ReturnType>(
      batchSize, tileSize, xgBoostParser, xgboostModelPath,
      modelGlobalsJSONPath, serializer, representation, scheduleManipulator);

  std::cout << configName << "," << xgboostModelFile << ","
            << representationName << "," << batchSize << ","
            << "false"
            << "," << time.totalTimePerSample << "," << time.kernelTimePerSample
            << std::endl;
}

bool returnFalse(const std::string &benchmarkName) { return false; }

void RunCustomScheduleXGBoostGPUBenchmarks(
    const std::string &configName, const std::string representationName,
    int32_t batchSize,
    std::function<void(decisionforest::Schedule &)> scheduleManipulator,
    int32_t tileSize = 1,
    std::function<bool(const std::string &)> skipBenchmark = returnFalse) {

  mlir::decisionforest::measureGpuKernelTime = true;
  // std::cout << "config,model,representation,batch_size,unroll,total,kernel"
  //           << std::endl;

  std::vector<std::string> benchmarks{
      "abalone_xgb_model_save.json", "airline_xgb_model_save.json",
      "airline-ohe_xgb_model_save.json",
      // "bosch_xgb_model_save.json",
      "covtype_xgb_model_save.json", "epsilon_xgb_model_save.json",
      "higgs_xgb_model_save.json", "letters_xgb_model_save.json",
      "year_prediction_msd_xgb_model_save.json"};

  std::vector<bool> isMultiClass{false, false, false, // false,
                                 true,  false, false, true, false};

  for (size_t i = 0; i < benchmarks.size(); ++i) {
    if (skipBenchmark(benchmarks[i])) {
      std::cout << configName << "," << benchmarks[i] << ","
                << representationName << "," << batchSize << ","
                << "false"
                << "," << -1 << "," << -1 << std::endl;
      continue;
    }
    if (isMultiClass[i]) {
      assert(benchmarks[i] == "covtype_xgb_model_save.json" ||
             benchmarks[i] == "letters_xgb_model_save.json");
      RunCustomScheduleBenchmarks<float, int8_t>(configName, benchmarks[i],
                                                 representationName, batchSize,
                                                 tileSize, scheduleManipulator);
    } else {
      assert(benchmarks[i] != "covtype_xgb_model_save.json" &&
             benchmarks[i] != "letters_xgb_model_save.json");
      RunCustomScheduleBenchmarks<float, float>(configName, benchmarks[i],
                                                representationName, batchSize,
                                                tileSize, scheduleManipulator);
    }
  }

  mlir::decisionforest::measureGpuKernelTime = false;
}

// ===---------------------------------------------------=== //
// GPU Custom Schedule Benchmarks
// ===---------------------------------------------------=== //

void RunAllSimpleGPUScheduleBenchmarks() {
  for (auto batchSize : batchSizes) {
    for (auto numRowsPerTB : rowsPerTB) {
      if (numRowsPerTB > batchSize)
        break;

      auto scheduleManipulator = std::bind(decisionforest::GPUBasicSchedule,
                                           std::placeholders::_1, numRowsPerTB);
      std::string configName = "basic-" + std::to_string(numRowsPerTB);
      RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_array", batchSize,
                                            scheduleManipulator);
      RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_sparse", batchSize,
                                            scheduleManipulator);
      RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_reorg", batchSize,
                                            scheduleManipulator);
    }
  }
}

void RunAllSimpleGPUScheduleWithRowCacheBenchmarks() {
  for (auto batchSize : batchSizes) {
    for (auto numRowsPerTB : rowsPerTB) {
      if (numRowsPerTB > batchSize)
        break;

      auto scheduleManipulator =
          std::bind(decisionforest::GPUBasicScheduleCacheRows,
                    std::placeholders::_1, numRowsPerTB);
      std::string configName = "basic-cacheRow-" + std::to_string(numRowsPerTB);
      RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_array", batchSize,
                                            scheduleManipulator);
      RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_sparse", batchSize,
                                            scheduleManipulator);
      RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_reorg", batchSize,
                                            scheduleManipulator);
    }
  }
}

void RunAllOneTreeAtATimeGPUScheduleBenchmarks() {
  for (auto batchSize : batchSizes) {
    for (auto numRowsPerTB : rowsPerTB) {
      if (numRowsPerTB > batchSize)
        break;
      for (auto numRowsPerThread : rowsPerThread) {
        if (numRowsPerThread >= numRowsPerTB)
          break;

        auto scheduleManipulator =
            std::bind(decisionforest::OneTreeAtATimeGPUSchedule,
                      std::placeholders::_1, numRowsPerTB, numRowsPerThread);
        std::string configName = "oneTree-" + std::to_string(numRowsPerTB) +
                                 "-" + std::to_string(numRowsPerThread);
        RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_array",
                                              batchSize, scheduleManipulator);
        RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_sparse",
                                              batchSize, scheduleManipulator);
        RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_reorg",
                                              batchSize, scheduleManipulator);
      }
    }
  }
}

void RunOneTreeAtATimeAndCacheRowsGPUScheduleBenchmarks() {

  for (auto batchSize : batchSizes) {
    for (auto numRowsPerTB : rowsPerTB) {
      if (numRowsPerTB > batchSize)
        break;
      for (auto numRowsPerThread : rowsPerThread) {
        if (numRowsPerThread >= numRowsPerTB)
          break;
        auto tbSize = numRowsPerTB / numRowsPerThread;
        if (tbSize < MIN_TB_SIZE)
          continue;
        if (tbSize > MAX_TB_SIZE)
          break;

        auto scheduleManipulator =
            std::bind(decisionforest::OneTreeAtATimeCacheRowsGPUSchedule,
                      std::placeholders::_1, numRowsPerTB, numRowsPerThread);
        std::string configName = "onetree-cacherow-" +
                                 std::to_string(numRowsPerTB) + "-" +
                                 std::to_string(numRowsPerThread);
        RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_array",
                                              batchSize, scheduleManipulator);
        RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_sparse",
                                              batchSize, scheduleManipulator);
        RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_reorg",
                                              batchSize, scheduleManipulator);
      }
    }
  }
}

void RunOneTreeAtATimeAndCacheTreesGPUScheduleBenchmarks() {

  for (auto batchSize : batchSizes) {
    for (auto numRowsPerTB : rowsPerTB) {
      if (numRowsPerTB > batchSize)
        break;
      for (auto numRowsPerThread : rowsPerThread) {
        if (numRowsPerThread >= numRowsPerTB)
          break;
        auto tbSize = numRowsPerTB / numRowsPerThread;
        if (tbSize < MIN_TB_SIZE)
          continue;
        if (tbSize > MAX_TB_SIZE)
          break;

        auto scheduleManipulator =
            std::bind(decisionforest::OneTreeAtATimeCacheTreeGPUSchedule,
                      std::placeholders::_1, numRowsPerTB, numRowsPerThread);
        std::string configName = "onetree-cachetree-" +
                                 std::to_string(numRowsPerTB) + "-" +
                                 std::to_string(numRowsPerThread);
        RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_array",
                                              batchSize, scheduleManipulator);
        RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_sparse",
                                              batchSize, scheduleManipulator);
        RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_reorg",
                                              batchSize, scheduleManipulator);
      }
    }
  }
}

void RunOneTreeAtATimeAndCacheTreesAndRowsGPUScheduleBenchmarks() {

  for (auto batchSize : batchSizes) {
    for (auto numRowsPerTB : rowsPerTB) {
      if (numRowsPerTB > batchSize)
        break;
      for (auto numRowsPerThread : rowsPerThread) {
        if (numRowsPerThread >= numRowsPerTB)
          break;
        auto tbSize = numRowsPerTB / numRowsPerThread;
        if (tbSize < MIN_TB_SIZE)
          continue;
        if (tbSize > MAX_TB_SIZE)
          break;

        auto scheduleManipulator = std::bind(
            decisionforest::OneTreeAtATimeCacheRowsAndTreesGPUSchedule,
            std::placeholders::_1, numRowsPerTB, numRowsPerThread);
        std::string configName = "onetree-cachetreeAndRow-" +
                                 std::to_string(numRowsPerTB) + "-" +
                                 std::to_string(numRowsPerThread);
        RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_array",
                                              batchSize, scheduleManipulator);
        RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_sparse",
                                              batchSize, scheduleManipulator);
        RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_reorg",
                                              batchSize, scheduleManipulator);
      }
    }
  }
}

void RunAllTreeParallelizationGPUScheduleBenchmarks() {
  for (auto batchSize : batchSizes) {
    for (auto numRowsPerTB : rowsPerTB) {
      if (numRowsPerTB > batchSize)
        break;
      for (auto numRowsPerThread : rowsPerThread) {
        if (numRowsPerThread >= numRowsPerTB)
          break;
        for (auto treeThreads : numTreeThreads) {
          auto tbSize = numRowsPerTB / numRowsPerThread * treeThreads;
          if (tbSize > MAX_TB_SIZE)
            break;
          if (tbSize < MIN_TB_SIZE)
            continue;
          auto scheduleManipulator =
              std::bind(decisionforest::SplitTreesAcrossThreadsGPUSchedule,
                        std::placeholders::_1, numRowsPerTB, numRowsPerThread,
                        treeThreads);
          std::string configName = "treepar-" + std::to_string(numRowsPerTB) +
                                   "-" + std::to_string(numRowsPerThread) +
                                   "-" + std::to_string(treeThreads);
          RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_array",
                                                batchSize, scheduleManipulator);
          RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_sparse",
                                                batchSize, scheduleManipulator);
          RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_reorg",
                                                batchSize, scheduleManipulator);
        }
      }
    }
  }
}

void RunAllTreeParallelizationAndCacheRowsGPUScheduleBenchmarks() {

  for (auto batchSize : batchSizes) {
    for (auto numRowsPerTB : rowsPerTB) {
      if (numRowsPerTB > batchSize)
        break;
      for (auto numRowsPerThread : rowsPerThread) {
        if (numRowsPerThread >= numRowsPerTB)
          break;
        for (auto treeThreads : numTreeThreads) {
          auto tbSize = numRowsPerTB / numRowsPerThread * treeThreads;
          if (tbSize > MAX_TB_SIZE)
            break;
          if (tbSize < MIN_TB_SIZE)
            continue;
          auto scheduleManipulator = std::bind(
              decisionforest::SplitTreesAcrossThreadsAndCacheRowsGPUSchedule,
              std::placeholders::_1, numRowsPerTB, numRowsPerThread,
              treeThreads);
          std::string configName = "treepar-cacherow-" +
                                   std::to_string(numRowsPerTB) + "-" +
                                   std::to_string(numRowsPerThread) + "-" +
                                   std::to_string(treeThreads);
          RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_array",
                                                batchSize, scheduleManipulator);
          RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_sparse",
                                                batchSize, scheduleManipulator);
          RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_reorg",
                                                batchSize, scheduleManipulator);
        }
      }
    }
  }
}

void RunAllTreeParallelizationAndCacheRowsGPUScheduleBenchmarks_Reorg() {

  for (auto batchSize : batchSizes) {
    for (auto numRowsPerTB : rowsPerTB) {
      if (numRowsPerTB > batchSize)
        break;
      for (auto numRowsPerThread : rowsPerThread) {
        if (numRowsPerThread >= numRowsPerTB)
          break;
        for (auto treeThreads : numTreeThreads) {
          auto tbSize = numRowsPerTB / numRowsPerThread * treeThreads;
          if (tbSize > MAX_TB_SIZE)
            break;
          if (tbSize < MIN_TB_SIZE)
            continue;
          auto scheduleManipulator = std::bind(
              decisionforest::
                  SplitTreesAcrossThreadsAndCacheRowsGPUSchedule_Reorg,
              std::placeholders::_1, numRowsPerTB, numRowsPerThread,
              treeThreads);
          std::string configName = "treepar-cacherow-reorg-" +
                                   std::to_string(numRowsPerTB) + "-" +
                                   std::to_string(numRowsPerThread) + "-" +
                                   std::to_string(treeThreads);
          RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_reorg",
                                                batchSize, scheduleManipulator);
        }
      }
    }
  }
}

void RunAllTreeParallelizationCacheRowsAndInterleaveTreesGPUScheduleBenchmarks() {
  for (auto batchSize : batchSizes) {
    for (auto numRowsPerTB : rowsPerTB) {
      if (numRowsPerTB > batchSize)
        break;
      for (auto numRowsPerThread : rowsPerThread) {
        if (numRowsPerThread >= numRowsPerTB)
          break;
        for (auto treeThreads : numTreeThreads) {
          auto tbSize = numRowsPerTB / numRowsPerThread * treeThreads;
          if (tbSize > MAX_TB_SIZE)
            break;
          if (tbSize < MIN_TB_SIZE)
            continue;
          for (auto depth : interleaveDepth) {
            auto scheduleManipulator = std::bind(
                decisionforest::
                    SplitTreesAcrossThreadsCacheRowsAndInterleaveTreesGPUSchedule,
                std::placeholders::_1, numRowsPerTB, numRowsPerThread,
                treeThreads, depth);
            std::string configName = "treepar-cacherow-interleavetrees-" +
                                     std::to_string(numRowsPerTB) + "-" +
                                     std::to_string(numRowsPerThread) + "-" +
                                     std::to_string(treeThreads) + "-" +
                                     std::to_string(depth);
            RunCustomScheduleXGBoostGPUBenchmarks(
                configName, "gpu_array", batchSize, scheduleManipulator);
            RunCustomScheduleXGBoostGPUBenchmarks(
                configName, "gpu_sparse", batchSize, scheduleManipulator);
            RunCustomScheduleXGBoostGPUBenchmarks(
                configName, "gpu_reorg", batchSize, scheduleManipulator);
          }
        }
      }
    }
  }
}

void RunAllTreeParallelizationCacheRowsAndInterleaveRowsGPUScheduleBenchmarks() {
  for (auto batchSize : batchSizes) {
    for (auto numRowsPerTB : rowsPerTB) {
      if (numRowsPerTB > batchSize)
        break;
      for (auto numRowsPerThread : rowsPerThread) {
        if (numRowsPerThread >= numRowsPerTB)
          break;
        if (numRowsPerThread == 1)
          continue;
        for (auto treeThreads : numTreeThreads) {
          auto tbSize = numRowsPerTB / numRowsPerThread * treeThreads;
          if (tbSize > MAX_TB_SIZE)
            break;
          if (tbSize < MIN_TB_SIZE)
            continue;
          auto scheduleManipulator = std::bind(
              decisionforest::
                  SplitTreesAcrossThreadsCacheRowsAndInterleaveRowsGPUSchedule,
              std::placeholders::_1, numRowsPerTB, numRowsPerThread,
              treeThreads);
          std::string configName = "treepar-cacherow-interleaverows-" +
                                   std::to_string(numRowsPerTB) + "-" +
                                   std::to_string(numRowsPerThread) + "-" +
                                   std::to_string(treeThreads);
          RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_array",
                                                batchSize, scheduleManipulator);
          // RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_sparse",
          //                                       batchSize,
          //                                       scheduleManipulator);
          // RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_reorg",
          //                                       batchSize,
          //                                       scheduleManipulator);
        }
      }
    }
  }
}

void RunAllTreeParallelizationGPUScheduleBenchmarksTiled() {

  for (auto batchSize : batchSizes) {
    for (auto numRowsPerTB : rowsPerTB) {
      if (numRowsPerTB > batchSize)
        break;
      for (auto numRowsPerThread : rowsPerThread) {
        if (numRowsPerThread >= numRowsPerTB)
          break;
        for (auto treeThreads : numTreeThreads) {
          auto tbSize = numRowsPerTB / numRowsPerThread * treeThreads;
          if (tbSize > MAX_TB_SIZE)
            break;
          if (tbSize < MIN_TB_SIZE)
            continue;
          for (auto tileSize : tileSizes) {
            auto scheduleManipulator = std::bind(
                decisionforest::SplitTreesAcrossThreadsGPUSchedule_Tiling,
                std::placeholders::_1, numRowsPerTB, numRowsPerThread,
                treeThreads);
            std::string configName =
                "treepar-tiled-" + std::to_string(numRowsPerTB) + "-" +
                std::to_string(numRowsPerThread) + "-" +
                std::to_string(treeThreads) + "-" + std::to_string(tileSize);
            RunCustomScheduleXGBoostGPUBenchmarks(
                configName, "gpu_array", batchSize, scheduleManipulator,
                tileSize);
            RunCustomScheduleXGBoostGPUBenchmarks(
                configName, "gpu_sparse", batchSize, scheduleManipulator,
                tileSize);
            // RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_reorg",
            //                                       batchSize,
            //                                       scheduleManipulator);
          }
        }
      }
    }
  }
}

void RunAllTreeParallelizationAndCacheRowsGPUScheduleBenchmarksTiled() {

  for (auto batchSize : batchSizes) {
    for (auto numRowsPerTB : rowsPerTB) {
      if (numRowsPerTB > batchSize)
        break;
      for (auto numRowsPerThread : rowsPerThread) {
        if (numRowsPerThread >= numRowsPerTB)
          break;
        for (auto treeThreads : numTreeThreads) {
          auto tbSize = numRowsPerTB / numRowsPerThread * treeThreads;
          if (tbSize > MAX_TB_SIZE)
            break;
          if (tbSize < MIN_TB_SIZE)
            continue;
          for (auto tileSize : tileSizes) {
            auto scheduleManipulator = std::bind(
                decisionforest::
                    SplitTreesAcrossThreadsAndCacheRowsGPUSchedule_Tiling,
                std::placeholders::_1, numRowsPerTB, numRowsPerThread,
                treeThreads);
            std::string configName =
                "treepar-cacherow-tiled-" + std::to_string(numRowsPerTB) + "-" +
                std::to_string(numRowsPerThread) + "-" +
                std::to_string(treeThreads) + "-" + std::to_string(tileSize);
            RunCustomScheduleXGBoostGPUBenchmarks(
                configName, "gpu_array", batchSize, scheduleManipulator,
                tileSize);
            RunCustomScheduleXGBoostGPUBenchmarks(
                configName, "gpu_sparse", batchSize, scheduleManipulator,
                tileSize);
            // RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_reorg",
            //                                       batchSize,
            //                                       scheduleManipulator);
          }
        }
      }
    }
  }
}

void RunAllTreeParallelizationAndCacheTreesGPUScheduleBenchmarks() {

  for (auto batchSize : batchSizes) {
    for (auto numRowsPerTB : rowsPerTB) {
      if (numRowsPerTB > batchSize)
        break;
      for (auto numRowsPerThread : rowsPerThread) {
        if (numRowsPerThread >= numRowsPerTB)
          break;
        for (auto treeThreads : numTreeThreads) {
          auto tbSize = numRowsPerTB / numRowsPerThread * treeThreads;
          if (tbSize > MAX_TB_SIZE)
            break;
          if (tbSize < MIN_TB_SIZE)
            continue;
          auto scheduleManipulator = std::bind(
              decisionforest::SplitTreesAcrossThreadsAndCacheTreesGPUSchedule,
              std::placeholders::_1, numRowsPerTB, numRowsPerThread,
              treeThreads);
          std::string configName = "treepar-cachetree-" +
                                   std::to_string(numRowsPerTB) + "-" +
                                   std::to_string(numRowsPerThread) + "-" +
                                   std::to_string(treeThreads);
          RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_array",
                                                batchSize, scheduleManipulator);
          RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_sparse",
                                                batchSize, scheduleManipulator);
          RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_reorg",
                                                batchSize, scheduleManipulator);
        }
      }
    }
  }
}

void RunAllTreeParallelizationCacheTreesAndRowsGPUScheduleBenchmarks() {

  for (auto batchSize : batchSizes) {
    for (auto numRowsPerTB : rowsPerTB) {
      if (numRowsPerTB > batchSize)
        break;
      for (auto numRowsPerThread : rowsPerThread) {
        if (numRowsPerThread >= numRowsPerTB)
          break;
        for (auto treeThreads : numTreeThreads) {
          auto tbSize = numRowsPerTB / numRowsPerThread * treeThreads;
          if (tbSize > MAX_TB_SIZE)
            break;
          if (tbSize < MIN_TB_SIZE)
            continue;
          auto scheduleManipulator = std::bind(
              decisionforest::
                  SplitTreesAcrossThreadsAndCacheTreesAndRowsGPUSchedule,
              std::placeholders::_1, numRowsPerTB, numRowsPerThread,
              treeThreads);
          std::string configName = "treepar-cachetreeAndRows-" +
                                   std::to_string(numRowsPerTB) + "-" +
                                   std::to_string(numRowsPerThread) + "-" +
                                   std::to_string(treeThreads);
          RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_array",
                                                batchSize, scheduleManipulator);
          RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_sparse",
                                                batchSize, scheduleManipulator);
          RunCustomScheduleXGBoostGPUBenchmarks(configName, "gpu_reorg",
                                                batchSize, scheduleManipulator);
        }
      }
    }
  }
}

void RunAllCustomScheduleBenchmarks() {
  RunAllSimpleGPUScheduleBenchmarks();
  RunAllSimpleGPUScheduleWithRowCacheBenchmarks();

  // One tree at a time benchmarks
  // RunAllOneTreeAtATimeGPUScheduleBenchmarks();
  RunOneTreeAtATimeAndCacheRowsGPUScheduleBenchmarks();
  // RunOneTreeAtATimeAndCacheTreesGPUScheduleBenchmarks();
  // RunOneTreeAtATimeAndCacheTreesAndRowsGPUScheduleBenchmarks();

  // Tree parallelization benchmarks
  RunAllTreeParallelizationGPUScheduleBenchmarks();
  RunAllTreeParallelizationAndCacheRowsGPUScheduleBenchmarks();
  RunAllTreeParallelizationAndCacheRowsGPUScheduleBenchmarks_Reorg();
  // RunAllTreeParallelizationAndCacheTreesGPUScheduleBenchmarks();
  // RunAllTreeParallelizationCacheTreesAndRowsGPUScheduleBenchmarks();

  // Interleaving trees
  // RunAllTreeParallelizationCacheRowsAndInterleaveTreesGPUScheduleBenchmarks();
  // RunAllTreeParallelizationCacheRowsAndInterleaveRowsGPUScheduleBenchmarks();

  // Tiled benchmarks
  // RunAllTreeParallelizationAndCacheRowsGPUScheduleBenchmarksTiled();
  // RunAllTreeParallelizationGPUScheduleBenchmarksTiled();
}

void RunXGBoostGPUBenchmarks() {
  mlir::decisionforest::measureGpuKernelTime = true;
  // mlir::decisionforest::numberOfKernelRuns = NUM_RUNS;
  // RunAllAutoScheduleXGBoostGPUBenchmarks();
  // RunAllCustomScheduleBenchmarks();
  std::vector<int32_t> batchSizes{256, 512, 1024, 4096, 8192, 16384};
  for (auto batchSize : batchSizes)
    findBestGPUSchedule(GetXGBoostModelPath("abalone_xgb_model_save.json"),
                        batchSize);
}

} // namespace test
} // namespace TreeBeard

#endif // TREEBEARD_GPU_SUPPORT
