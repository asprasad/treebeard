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

using namespace mlir;

using mlir::decisionforest::GPUBasicSchedule;
using mlir::decisionforest::TahoeSharedDataStrategy;
using mlir::decisionforest::TahoeSharedDataStrategy_Modified;
using mlir::decisionforest::TahoeSharedForestStrategy;
using mlir::decisionforest::TahoeSharedPartialForestStrategy;

#define NUM_RUNS 200
#define VERIFY_RESULT false

const int32_t MIN_TB_SIZE = 32;
const int32_t MAX_TB_SIZE = 1024;
const int32_t MAX_SHMEM_SIZE = 49152;
const bool PAD_TREES = true;
constexpr bool SH_MEM_REDUCE = false;

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

struct GPUBenchmarkExecState {
  std::shared_ptr<TreeBeard::TreebeardContext> tbContext;
  mlir::ModuleOp module;
};

struct GPUTimes {
  double totalTimePerSample;
  double kernelTimePerSample;
};

// ===---------------------------------------------------=== //
//  Helpers
// ===---------------------------------------------------=== //

bool useSameRowForAllInputs = false;

class TestDataReader {
  std::map<std::string, std::vector<double>> m_benchmarkToDataMap;

  std::vector<double> readDataForBenchmark(const std::string &benchmarkName) {

    auto modelFilePath = GetXGBoostModelPath(benchmarkName);
    auto csvFilename = modelFilePath + ".test.sampled.csv";

    TestCSVReader csvReader(csvFilename, 2000 /*num lines*/);
    assert(csvReader.NumberOfRows() == 2000);

    std::vector<double> data;
    for (size_t i = 0; i < csvReader.NumberOfRows(); ++i) {
      if (useSameRowForAllInputs) {
        auto row = csvReader.GetRowOfType<double>(0);
        data.insert(data.end(), row.begin(), row.end());
      } else {
        auto row = csvReader.GetRowOfType<double>(i);
        data.insert(data.end(), row.begin(), row.end());
      }
    }

    m_benchmarkToDataMap[benchmarkName] = data;
    m_benchmarkToDataMap[csvFilename] = data;

    return data;
  }

public:
  TestDataReader() {
    std::list<std::string> benchmarks{
        "abalone_xgb_model_save.json",
        "airline_xgb_model_save.json",
        "airline-ohe_xgb_model_save.json",
        "bosch_xgb_model_save.json",
        "covtype_xgb_model_save.json",
        "epsilon_xgb_model_save.json",
        "higgs_xgb_model_save.json",
        "letters_xgb_model_save.json",
        "year_prediction_msd_xgb_model_save.json"};
    for (auto &benchmark : benchmarks) {
      readDataForBenchmark(benchmark);
    }
  }

  int32_t getRowSize(const std::string &benchmarkName) {
    auto iter = m_benchmarkToDataMap.find(benchmarkName);
    assert(iter != m_benchmarkToDataMap.end());
    return iter->second.size() / 2000;
  }

  template <typename T>
  std::vector<T> getData(const std::string &benchmarkName, int32_t batchNum,
                         int32_t batchSize) {
    auto iter = m_benchmarkToDataMap.find(benchmarkName);
    assert(iter != m_benchmarkToDataMap.end());
    auto rowSize = iter->second.size() / 2000;
    auto start = iter->second.begin() + (batchNum * batchSize * rowSize);
    std::vector<T> batch(start, start + batchSize * rowSize);
    return batch;
  }
};

TestDataReader dataReader;

template <typename T>
void populateInputdata(int32_t batchSize,
                       std::vector<std::vector<T>> &inputData,
                       const std::string &csvFileName) {
  const int32_t numRows = 2000;

  if (batchSize < numRows) {
    for (size_t i = batchSize; i <= numRows; i += batchSize) {
      auto batchNum = (i / batchSize) - 1;
      std::vector<T> batch =
          dataReader.getData<T>(csvFileName, batchNum, batchSize);
      inputData.push_back(batch);
    }
  } else {
    auto numRepeat = batchSize / numRows;
    auto remainder = batchSize % numRows;
    std::vector<T> batch,
        allRows = dataReader.getData<T>(csvFileName, 0, numRows);

    for (auto i = 0; i < numRepeat; ++i) {
      batch.insert(batch.end(), allRows.begin(), allRows.end());
    }

    auto rowSize = dataReader.getRowSize(csvFileName);
    batch.insert(batch.end(), allRows.begin(),
                 allRows.begin() + remainder * rowSize);

    inputData.push_back(batch);
  }
}

template <typename ThresholdType, typename IndexType, typename ReturnType>
GPUTimes BenchmarkGPUCodeGeneration(mlir::ModuleOp module,
                                    TreeBeard::TreebeardContext &tbContext) {

  auto batchSize = tbContext.options.batchSize;

  GPUInferenceRunnerForTest inferenceRunner(
      tbContext.serializer, module, tbContext.options.tileSize,
      sizeof(ThresholdType) * 8, sizeof(IndexType) * 8);

  if (VERIFY_RESULT) {
    // NOTE: Setting this calls the kernel in a loop rendering the result
    // incorrect.
    assert(mlir::decisionforest::numberOfKernelRuns == 1 &&
           "Number of kernel runs should be == 1 to verify results!");
    bool validResult =
        ValidateModuleOutputAgainstCSVdata<ThresholdType, ReturnType>(
            inferenceRunner, tbContext.modelPath + ".test.sampled.csv",
            batchSize);
    assert(validResult && "Result validation failed");
  }

  auto csvFilename = tbContext.modelPath + ".test.sampled.csv";

  std::vector<std::vector<ThresholdType>> inputData;
  populateInputdata<ThresholdType>(batchSize, inputData, csvFilename);

  std::vector<ReturnType> result(batchSize, -1);
  bool runKernelMultipleTimes = mlir::decisionforest::numberOfKernelRuns > 1;
  auto numTrials = runKernelMultipleTimes ? 1 : NUM_RUNS;
  auto begin = std::chrono::steady_clock::now();
  for (int32_t trial = 0; trial < numTrials; ++trial) {
    for (auto &batch : inputData) {
      inferenceRunner.RunInference<ThresholdType, ReturnType>(batch.data(),
                                                              result.data());
      // Hack to figure out whether the kernel failed to run
      if (result.at(0) == ReturnType(-1))
        return GPUTimes{-1, -1};
    }
  }
  auto end = std::chrono::steady_clock::now();

  int64_t numKernelCallsPerRun = mlir::decisionforest::numberOfKernelRuns;

  int64_t numSamples =
      numKernelCallsPerRun * numTrials * inputData.size() * batchSize;
  int64_t timeTaken =
      std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
          .count();

  double totalTimePerSample = 0;
  auto kernelTimePerSample =
      (double)inferenceRunner.GetKernelExecutionTime() / (double)numSamples;
  if (runKernelMultipleTimes) {
    auto totalOverhead = timeTaken - inferenceRunner.GetKernelExecutionTime();
    auto overheadPerSample =
        ((double)(totalOverhead) / (batchSize * inputData.size()));
    totalTimePerSample = overheadPerSample + kernelTimePerSample;
  } else {
    totalTimePerSample = (double)timeTaken / (double)numSamples;
  }
  return GPUTimes{totalTimePerSample, kernelTimePerSample};
}

// ===---------------------------------------------------=== //
//  GPU Autoscheduling Benchmark Helpers
// ===---------------------------------------------------=== //

template <typename ThresholdType, typename IndexType, typename ReturnType>
GPUBenchmarkExecState CompileAutoScheduleBenchmark(
    const int32_t batchSize, ForestCreator &forestCreator,
    const std::string &modelPath, const std::string &modelGlobalsJSONPath,
    std::shared_ptr<decisionforest::IModelSerializer> serializer,
    std::shared_ptr<decisionforest::IRepresentation> representation,
    int32_t tileSize, int32_t tileShapeBitwidth, int32_t childIndexBitWidth,
    TreeBeard::GPUAutoScheduleOptions &gpuAutoScheduleOptions) {

  assert(!gpuAutoScheduleOptions.unrollTreeWalks || PAD_TREES);

  TreeBeard::CompilerOptions options(
      sizeof(FloatType) * 8, sizeof(FloatType) * 8, true, sizeof(IndexType) * 8,
      sizeof(IndexType) * 8, sizeof(FloatType) * 8, batchSize, tileSize,
      tileShapeBitwidth, childIndexBitWidth, TreeBeard::TilingType::kUniform,
      gpuAutoScheduleOptions.unrollTreeWalks && PAD_TREES, false, nullptr);

  // [HACK!] Create a shared pointer that points to forestCreator
  std::shared_ptr<ForestCreator> forestCreatorPtr(&forestCreator,
                                                  NoOpDeleter());

  GPUBenchmarkExecState gpuExecInfo;
  auto tbContext = std::make_shared<TreeBeard::TreebeardContext>(
      forestCreator.GetContext(), modelPath, modelGlobalsJSONPath, options,
      representation, serializer, forestCreatorPtr, &gpuAutoScheduleOptions);

  auto module = ConstructGPUModuleFromTreebeardContext(*tbContext);
  gpuExecInfo.module = module;
  gpuExecInfo.tbContext = tbContext;

  return gpuExecInfo;
}

template <typename ThresholdType, typename IndexType, typename ReturnType>
GPUBenchmarkExecState CompileXGBoostAutoScheduleBenchmark(
    const std::string &xgboostModelFile, MLIRContext &context,
    TreeBeard::GPUAutoScheduleOptions &gpuAutoScheduleOptions,
    const std::string &representationName, int32_t batchSize) {
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

  TreeBeard::XGBoostJSONParser<ThresholdType, ReturnType, IndexType, IndexType,
                               ThresholdType>
      xgBoostParser(context, xgboostModelPath, serializer, batchSize);

  return CompileAutoScheduleBenchmark<ThresholdType, IndexType, ReturnType>(
      batchSize, xgBoostParser, xgboostModelPath, modelGlobalsJSONPath,
      serializer, representation,
      1,                     // Tile size
      16,                    // Tile shape width
      sizeof(IndexType) * 8, // child index width
      gpuAutoScheduleOptions);
}

template <typename ThresholdType, typename ReturnType, bool cacheRows,
          bool cacheTrees, bool unrollTreeWalk, bool sharedMemoryReduce>
GPUBenchmarkExecState CompileXGBoostAutoScheduleBenchmark(
    const std::string &xgboostModelFile, MLIRContext &context,
    const std::string representationName, int32_t batchSize,
    int32_t numRowsPerTB, int32_t numRowsPerThread, int32_t numTreeThreads,
    int numTreesAtATime, int treeInterleaveDepth) {
  using IndexType = int16_t;

  TreeBeard::GPUAutoScheduleOptions gpuAutoScheduleOptions{
      numRowsPerTB,        numRowsPerThread,  -1,         numTreeThreads,
      numTreesAtATime,     cacheRows,         cacheTrees, unrollTreeWalk,
      treeInterleaveDepth, sharedMemoryReduce};

  return CompileXGBoostAutoScheduleBenchmark<ThresholdType, IndexType,
                                             ReturnType>(
      xgboostModelFile, context, gpuAutoScheduleOptions, representationName,
      batchSize);
}

template <typename ThresholdType, typename ReturnType, bool cacheRows,
          bool cacheTrees, bool unrollTreeWalk,
          bool sharedMemoryReduce = SH_MEM_REDUCE>
GPUTimes BenchmarkIfNoSharedMemOverflow(const std::string &modelName,
                                        const std::string &representationName,
                                        int32_t batchSize, int32_t numRowsPerTB,
                                        int32_t numRowsPerThread,
                                        int32_t numTreeThreads,
                                        int32_t treeInterleaveDepth) {
  using IndexType = int16_t;

  auto xgboostModelFile = modelName + "_xgb_model_save.json";
  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);
  auto execInfo =
      CompileXGBoostAutoScheduleBenchmark<ThresholdType, ReturnType, cacheRows,
                                          cacheTrees, unrollTreeWalk,
                                          sharedMemoryReduce>(
          xgboostModelFile, context, representationName, batchSize,
          numRowsPerTB, numRowsPerThread, numTreeThreads, 1,
          treeInterleaveDepth);

  if (execInfo.tbContext->gpuCompileInfo.sharedMemoryInBytes <=
      MAX_SHMEM_SIZE) {

    auto times =
        BenchmarkGPUCodeGeneration<ThresholdType, IndexType, ReturnType>(
            execInfo.module, *execInfo.tbContext);

    std::cout << modelName << "," << representationName << "," << batchSize
              << "," << numRowsPerTB << "," << numRowsPerThread << ","
              << numTreeThreads << "," << treeInterleaveDepth << std::boolalpha
              << "," << cacheRows << "," << cacheTrees << "," << unrollTreeWalk
              << "," << times.totalTimePerSample << ","
              << times.kernelTimePerSample << std::endl;
    return times;
  }
  return GPUTimes{-1, -1};
}

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

// ===---------------------------------------------------=== //
// GPU Auto-tuning Heuristic
// ===---------------------------------------------------=== //

class GPUAutoTuner {
  std::vector<int32_t> rowsPerTB;
  std::vector<int32_t> rowsPerThread;
  std::vector<int32_t> numTreeThreads;
  bool m_shouldTryUnroll = true;
  bool m_shouldCacheRows = true;
  const bool m_verboseLogs = false;

  int32_t m_numFeatures;
  int32_t m_numTrees;
  int32_t m_batchSize;
  bool m_multiClass;

  bool m_shouldPadTrees;
  GPUAutoScheduleOptions m_bestOptions;
  std::string m_bestRep;
  std::string m_model;

  void updateConfig(const GPUAutoScheduleOptions &options,
                    const std::string &rep, bool shouldPadTrees) {
    m_bestOptions = options;
    m_bestRep = rep;
    m_shouldPadTrees = shouldPadTrees;

    if (m_verboseLogs) {
      std::cout << "\nUpdating best configuration\n";
      this->printBestSchedule();
    }
  }

  void computeScheduleSubset() {
    // if batchSize < 1k, then focus on larger number of tree threads
    if (m_batchSize <= 1024) {
      numTreeThreads = {20, 50};
      rowsPerTB = {8, 32};
      rowsPerThread = {1};
    } else {
      const int32_t featureThreshold = 100;
      // if we have a large batch sizes, with a large
      // feature count, use a large number of tree threads
      if (m_numFeatures > featureThreshold) {
        numTreeThreads = {20, 50};
        rowsPerTB = {8, 32};
        rowsPerThread = {1};
      } else {
        numTreeThreads = {1, 2, 10};
        rowsPerTB = {32, 64};
        rowsPerThread = {1};
      }
    }
  }

  void initParameters(const std::string &modelName, int32_t batchSize) {
    std::vector<std::string> benchmarks{"abalone", "airline", "airline-ohe",
                                        // "bosch",
                                        "covtype", "epsilon", "higgs",
                                        "letters", "year_prediction_msd"};

    std::vector<bool> isMultiClass{false, false, false, // false,
                                   true,  false, false, true, false};

    std::vector<int32_t> numFeatures{8, 13, 692, 54, 2000, 28, 16, 90};

    std::vector<int32_t> numTrees{1000, 100, 1000, 800, 100, 100, 26000, 100};

    auto it = std::find(benchmarks.begin(), benchmarks.end(), modelName);
    assert(it != benchmarks.end());
    auto index = std::distance(benchmarks.begin(), it);

    m_numFeatures = numFeatures[index];
    m_numTrees = numTrees[index];
    m_multiClass = isMultiClass[index];

    computeScheduleSubset();

    // If the num features is too large to fit 8 rows
    // into shared mem, then don't try to cache
    if (m_numFeatures > 1500)
      m_shouldCacheRows = false;
  }

  void trySharedReduce(double &bestKernelTime) {
    if (!m_multiClass)
      return;
    std::cout << "Checking shared reduce\n";
    printBestSchedule();
    std::cout << "*************\n";

    if (m_bestOptions.unrollTreeWalks == false) {
      constexpr bool unrollTreeWalks = false;
      GPUAutoScheduleOptions options = m_bestOptions;
      options.sharedMemoryReduce = true;

      auto time = BenchmarkIfNoSharedMemOverflow<float, int8_t, true, false,
                                                 unrollTreeWalks, true>(
          m_model, m_bestRep, m_batchSize, options.numRowsPerTB,
          options.numRowsPerThread, options.numTreeThreads,
          options.treeWalkInterleaveFactor);
      std::cout << time.kernelTimePerSample << std::endl;
      if (time.kernelTimePerSample != -1 &&
          time.kernelTimePerSample < bestKernelTime) {
        updateConfig(options, m_bestRep, false);
        bestKernelTime = time.kernelTimePerSample;
      }

      options.cacheRows = false;
      time = BenchmarkIfNoSharedMemOverflow<float, int8_t, false, false,
                                            unrollTreeWalks, true>(
          m_model, m_bestRep, m_batchSize, options.numRowsPerTB,
          options.numRowsPerThread, options.numTreeThreads,
          options.treeWalkInterleaveFactor);
      std::cout << time.kernelTimePerSample << std::endl;
      if (time.kernelTimePerSample != -1 &&
          time.kernelTimePerSample < bestKernelTime) {
        updateConfig(options, m_bestRep, false);
        bestKernelTime = time.kernelTimePerSample;
      }

    } else {
      constexpr bool unrollTreeWalks = true;
      GPUAutoScheduleOptions options = m_bestOptions;
      options.sharedMemoryReduce = true;

      auto time = BenchmarkIfNoSharedMemOverflow<float, int8_t, true, false,
                                                 unrollTreeWalks, true>(
          m_model, m_bestRep, m_batchSize, options.numRowsPerTB,
          options.numRowsPerThread, options.numTreeThreads,
          options.treeWalkInterleaveFactor);

      if (time.kernelTimePerSample != -1 &&
          time.kernelTimePerSample < bestKernelTime) {
        updateConfig(options, m_bestRep, false);
        bestKernelTime = time.kernelTimePerSample;
      }

      options.cacheRows = false;
      time = BenchmarkIfNoSharedMemOverflow<float, int8_t, false, false,
                                            unrollTreeWalks, true>(
          m_model, m_bestRep, m_batchSize, options.numRowsPerTB,
          options.numRowsPerThread, options.numTreeThreads,
          options.treeWalkInterleaveFactor);

      if (time.kernelTimePerSample != -1 &&
          time.kernelTimePerSample < bestKernelTime) {
        updateConfig(options, m_bestRep, false);
        bestKernelTime = time.kernelTimePerSample;
      }
    }
  }

  template <bool cacheRows>
  void runBenchmarks(double &bestKernelTime, const std::string &rep,
                     int32_t numRowsPerTB, int32_t numRowsPerThread,
                     int32_t treeThreads) {
    constexpr bool cacheTrees = false;
    constexpr bool sharedMemoryReduce = false;
    {
      constexpr bool unrollTreeWalks = false;
      GPUAutoScheduleOptions options{
          numRowsPerTB, numRowsPerThread, 1,  treeThreads,       1, cacheRows,
          cacheTrees,   unrollTreeWalks,  -1, sharedMemoryReduce};

      auto time = m_multiClass
                      ? BenchmarkIfNoSharedMemOverflow<float, int8_t, cacheRows,
                                                       false, unrollTreeWalks>(
                            m_model, rep, m_batchSize, numRowsPerTB,
                            numRowsPerThread, treeThreads, -1)
                      : BenchmarkIfNoSharedMemOverflow<float, float, cacheRows,
                                                       false, unrollTreeWalks>(
                            m_model, rep, m_batchSize, numRowsPerTB,
                            numRowsPerThread, treeThreads, -1);
      if (time.kernelTimePerSample != -1 &&
          time.kernelTimePerSample < bestKernelTime) {
        updateConfig(options, rep, false);
        bestKernelTime = time.kernelTimePerSample;
      }
    }

    // TODO we need to check if we need to check unrolling for the
    // current model and config
    {
      constexpr bool unrollTreeWalks = true;
      constexpr int32_t interleaveDepth = 2;

      GPUAutoScheduleOptions options{numRowsPerTB,
                                     numRowsPerThread,
                                     1,
                                     treeThreads,
                                     1,
                                     cacheRows,
                                     cacheTrees,
                                     unrollTreeWalks,
                                     interleaveDepth,
                                     sharedMemoryReduce};

      auto time = m_multiClass
                      ? BenchmarkIfNoSharedMemOverflow<float, int8_t, cacheRows,
                                                       false, unrollTreeWalks>(
                            m_model, rep, m_batchSize, numRowsPerTB,
                            numRowsPerThread, treeThreads, interleaveDepth)
                      : BenchmarkIfNoSharedMemOverflow<float, float, cacheRows,
                                                       false, unrollTreeWalks>(
                            m_model, rep, m_batchSize, numRowsPerTB,
                            numRowsPerThread, treeThreads, interleaveDepth);
      if (time.kernelTimePerSample != -1 &&
          time.kernelTimePerSample < bestKernelTime) {
        bestKernelTime = time.kernelTimePerSample;
        updateConfig(options, rep, false);
      }
    }

    {
      constexpr bool unrollTreeWalks = true;
      constexpr int32_t interleaveDepth = 4;

      GPUAutoScheduleOptions options{numRowsPerTB,
                                     numRowsPerThread,
                                     1,
                                     treeThreads,
                                     1,
                                     cacheRows,
                                     cacheTrees,
                                     unrollTreeWalks,
                                     interleaveDepth,
                                     sharedMemoryReduce};

      auto time = m_multiClass
                      ? BenchmarkIfNoSharedMemOverflow<float, int8_t, cacheRows,
                                                       false, unrollTreeWalks>(
                            m_model, rep, m_batchSize, numRowsPerTB,
                            numRowsPerThread, treeThreads, interleaveDepth)
                      : BenchmarkIfNoSharedMemOverflow<float, float, cacheRows,
                                                       false, unrollTreeWalks>(
                            m_model, rep, m_batchSize, numRowsPerTB,
                            numRowsPerThread, treeThreads, interleaveDepth);
      if (time.kernelTimePerSample != -1 &&
          time.kernelTimePerSample < bestKernelTime) {
        bestKernelTime = time.kernelTimePerSample;
        updateConfig(options, rep, false);
      }
    }
  }

public:
  GPUAutoTuner(const std::string &modelName, int32_t batchSize)
      : m_batchSize(batchSize), m_model(modelName) {
    initParameters(modelName, batchSize);
  }

  void exploreSchedules() {

    const auto numKernelRuns = mlir::decisionforest::numberOfKernelRuns;
    mlir::decisionforest::numberOfKernelRuns = NUM_RUNS;

    double bestKernelTime = std::numeric_limits<double>::max();
    // Caching rows seems generally better than not. So we will
    // first explore schedules with row caching and then disable
    // if needed if we want to enable shared reduction.
    for (auto numRowsPerTB : rowsPerTB) {
      for (auto numRowsPerThread : rowsPerThread) {
        if (numRowsPerThread >= numRowsPerTB)
          break;
        for (auto treeThreads : numTreeThreads) {
          auto tbSize = numRowsPerTB / numRowsPerThread * treeThreads;
          if (tbSize > MAX_TB_SIZE)
            continue;
          if (tbSize < MIN_TB_SIZE)
            continue;
          for (auto rep : {"gpu_array", "gpu_sparse", "gpu_reorg"}) {
            if (m_shouldCacheRows)
              runBenchmarks<true>(bestKernelTime, rep, numRowsPerTB,
                                  numRowsPerThread, treeThreads);
            else
              runBenchmarks<false>(bestKernelTime, rep, numRowsPerTB,
                                   numRowsPerThread, treeThreads);
          }
        }
      }
    }

    // Now check if we need to do a shared reduction.
    trySharedReduce(bestKernelTime);

    mlir::decisionforest::numberOfKernelRuns = numKernelRuns;

    std::cout << m_model << "\t" << m_batchSize << std::endl;
    std::cout << "Best kernel execution time: " << bestKernelTime << std::endl;
  }

  void printBestSchedule() {
    std::cout << "Best schedule for " << m_model << " is: " << std::endl;
    std::cout << "\tnumRowsPerTB: " << m_bestOptions.numRowsPerTB << std::endl;
    std::cout << "\tnumRowsPerThread: " << m_bestOptions.numRowsPerThread
              << std::endl;
    std::cout << "\tnumTreeThreads: " << m_bestOptions.numTreeThreads
              << std::endl;
    std::cout << "\tcacheRows: " << m_bestOptions.cacheRows << std::endl;
    std::cout << "\tcacheTrees: " << m_bestOptions.cacheTrees << std::endl;
    std::cout << "\tunrollTreeWalks: " << m_bestOptions.unrollTreeWalks
              << std::endl;
    std::cout << "\tinterleaveDepth: " << m_bestOptions.treeWalkInterleaveFactor
              << std::endl;
    std::cout << "\tsharedMemoryReduce: " << m_bestOptions.sharedMemoryReduce
              << std::endl;
    std::cout << "\tRepresentation: " << m_bestRep << std::endl;
    std::cout << "\tPadTrees: " << m_shouldPadTrees << std::endl;
  }
};

void findBestGPUSchedule(const std::string &benchmark, int32_t batchSize) {
  GPUAutoTuner tuner(benchmark, batchSize);
  tuner.exploreSchedules();
  tuner.printBestSchedule();
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
    findBestGPUSchedule("epsilon", batchSize);
}

} // namespace test
} // namespace TreeBeard

#endif // TREEBEARD_GPU_SUPPORT
