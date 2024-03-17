#ifndef _GPUTESTUTILS_H_
#define _GPUTESTUTILS_H_

#include "DecisionForest.h"
#include "ExecutionHelpers.h"
#include "schedule.h"
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>

#include "GPUExecutionHelper.h"
#include "GPUSupportUtils.h"
#include "GPUTestUtils.h"
#include "TestUtilsCommon.h"

namespace TreeBeard {
namespace test {

extern int32_t NUM_RUNS;
constexpr bool VERIFY_RESULT = false;

constexpr int32_t MIN_TB_SIZE = 32;
constexpr int32_t MAX_TB_SIZE = 1024;
constexpr int32_t MAX_SHMEM_SIZE = 49152;
constexpr bool PAD_TREES = true;
constexpr bool SH_MEM_REDUCE = false;

// ===---------------------------------------------------=== //
//  Helpers from XGBoostGPUBenchmarks.cpp
// ===---------------------------------------------------=== //

struct GPUBenchmarkExecState {
  std::shared_ptr<TreeBeard::TreebeardContext> tbContext;
  mlir::ModuleOp module;
};

struct GPUTimes {
  double totalTimePerSample;
  double kernelTimePerSample;
};

class TestDataReader {
  const bool useSameRowForAllInputs = false;
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

template <typename T>
void populateInputdata(int32_t batchSize,
                       std::vector<std::vector<T>> &inputData,
                       const std::string &csvFileName) {
  static TestDataReader dataReader;
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
    const std::string &representationName, int32_t batchSize,
    bool modelPathPassed) {
  auto xgboostModelPath = modelPathPassed
                              ? xgboostModelFile
                              : GetXGBoostModelPath(xgboostModelFile);
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
    int numTreesAtATime, int treeInterleaveDepth, bool modelPathPassed) {
  using IndexType = int16_t;

  TreeBeard::GPUAutoScheduleOptions gpuAutoScheduleOptions{
      numRowsPerTB,        numRowsPerThread,  -1,         numTreeThreads,
      numTreesAtATime,     cacheRows,         cacheTrees, unrollTreeWalk,
      treeInterleaveDepth, sharedMemoryReduce};

  return CompileXGBoostAutoScheduleBenchmark<ThresholdType, IndexType,
                                             ReturnType>(
      xgboostModelFile, context, gpuAutoScheduleOptions, representationName,
      batchSize, modelPathPassed);
}

template <typename ThresholdType, typename ReturnType, bool cacheRows,
          bool cacheTrees, bool unrollTreeWalk,
          bool sharedMemoryReduce = SH_MEM_REDUCE>
GPUTimes BenchmarkIfNoSharedMemOverflow(const std::string &modelName,
                                        const std::string &representationName,
                                        int32_t batchSize, int32_t numRowsPerTB,
                                        int32_t numRowsPerThread,
                                        int32_t numTreeThreads,
                                        int32_t treeInterleaveDepth,
                                        bool modelPathPassed = false) {
  using IndexType = int16_t;

  auto xgboostModelFile =
      modelPathPassed ? modelName : modelName + "_xgb_model_save.json";
  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);
  auto execInfo =
      CompileXGBoostAutoScheduleBenchmark<ThresholdType, ReturnType, cacheRows,
                                          cacheTrees, unrollTreeWalk,
                                          sharedMemoryReduce>(
          xgboostModelFile, context, representationName, batchSize,
          numRowsPerTB, numRowsPerThread, numTreeThreads, 1,
          treeInterleaveDepth, modelPathPassed);

  if (execInfo.tbContext->gpuCompileInfo.sharedMemoryInBytes <=
      MAX_SHMEM_SIZE) {

    auto times =
        BenchmarkGPUCodeGeneration<ThresholdType, IndexType, ReturnType>(
            execInfo.module, *execInfo.tbContext);

    std::cerr << modelName << "," << representationName << "," << batchSize
              << "," << numRowsPerTB << "," << numRowsPerThread << ","
              << numTreeThreads << "," << treeInterleaveDepth << std::boolalpha
              << "," << cacheRows << "," << cacheTrees << "," << unrollTreeWalk
              << "," << sharedMemoryReduce << "," << times.totalTimePerSample
              << "," << times.kernelTimePerSample << std::endl;
    return times;
  }
  return GPUTimes{-1, -1};
}

// ===---------------------------------------------------=== //
//  GPU Auto-tuning Heuristic Benchmark Helpers
// ===---------------------------------------------------=== //
template <typename ThresholdType, typename IndexType, typename ReturnType>
GPUBenchmarkExecState CompileAutoScheduleBenchmark(
    ForestCreator &forestCreator, const std::string &modelGlobalsJSONPath,
    MLIRContext &context,
    TreeBeard::GPUAutoScheduleOptions &gpuAutoScheduleOptions,
    const std::string &representationName, int32_t batchSize,
    const std::string &modelPath) {
  auto serializer = forestCreator.GetSerializer();
  auto representation =
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          representationName);

  return CompileAutoScheduleBenchmark<ThresholdType, IndexType, ReturnType>(
      batchSize, forestCreator, modelPath, modelGlobalsJSONPath, serializer,
      representation,
      1,                     // Tile size
      16,                    // Tile shape width
      sizeof(IndexType) * 8, // child index width
      gpuAutoScheduleOptions);
}

template <typename ThresholdType, typename ReturnType, bool cacheRows,
          bool cacheTrees, bool unrollTreeWalk,
          bool sharedMemoryReduce = SH_MEM_REDUCE>
GPUTimes BenchmarkIfNoSharedMemOverflow(ForestCreator &forestCreator,
                                        const std::string &representationName,
                                        int32_t batchSize, int32_t numRowsPerTB,
                                        int32_t numRowsPerThread,
                                        int32_t numTreeThreads,
                                        int32_t treeInterleaveDepth,
                                        const std::string &modelPath) {
  using IndexType = int16_t;

  TreeBeard::GPUAutoScheduleOptions gpuAutoScheduleOptions{
      numRowsPerTB,          numRowsPerThread,  -1,         numTreeThreads,
      1 /*numTreesAtATime*/, cacheRows,         cacheTrees, unrollTreeWalk,
      treeInterleaveDepth,   sharedMemoryReduce};

  auto &context = forestCreator.GetContext();
  TreeBeard::InitializeMLIRContext(context);
  auto execInfo =
      CompileAutoScheduleBenchmark<ThresholdType, IndexType, ReturnType>(
          forestCreator, forestCreator.GetModelGlobalsJSONFilePath(), context,
          gpuAutoScheduleOptions, representationName, batchSize, modelPath);

  if (execInfo.tbContext->gpuCompileInfo.sharedMemoryInBytes <=
      MAX_SHMEM_SIZE) {

    auto times =
        BenchmarkGPUCodeGeneration<ThresholdType, IndexType, ReturnType>(
            execInfo.module, *execInfo.tbContext);

    std::cerr << representationName << "," << batchSize << "," << numRowsPerTB
              << "," << numRowsPerThread << "," << numTreeThreads << ","
              << treeInterleaveDepth << std::boolalpha << "," << cacheRows
              << "," << cacheTrees << "," << unrollTreeWalk << ","
              << sharedMemoryReduce << "," << times.totalTimePerSample << ","
              << times.kernelTimePerSample << std::endl;
    return times;
  }
  return GPUTimes{-1, -1};
}

} // namespace test
} // namespace TreeBeard

#endif // _GPUTESTUTILS_H_