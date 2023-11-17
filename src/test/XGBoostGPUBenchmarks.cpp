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

using namespace mlir;

using mlir::decisionforest::GPUBasicSchedule;
using mlir::decisionforest::TahoeSharedDataStrategy;
using mlir::decisionforest::TahoeSharedDataStrategy_Modified;
using mlir::decisionforest::TahoeSharedForestStrategy;
using mlir::decisionforest::TahoeSharedPartialForestStrategy;

#define NUM_RUNS 100

namespace TreeBeard {
namespace test {

struct GPUTimes {
  double totalTimePerSample;
  double kernelTimePerSample;
};

template <typename ThresholdType, typename IndexType, typename ReturnType>
GPUTimes BenchmarkGPUCodeGeneration(TreeBeard::TreebeardContext &tbContext) {

  auto module = ConstructGPUModuleFromTreebeardContext(tbContext);
  auto batchSize = tbContext.options.batchSize;

  GPUInferenceRunnerForTest inferenceRunner(
      tbContext.serializer, module, tbContext.options.tileSize,
      sizeof(ThresholdType) * 8, sizeof(IndexType) * 8);
  TestCSVReader csvReader(tbContext.modelPath + ".test.sampled.csv",
                          2000 /*num lines*/);

  assert(csvReader.NumberOfRows() == 2000);
  std::vector<std::vector<ThresholdType>> inputData;
  for (size_t i = batchSize; i <= csvReader.NumberOfRows(); i += batchSize) {
    std::vector<ThresholdType> batch, preds;
    for (int32_t j = 0; j < batchSize; ++j) {
      auto rowIndex = (i - batchSize) + j;
      auto row = csvReader.GetRowOfType<ThresholdType>(rowIndex);
      row.pop_back();
      batch.insert(batch.end(), row.begin(), row.end());
    }
    inputData.push_back(batch);
  }

  std::vector<ReturnType> result(batchSize, -1);
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  for (int32_t trial = 0; trial < NUM_RUNS; ++trial) {
    for (auto &batch : inputData) {
      inferenceRunner.RunInference<ThresholdType, ReturnType>(batch.data(),
                                                              result.data());
    }
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  int64_t numSamples = NUM_RUNS * inputData.size() * batchSize;
  int64_t timeTaken =
      std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
          .count();
  auto totalTimePerSample = (double)timeTaken / (double)numSamples;
  auto kernelTimePerSample =
      (double)inferenceRunner.GetKernelExecutionTime() / (double)numSamples;
  return GPUTimes{totalTimePerSample, kernelTimePerSample};
}

template <typename ThresholdType, typename IndexType, typename ReturnType>
GPUTimes BenchmarkAutoScheduleCodeGeneration(
    const int32_t batchSize, ForestCreator &forestCreator,
    const std::string &modelPath, const std::string &modelGlobalsJSONPath,
    std::shared_ptr<decisionforest::IModelSerializer> serializer,
    std::shared_ptr<decisionforest::IRepresentation> representation,
    int32_t tileSize, int32_t tileShapeBitwidth, int32_t childIndexBitWidth,
    TreeBeard::GPUAutoScheduleOptions &gpuAutoScheduleOptions) {

  TreeBeard::CompilerOptions options(
      sizeof(FloatType) * 8, sizeof(FloatType) * 8, true, sizeof(IndexType) * 8,
      sizeof(IndexType) * 8, sizeof(FloatType) * 8, batchSize, tileSize,
      tileShapeBitwidth, childIndexBitWidth, TreeBeard::TilingType::kUniform,
      true, false, nullptr);

  // [HACK!] Create a shared pointer that points to forestCreator
  std::shared_ptr<ForestCreator> forestCreatorPtr(&forestCreator,
                                                  NoOpDeleter());

  TreeBeard::TreebeardContext tbContext(
      forestCreator.GetContext(), modelPath, modelGlobalsJSONPath, options,
      representation, serializer, forestCreatorPtr, &gpuAutoScheduleOptions);

  return BenchmarkGPUCodeGeneration<ThresholdType, IndexType, ReturnType>(
      tbContext);
}

template <typename ThresholdType, typename IndexType, typename ReturnType>
GPUTimes BenchmarkXGBoostAutoSchedule(
    const std::string &xgboostModelFile,
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

  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);
  TreeBeard::XGBoostJSONParser<ThresholdType, ReturnType, IndexType, IndexType,
                               ThresholdType>
      xgBoostParser(context, xgboostModelPath, serializer, batchSize);
  return BenchmarkAutoScheduleCodeGeneration<ThresholdType, IndexType,
                                             ReturnType>(
      batchSize, xgBoostParser, xgboostModelPath, modelGlobalsJSONPath,
      serializer, representation,
      1,                     // Tile size
      16,                    // Tile shape width
      sizeof(IndexType) * 8, // child index width
      gpuAutoScheduleOptions);
}

template <typename ThresholdType, typename ReturnType, bool cacheRows,
          bool cacheTrees, bool unrollTreeWalk>
GPUTimes BenchmarkXGBoostAutoSchedule(const std::string &xgboostModelFile,
                                      const std::string representationName,
                                      int32_t batchSize, int32_t numRowsPerTB,
                                      int32_t numRowsPerThread,
                                      int32_t numTreeThreads,
                                      int numTreesAtATime) {
  using IndexType = int16_t;

  TreeBeard::GPUAutoScheduleOptions gpuAutoScheduleOptions{
      numRowsPerTB,    numRowsPerThread, -1,         numTreeThreads,
      numTreesAtATime, cacheRows,        cacheTrees, unrollTreeWalk};

  return BenchmarkXGBoostAutoSchedule<ThresholdType, IndexType, ReturnType>(
      xgboostModelFile, gpuAutoScheduleOptions, representationName, batchSize);
}

template <typename ThresholdType, typename ReturnType>
void RunAutoScheduleBenchmarks(const std::string &xgboostModelFile,
                               const std::string representationName,
                               int32_t batchSize) {

  auto noUnrollTime = BenchmarkXGBoostAutoSchedule<ThresholdType, ReturnType,
                                                   false, false, false>(
      xgboostModelFile, representationName, batchSize, 64, 8, 2, 1);
  auto unrollTime = BenchmarkXGBoostAutoSchedule<ThresholdType, ReturnType,
                                                 false, false, true>(
      xgboostModelFile, representationName, batchSize, 64, 8, 2, 1);

  std::cout << "model,representation,batch_size,unroll,total,kernel"
            << std::endl;
  std::cout << xgboostModelFile << "," << representationName << "," << batchSize
            << ","
            << "false"
            << "," << noUnrollTime.totalTimePerSample << ","
            << noUnrollTime.kernelTimePerSample << std::endl;
  std::cout << xgboostModelFile << "," << representationName << "," << batchSize
            << ","
            << "true"
            << "," << unrollTime.totalTimePerSample << ","
            << unrollTime.kernelTimePerSample << std::endl;
}

void RunXGBoostGPUBenchmarks() {
  mlir::decisionforest::measureGpuKernelTime = true;

  RunAutoScheduleBenchmarks<float, float>("airline_xgb_model_save.json",
                                          "gpu_sparse", 256);
  RunAutoScheduleBenchmarks<float, float>("airline_xgb_model_save.json",
                                          "gpu_sparse", 256);
  RunAutoScheduleBenchmarks<float, float>("airline-ohe_xgb_model_save.json",
                                          "gpu_sparse", 256);
  RunAutoScheduleBenchmarks<float, float>("bosch_xgb_model_save.json",
                                          "gpu_sparse", 256);
  RunAutoScheduleBenchmarks<float, int8_t>("covtype_xgb_model_save.json",
                                           "gpu_sparse", 256);
  RunAutoScheduleBenchmarks<float, float>("epsilon_xgb_model_save.json",
                                          "gpu_sparse", 256);
  RunAutoScheduleBenchmarks<float, float>("higgs_xgb_model_save.json",
                                          "gpu_sparse", 256);
  RunAutoScheduleBenchmarks<float, int8_t>("letters_xgb_model_save.json",
                                           "gpu_sparse", 256);
  RunAutoScheduleBenchmarks<float, float>(
      "year_prediction_msd_xgb_model_save.json", "gpu_sparse", 256);
  mlir::decisionforest::measureGpuKernelTime = false;
}

} // namespace test
} // namespace TreeBeard

#endif // TREEBEARD_GPU_SUPPORT