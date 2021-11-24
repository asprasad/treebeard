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

constexpr int32_t NUM_RUNS = 3;

template<typename FloatType>
int64_t Test_CodeGenForJSON_VariableBatchSize(int64_t batchSize, const std::string& modelJsonPath, int32_t tileSize) {
  mlir::MLIRContext context;

  TreeBeard::InitializeMLIRContext(context);
  auto module = TreeBeard::ConstructLLVMDialectModuleFromXGBoostJSON<FloatType, FloatType, int32_t, int32_t, FloatType>(context, modelJsonPath, batchSize, tileSize);
  decisionforest::InferenceRunner inferenceRunner(module, tileSize, sizeof(FloatType)*8, sizeof(int32_t)*8);
  
  TestCSVReader csvReader(modelJsonPath + ".csv");
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
  // char ch;
  // std::cout << "Attach profiler and press any key...";
  // std::cin >> ch;

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

  // std::cout << "Detach profiler and press any key...";
  // std::cin >> ch;

  return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
}

template<typename FPType, int32_t TileSize, int32_t BatchSize>
void RunSingleBenchmark_SingleConfig(const std::string& modelName) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/" + modelName + "_xgb_model_save.json";
  auto time = Test_CodeGenForJSON_VariableBatchSize<FPType>(BatchSize, modelJSONPath, TileSize);
  std::cout << "\t" + modelName << "\t" << time << std::endl;
}


template<typename FPType, int32_t TileSize, int32_t BatchSize>
void RunBenchmark_SingleConfig() {
  std::cout << "Type\t" << GetTypeName(FPType()) << "\tBatch\t" << BatchSize << " \tTile\t" << TileSize << std::endl;
  RunSingleBenchmark_SingleConfig<FPType, TileSize, BatchSize>("abalone");
  RunSingleBenchmark_SingleConfig<FPType, TileSize, BatchSize>("airline");
  RunSingleBenchmark_SingleConfig<FPType, TileSize, BatchSize>("airline-ohe");
  RunSingleBenchmark_SingleConfig<FPType, TileSize, BatchSize>("bosch");
  RunSingleBenchmark_SingleConfig<FPType, TileSize, BatchSize>("epsilon");
  RunSingleBenchmark_SingleConfig<FPType, TileSize, BatchSize>("higgs");
  RunSingleBenchmark_SingleConfig<FPType, TileSize, BatchSize>("year_prediction_msd");
}

void RunXGBoostBenchmarks() {
  {
    using FPType = double;
    RunBenchmark_SingleConfig<FPType, 1, 1>();
    RunBenchmark_SingleConfig<FPType, 1, 2>();
    RunBenchmark_SingleConfig<FPType, 1, 4>();

    RunBenchmark_SingleConfig<FPType, 2, 1>();
    RunBenchmark_SingleConfig<FPType, 2, 2>();
    RunBenchmark_SingleConfig<FPType, 2, 4>();

    RunBenchmark_SingleConfig<FPType, 3, 1>();
    RunBenchmark_SingleConfig<FPType, 3, 2>();
    RunBenchmark_SingleConfig<FPType, 3, 4>();

    RunBenchmark_SingleConfig<FPType, 4, 1>();
    RunBenchmark_SingleConfig<FPType, 4, 2>();
    RunBenchmark_SingleConfig<FPType, 4, 4>();
  }
  {
    using FPType = float;
    RunBenchmark_SingleConfig<FPType, 1, 1>();
    RunBenchmark_SingleConfig<FPType, 1, 2>();
    RunBenchmark_SingleConfig<FPType, 1, 4>();

    RunBenchmark_SingleConfig<FPType, 2, 1>();
    RunBenchmark_SingleConfig<FPType, 2, 2>();
    RunBenchmark_SingleConfig<FPType, 2, 4>();

    RunBenchmark_SingleConfig<FPType, 3, 1>();
    RunBenchmark_SingleConfig<FPType, 3, 2>();
    RunBenchmark_SingleConfig<FPType, 3, 4>();

    RunBenchmark_SingleConfig<FPType, 4, 1>();
    RunBenchmark_SingleConfig<FPType, 4, 2>();
    RunBenchmark_SingleConfig<FPType, 4, 4>();
    RunBenchmark_SingleConfig<FPType, 5, 4>();
    RunBenchmark_SingleConfig<FPType, 6, 4>();
    RunBenchmark_SingleConfig<FPType, 8, 4>();
  }
}


} // test
} // TreeBeard
