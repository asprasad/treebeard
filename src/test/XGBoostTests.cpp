#include <vector>
#include <sstream>
#include "Dialect.h"
#include "TestUtilsCommon.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "llvm/ADT/STLExtras.h"

#include "xgboostparser.h"
#include "ExecutionHelpers.h"
#include "TreeTilingDescriptor.h"
#include "TreeTilingUtils.h"
#include "ForestTestUtils.h"
#include "CompileUtils.h"

using namespace mlir;
using namespace mlir::decisionforest;

namespace TreeBeard
{
namespace test
{

bool RunSingleBatchSizeForXGBoostTests = true;

// ===---------------------------------------------------=== //
// XGBoost Scalar Inference Tests
// ===---------------------------------------------------=== //
template<typename FloatType, typename FeatureIndexType=int32_t, typename ResultType=FloatType>
bool Test_CodeGenForJSON_VariableBatchSize(TestArgs_t& args, int64_t batchSize, const std::string& modelJsonPath, const std::string& csvPath, 
                                           int32_t tileSize, int32_t tileShapeBitWidth, int32_t childIndexBitWidth,
                                           bool makeAllLeavesSameDepth, bool reorderTrees, ScheduleManipulator_t scheduleManipulatorFunc=nullptr,
                                           int32_t pipelineSize = -1) {
  TestCSVReader csvReader(csvPath);

  using NodeIndexType = int32_t;
  int32_t floatTypeBitWidth = sizeof(FloatType)*8;
  ScheduleManipulationFunctionWrapper scheduleManipulator(scheduleManipulatorFunc);
  TreeBeard::CompilerOptions options(floatTypeBitWidth, sizeof(ResultType)*8, IsFloatType(ResultType()), sizeof(FeatureIndexType)*8, sizeof(NodeIndexType)*8,
                                     floatTypeBitWidth, batchSize, tileSize, tileShapeBitWidth, childIndexBitWidth,
                                     TreeBeard::TilingType::kUniform, makeAllLeavesSameDepth, reorderTrees, 
                                     scheduleManipulatorFunc ? &scheduleManipulator : nullptr);

  options.SetPipelineSize(pipelineSize);
  auto modelGlobalsJSONFilePath = TreeBeard::ModelJSONParser<FloatType, FloatType, int32_t, int32_t, FloatType>::ModelGlobalJSONFilePathFromJSONFilePath(modelJsonPath);
  auto module = TreeBeard::ConstructLLVMDialectModuleFromXGBoostJSON<FloatType, ResultType, FeatureIndexType>(args.context, modelJsonPath, modelGlobalsJSONFilePath, options);

  decisionforest::InferenceRunner inferenceRunner(modelGlobalsJSONFilePath, module, tileSize, sizeof(FloatType)*8, sizeof(FeatureIndexType)*8);
  
  // inferenceRunner.PrintLengthsArray();
  // inferenceRunner.PrintOffsetsArray();
  std::vector<std::vector<FloatType>> inputData;
  std::vector<std::vector<FloatType>> xgBoostPredictions;
  for (size_t i=batchSize  ; i<csvReader.NumberOfRows()-1 ; i += batchSize) {
    std::vector<FloatType> batch, preds;
    for (int32_t j=0 ; j<batchSize ; ++j) {
      auto rowIndex = (i-batchSize) + j;
      auto row = csvReader.GetRowOfType<FloatType>(rowIndex);
      auto xgBoostPrediction = row.back();
      row.pop_back();
      preds.push_back(xgBoostPrediction);
      batch.insert(batch.end(), row.begin(), row.end());
    }
    inputData.push_back(batch);
    xgBoostPredictions.push_back(preds);
  }
  size_t rowSize = csvReader.GetRow(0).size() - 1; // The last entry is the xgboost prediction
  auto currentPredictionsIter = xgBoostPredictions.begin();
  for(auto& batch : inputData) {
    assert (batch.size() % batchSize == 0);
    std::vector<ResultType> result(batchSize, -1);
    inferenceRunner.RunInference<FloatType, ResultType>(batch.data(), result.data(), rowSize, batchSize);
    for(int64_t rowIdx=0 ; rowIdx<batchSize ; ++rowIdx) {
      // This needs to be a vector of doubles because the type is hardcoded for Forest::Predict
      // std::vector<double> row(batch.begin() + rowIdx*rowSize, batch.begin() + (rowIdx+1)*rowSize);
      ResultType expectedResult = (*currentPredictionsIter)[rowIdx];
      
      // FloatType forestPrediction = xgBoostParser.GetForest()->Predict_Float(row);
      // Test_ASSERT(FPEqual<ResultType>(forestPrediction, expectedResult));

      Test_ASSERT(FPEqual<ResultType>(result[rowIdx], expectedResult));
      // std::cout << forestPrediction << "\t" << result[rowIdx] << "\t" << expectedResult << std::endl;
    }
    ++currentPredictionsIter;
  }
  return true;
}

template<typename FloatType>
bool Test_RandomXGBoostJSONs_VariableTrees_VariableBatchSize(TestArgs_t& args, int32_t batchSize, 
                                                             const std::string& modelDirRelativePath, 
                                                             int32_t tileSize, int32_t tileShapeBitWidth, int32_t childIndexBitWidth,
                                                             bool makeAllTreesSameDepth, bool reorderTrees,
                                                             ScheduleManipulator_t scheduleManipulatorFunc = nullptr,
                                                             int32_t pipelineSize = -1) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/" + modelDirRelativePath;
  auto modelListFile = testModelsDir + "/ModelList.txt";
  std::ifstream fin(modelListFile);
  if (!fin)
    return false;
  while(!fin.eof()) {
    std::string modelJSONName;
    std::getline(fin, modelJSONName);
    auto modelJSONPath = testModelsDir + "/" + modelJSONName;
    // std::cout << "Model file : " << modelJSONPath << std::endl;
    Test_ASSERT(Test_CodeGenForJSON_VariableBatchSize<FloatType>(args, batchSize, modelJSONPath, modelJSONPath+".csv",
                                                                 tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllTreesSameDepth,
                                                                 reorderTrees, scheduleManipulatorFunc, pipelineSize));
  }
  return true;
}

template<typename FloatType=double>
bool Test_RandomXGBoostJSONs_1Tree_VariableBatchSize(TestArgs_t& args, int32_t batchSize, int32_t tileSize=1, int32_t tileShapeBitWidth=32,
                                                     int32_t childIndexBitWidth=1,
                                                     bool makeAllTreesSameDepth=false, bool reorderTrees=false,
                                                     ScheduleManipulator_t scheduleManipulatorFunc = nullptr,
                                                     int32_t pipelineSize = -1) {
  return Test_RandomXGBoostJSONs_VariableTrees_VariableBatchSize<FloatType>(args, batchSize, "xgb_models/test/Random_1Tree", 
                                                                            tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllTreesSameDepth,
                                                                            reorderTrees, scheduleManipulatorFunc, pipelineSize);
}

template<typename FloatType=double>
bool Test_RandomXGBoostJSONs_2Trees_VariableBatchSize(TestArgs_t& args, int32_t batchSize, int32_t tileSize=1, int32_t tileShapeBitWidth=32,
                                                      int32_t childIndexBitWidth=1, bool makeAllTreesSameDepth=false, bool reorderTrees=false) {
  return Test_RandomXGBoostJSONs_VariableTrees_VariableBatchSize<FloatType>(args, batchSize, "xgb_models/test/Random_2Tree",
                                                                            tileSize, tileShapeBitWidth, childIndexBitWidth, 
                                                                            makeAllTreesSameDepth, reorderTrees);
}

template<typename FloatType=double>
bool Test_RandomXGBoostJSONs_4Trees_VariableBatchSize(TestArgs_t& args, int32_t batchSize, int32_t tileSize=1, int32_t tileShapeBitWidth=32,
                                                      int32_t childIndexBitWidth=1, bool makeAllTreesSameDepth=false, bool reorderTrees=false,
                                                      ScheduleManipulator_t scheduleManipulatorFunc = nullptr,
                                                      int32_t pipelineSize = -1) {
  return Test_RandomXGBoostJSONs_VariableTrees_VariableBatchSize<FloatType>(args, batchSize, "xgb_models/test/Random_4Tree",
                                                                            tileSize, tileShapeBitWidth, childIndexBitWidth, 
                                                                            makeAllTreesSameDepth, reorderTrees, scheduleManipulatorFunc,
                                                                            pipelineSize);
}

bool Test_RandomXGBoostJSONs_1Tree_BatchSize1(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize(args, 1);
}

bool Test_RandomXGBoostJSONs_1Tree_BatchSize8_TileSize2_4Pipelined(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize(args, 8, 2, 32, 2, true, true, nullptr, 4);
}

bool Test_RandomXGBoostJSONs_1Tree_BatchSize8_TileSize4_4Pipelined(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize(args, 8, 4, 32, 3, true, true, nullptr, 4);
}

bool Test_RandomXGBoostJSONs_4Trees_BatchSize4_4Pipelined(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize(args, 8, 4, 32, 3, true, true, nullptr, 4);
}

bool Test_RandomXGBoostJSONs_1Tree_BatchSize2(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize(args, 2);
}

bool Test_RandomXGBoostJSONs_1Tree_BatchSize4(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize(args, 4);
}

bool Test_RandomXGBoostJSONs_2Trees_BatchSize1(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize(args, 1);
}

bool Test_RandomXGBoostJSONs_2Trees_BatchSize2(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize(args, 2);
}

bool Test_RandomXGBoostJSONs_2Trees_BatchSize4(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize(args, 4);
}

bool Test_RandomXGBoostJSONs_4Trees_BatchSize1(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize(args, 1);
}

bool Test_RandomXGBoostJSONs_4Trees_BatchSize2(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize(args, 2);
}

bool Test_RandomXGBoostJSONs_4Trees_BatchSize4(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize(args, 4);
}

bool Test_RandomXGBoostJSONs_1Tree_BatchSize1_Float(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<float>(args, 1);
}

bool Test_RandomXGBoostJSONs_1Tree_BatchSize2_Float(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<float>(args, 2);
}

bool Test_RandomXGBoostJSONs_1Tree_BatchSize4_Float(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<float>(args, 4);
}

bool Test_RandomXGBoostJSONs_2Trees_BatchSize1_Float(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;  
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<float>(args, 1);
}

bool Test_RandomXGBoostJSONs_2Trees_BatchSize2_Float(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;  
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<float>(args, 2);
}

bool Test_RandomXGBoostJSONs_2Trees_BatchSize4_Float(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<float>(args, 4);
}

bool Test_RandomXGBoostJSONs_4Trees_BatchSize1_Float(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;  
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<float>(args, 1);
}

bool Test_RandomXGBoostJSONs_4Trees_BatchSize2_Float(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;  
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<float>(args, 2);
}

bool Test_RandomXGBoostJSONs_4Trees_BatchSize4_Float(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<float>(args, 4);
}

// --------------------------------------------------------------------------
// XGBoost Sparse Code Gen Tests
// --------------------------------------------------------------------------

bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize1(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize(args, 1, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize2(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize(args, 2, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize4(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize(args, 4, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize1(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize(args, 1, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize2(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize(args, 2, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize4(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize(args, 4, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize1(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize(args, 1, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize2(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize(args, 2, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize4(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize(args, 4, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize1_Float(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<float>(args, 1, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize2_Float(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<float>(args, 2, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize4_Float(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<float>(args, 4, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize1_Float(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<float>(args, 1, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize2_Float(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<float>(args, 2, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize4_Float(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<float>(args, 4, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize1_Float(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<float>(args, 1, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize2_Float(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<float>(args, 2, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize4_Float(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<float>(args, 4, 1, 32, 32);
}

// --------------------------------------------------------------------------
// XGBoost Uniform Tiling Tests
// --------------------------------------------------------------------------

bool Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize1(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  {
    using FPType = double;
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 1, 2)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 1, 3)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 1, 4)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 1, 2)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 1, 3)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 1, 4)));
  }
  return true;
}

bool Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize1(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;  
  {
    using FPType = double;
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 1, 2)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 1, 3)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 1, 4)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 1, 2)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 1, 3)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 1, 4)));
  }
  return true;
}

bool Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize1(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  {
    using FPType = double;
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 1, 2)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 1, 3)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 1, 4)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 1, 2)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 1, 3)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 1, 4)));
  }
  return true;
}

bool Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize2(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  {
    using FPType = double;
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 2, 2)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 2, 3)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 2, 4)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 2, 2)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 2, 3)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 2, 4)));
  }
  return true;
}

bool Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize2(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  {
    using FPType = double;
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 2, 2)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 2, 3)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 2, 4)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 2, 2)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 2, 3)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 2, 4)));
  }
  return true;
}

bool Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize2(TestArgs_t& args) {
  if (RunSingleBatchSizeForXGBoostTests)
    return true;
  {
    using FPType = double;
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 2, 2)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 2, 3)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 2, 4)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 2, 2)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 2, 3)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 2, 4)));
  }
  return true;
}

bool Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4(TestArgs_t& args) {
  {
    using FPType = double;
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 2)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 3)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 4)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 2)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 3)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 4)));
  }
  return true;
}

bool Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4_Int8TileShape(TestArgs_t& args) {
  int32_t tileShapeBitWidth = 8;
  {
    using FPType = double;
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 2, tileShapeBitWidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 3, tileShapeBitWidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 4, tileShapeBitWidth)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 2, tileShapeBitWidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 3, tileShapeBitWidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 4, tileShapeBitWidth)));
  }
  return true;
}

bool Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4_Int16TileShape(TestArgs_t& args) {
  int32_t tileShapeBitWidth = 16;
  {
    using FPType = double;
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 2, tileShapeBitWidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 3, tileShapeBitWidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 4, tileShapeBitWidth)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 2, tileShapeBitWidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 3, tileShapeBitWidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 4, tileShapeBitWidth)));
  }
  return true;
}

bool Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4(TestArgs_t& args) {
  {
    using FPType = double;
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 2)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 3)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 4)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 2)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 3)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 4)));
  }
  return true;
}

bool Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4_Int8TileShape(TestArgs_t& args) {
  int32_t tileShapeBitwidth = 8;
  {
    using FPType = double;
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 2, tileShapeBitwidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 3, tileShapeBitwidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 4, tileShapeBitwidth)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 2, tileShapeBitwidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 3, tileShapeBitwidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 4, tileShapeBitwidth)));
  }
  return true;
}

bool Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4_Int16TileShape(TestArgs_t& args) {
  int32_t tileShapeBitwidth = 16;
  {
    using FPType = double;
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 2, tileShapeBitwidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 3, tileShapeBitwidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 4, tileShapeBitwidth)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 2, tileShapeBitwidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 3, tileShapeBitwidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 4, tileShapeBitwidth)));
  }
  return true;
}

bool Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4(TestArgs_t& args) {
  {
    using FPType = double;
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 2)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 3)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 4)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 2)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 3)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 4)));
  }
  return true;
}

bool Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4_Int8TileShape(TestArgs_t& args) {
  int32_t tileShapeBitWidth = 8;
  {
    using FPType = double;
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 2, tileShapeBitWidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 3, tileShapeBitWidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 4, tileShapeBitWidth)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 2, tileShapeBitWidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 3, tileShapeBitWidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 4, tileShapeBitWidth)));
  }
  return true;
}

bool Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4_Int16TileShape(TestArgs_t& args) {
  int32_t tileShapeBitWidth = 16;
  {
    using FPType = double;
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 2, tileShapeBitWidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 3, tileShapeBitWidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 4, tileShapeBitWidth)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 2, tileShapeBitWidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 3, tileShapeBitWidth)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 4, tileShapeBitWidth)));
  }
  return true;
}

// ===---------------------------------------------------=== //
// Random XGBoost Sparse Uniform Tiling Tests
// ===---------------------------------------------------=== //

bool Test_SparseUniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  {
    using FPType = double;
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 2, 32, 32)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 3, 32, 32)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 4, 32, 32)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 2, 32, 32)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 3, 32, 32)));
    Test_ASSERT((Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<FPType>(args, 4, 4, 32, 32)));
  }
  return true;
}

bool Test_SparseUniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  {
    using FPType = double;
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 2, 32, 32)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 3, 32, 32)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 4, 32, 32)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 2, 32, 32)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 3, 32, 32)));
    Test_ASSERT((Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<FPType>(args, 4, 4, 32, 32)));
  }
  return true;
}

bool Test_SparseUniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  {
    using FPType = double;
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 2, 32, 32)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 3, 32, 32)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 4, 32, 32)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 2, 32, 32)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 3, 32, 32)));
    Test_ASSERT((Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<FPType>(args, 4, 4, 32, 32)));
  }
  return true;
}

// ===---------------------------------------------------=== //
// XGBoost Benchmark Correctness Tests
// ===---------------------------------------------------=== //

bool Test_SingleTileSize_SingleModel(TestArgs_t &args, const std::string& modelJSONPath, int32_t tileSize, 
                                     bool skipInt8 = false, int32_t tileShapeBitWidth=32, int32_t childIndexBitWidth=1, std::string csvPath="",
                                     ScheduleManipulator_t scheduleManipulator=nullptr, bool makeAllLeavesSameDepth=false, bool reorderTrees=false,
                                     int32_t pipelineSize = -1) {
  if (csvPath == "")
    csvPath = modelJSONPath + ".csv";
  {
    using FPType = double;
    if (!RunSingleBatchSizeForXGBoostTests) {
      Test_ASSERT(Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize));
      Test_ASSERT(Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize));
    }
    Test_ASSERT(Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize));
  }
  {
    using FPType = double;
    using IntType = int16_t;
    if (!RunSingleBatchSizeForXGBoostTests) {
      Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 1, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
      Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 2, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
    }
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 4, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
  }
  if (!skipInt8)
  {
    using FPType = double;
    using IntType = int8_t;
    if (!RunSingleBatchSizeForXGBoostTests) {
      Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 1, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
      Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 2, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
    }
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 4, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
  }

  {
    using FPType = float;
    if (!RunSingleBatchSizeForXGBoostTests) {
      Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
      Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
    }
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
  }
  {
    using FPType = float;
    using IntType = int16_t;
    if (!RunSingleBatchSizeForXGBoostTests) {
      Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 1, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
      Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 2, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
    }
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 4, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
  }
  if (!skipInt8)
  {
    using FPType = float;
    using IntType = int8_t;
    if (!RunSingleBatchSizeForXGBoostTests) {
      Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 1, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
      Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 2, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
    }
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 4, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
  }
  return true;
}

bool Test_SingleTileSize_SingleModel_FloatOnly(TestArgs_t &args, const std::string& modelJSONPath, int32_t tileSize, 
                                     bool skipInt8 = false, int32_t tileShapeBitWidth=32, int32_t childIndexBitWidth=1, std::string csvPath="",
                                     ScheduleManipulator_t scheduleManipulator=nullptr, bool makeAllLeavesSameDepth=false, bool reorderTrees=false,
                                     int32_t pipelineSize = -1) {
  if (csvPath == "")
    csvPath = modelJSONPath + ".csv";
  {
    using FPType = float;
    if (!RunSingleBatchSizeForXGBoostTests) {
      Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
      Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
    }
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
  }
  {
    using FPType = float;
    using IntType = int16_t;
    if (!RunSingleBatchSizeForXGBoostTests) {
      Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 1, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
      Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 2, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
    }
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 4, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
  }
  if (!skipInt8)
  {
    using FPType = float;
    using IntType = int8_t;
    if (!RunSingleBatchSizeForXGBoostTests) {
      Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 1, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
      Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 2, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
    }
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 4, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, childIndexBitWidth, makeAllLeavesSameDepth, reorderTrees, scheduleManipulator, pipelineSize)));
  }
  return true;
}

bool Test_MultiClass_Int32ReturnType(TestArgs_t &args, const std::string& modelJSONPath, int32_t tileSize, 
                                     bool skipInt8 = false, int32_t tileShapeBitWidth=32, int32_t childIndexBitWidth=1, std::string csvPath="",
                                     ScheduleManipulator_t scheduleManipulator=nullptr, bool makeAllLeavesSameDepth=false, bool reorderTrees=false,
                                     int32_t pipelineSize=-1) {
  if (csvPath == "")
    csvPath = modelJSONPath + ".csv";
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<float, int32_t, int8_t>(
      args,
      8,
      modelJSONPath,
      csvPath,
      tileSize,
      tileShapeBitWidth,
      childIndexBitWidth,
      makeAllLeavesSameDepth,
      reorderTrees,
      scheduleManipulator,
      pipelineSize)));
  return true;
}

bool Test_Scalar_Abalone(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize);
}

bool Test_TileSize2_Abalone(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize);
}

bool Test_TileSize3_Abalone(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize);
}

bool Test_TileSize4_Abalone(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize);
}

bool Test_TileSize8_Abalone(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize);
}

bool Test_Scalar_Airline(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize);
}

bool Test_TileSize2_Airline(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize);
}

bool Test_TileSize3_Airline(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize);
}

bool Test_TileSize4_Airline(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize);
}

bool Test_TileSize8_Airline(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize);
}

bool Test_Scalar_AirlineOHE(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true);
}

bool Test_TileSize2_AirlineOHE(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true);
}

bool Test_TileSize3_AirlineOHE(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true);
}

bool Test_TileSize4_AirlineOHE(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true);
}

bool Test_TileSize8_AirlineOHE(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true);
}

bool Test_Scalar_Bosch(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true);
}

bool Test_TileSize2_Bosch(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true);
}

bool Test_TileSize3_Bosch(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true);
}

bool Test_TileSize4_Bosch(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true);
}

bool Test_TileSize8_Bosch(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true);
}

bool Test_Scalar_Epsilon(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true);
}

bool Test_TileSize2_Epsilon(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true);
}

bool Test_TileSize3_Epsilon(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true);
}

bool Test_TileSize4_Epsilon(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true);
}

bool Test_TileSize8_Epsilon(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true);
}

bool Test_Scalar_Higgs(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize);
}

bool Test_TileSize2_Higgs(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize);
}

bool Test_TileSize3_Higgs(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize);
}

bool Test_TileSize4_Higgs(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize);
}

bool Test_TileSize8_Higgs(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize);
}

bool Test_Scalar_Year(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize);
}

bool Test_TileSize2_Year(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize);
}

bool Test_TileSize3_Year(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize);
}

bool Test_TileSize4_Year(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize);
}

bool Test_TileSize8_Year(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize);
}

static bool Test_TileSizeVariable_CovType_Int8Type(TestArgs_t &args, int32_t tileSize) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/covtype_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".csv";
  return Test_MultiClass_Int32ReturnType(args, modelJSONPath, tileSize, false, 16, 16, csvPath);
}

bool Test_TileSize1_CovType_Int8Type(TestArgs_t &args) {
  return Test_TileSizeVariable_CovType_Int8Type(args, 1);
}

bool Test_TileSize2_CovType_Int8Type(TestArgs_t &args) {
  return Test_TileSizeVariable_CovType_Int8Type(args, 2);
}

bool Test_TileSize3_CovType_Int8Type(TestArgs_t &args) {
  return Test_TileSizeVariable_CovType_Int8Type(args, 3);
}

bool Test_TileSize4_CovType_Int8Type(TestArgs_t &args) {
  return Test_TileSizeVariable_CovType_Int8Type(args, 4);
}

bool Test_TileSize8_CovType_Int8Type(TestArgs_t &args) {
  return Test_TileSizeVariable_CovType_Int8Type(args, 8);
}

static bool Test_TileSizeVariable_Letters_Int8Type(TestArgs_t &args, int32_t tileSize) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/letters_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  return Test_MultiClass_Int32ReturnType(args, modelJSONPath, tileSize, false, 16, 16, csvPath);
}

static bool Test_TileSizeVariable_Letters_Pipelined_Int8Type(TestArgs_t &args, int32_t tileSize, int32_t pipelineSize) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/letters_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  return Test_MultiClass_Int32ReturnType(args, modelJSONPath, tileSize, false, 16, 16, csvPath, nullptr, true, true, pipelineSize);
}

bool Test_TileSize1_Letters_Int8Type(TestArgs_t &args) {
  return Test_TileSizeVariable_Letters_Int8Type(args, 1);
}

bool Test_TileSize2_Letters_Int8Type(TestArgs_t &args) {
  return Test_TileSizeVariable_Letters_Int8Type(args, 2);
}

bool Test_TileSize3_Letters_Int8Type(TestArgs_t &args) {
  return Test_TileSizeVariable_Letters_Int8Type(args, 3);
}

bool Test_TileSize4_Letters_Int8Type(TestArgs_t &args) {
  return Test_TileSizeVariable_Letters_Int8Type(args, 4);
}

bool Test_TileSize8_Letters_Int8Type(TestArgs_t &args) {
  return Test_TileSizeVariable_Letters_Int8Type(args, 8);
}

bool Test_TileSize3_Letters_2Pipelined_Int8Type(TestArgs_t &args) {
  return Test_TileSizeVariable_Letters_Pipelined_Int8Type(args, 3, 2);
}

bool Test_TileSize4_Letters_3Pipelined_Int8Type(TestArgs_t &args) {
  return Test_TileSizeVariable_Letters_Pipelined_Int8Type(args, 4, 3);
}

bool Test_TileSize8_Letters_5Pipelined_Int8Type(TestArgs_t &args) {
  return Test_TileSizeVariable_Letters_Pipelined_Int8Type(args, 8, 5);
}

// ===----------------------------------------------------------------=== //
// XGBoost Benchmark Code Gen Correctness Tests With XGBoost Schedule
// ===----------------------------------------------------------------=== //

bool Test_Scalar_Abalone_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize2_Abalone_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize3_Abalone_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize4_Abalone_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize8_Abalone_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_Scalar_Airline_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize2_Airline_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize3_Airline_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize4_Airline_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize8_Airline_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_Scalar_AirlineOHE_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize2_AirlineOHE_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize3_AirlineOHE_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize4_AirlineOHE_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize8_AirlineOHE_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_Scalar_Bosch_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize2_Bosch_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize3_Bosch_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize4_Bosch_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize8_Bosch_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_Scalar_Epsilon_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize2_Epsilon_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize3_Epsilon_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize4_Epsilon_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize8_Epsilon_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_Scalar_Higgs_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize2_Higgs_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize3_Higgs_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize4_Higgs_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize8_Higgs_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_Scalar_Year_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize2_Year_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize3_Year_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize4_Year_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_TileSize8_Year_OneTreeAtATimeSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 1, "", OneTreeAtATimeSchedule);
}

bool Test_Scalar_CovType_OneTreeAtATimeSchedule(TestArgs_t &args) {
  int32_t tileSize = 1;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/covtype_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".csv";
  return Test_MultiClass_Int32ReturnType(args, modelJSONPath, tileSize, false, 16, 16, csvPath, OneTreeAtATimeSchedule);
}

bool Test_TileSize8_CovType_OneTreeAtATimeSchedule(TestArgs_t &args) {
  int32_t tileSize = 8;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/covtype_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".csv";
  return Test_MultiClass_Int32ReturnType(args, modelJSONPath, tileSize, false, 16, 16, csvPath, OneTreeAtATimeSchedule);
}

// ===---------------------------------------------------=== //
// XGBoost Benchmark Sparse Code Gen Correctness Tests
// ===---------------------------------------------------=== //

bool Test_SparseScalar_Abalone(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32);
}

bool Test_SparseTileSize2_Abalone(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32);
}

bool Test_SparseTileSize3_Abalone(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32);
}

bool Test_SparseTileSize4_Abalone(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32);
}

bool Test_SparseTileSize8_Abalone(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32);
}

bool Test_SparseScalar_Airline(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32);
}

bool Test_SparseTileSize2_Airline(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32);
}

bool Test_SparseTileSize3_Airline(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32);
}

bool Test_SparseTileSize4_Airline(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32);
}

bool Test_SparseTileSize8_Airline(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32);
}

bool Test_SparseTileSize8_Pipeline4_Airline(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32, "", nullptr, true, true, 4);
}

bool Test_SparseScalar_AirlineOHE(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 32);
}

bool Test_SparseTileSize2_AirlineOHE(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
    auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 32);
}

bool Test_SparseTileSize3_AirlineOHE(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
    auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 32);
}

bool Test_SparseTileSize4_AirlineOHE(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
    auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 32);
}

bool Test_SparseTileSize8_AirlineOHE(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 32);
}

bool Test_SparseTileSize8_Pipelined4_AirlineOHE(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 32, "", nullptr, true, true, 4);
}

bool Test_SparseScalar_Bosch(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 32);
}

bool Test_SparseTileSize2_Bosch(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 32);
}

bool Test_SparseTileSize3_Bosch(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 32);
}

bool Test_SparseTileSize4_Bosch(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 32);
}

bool Test_SparseTileSize8_Bosch(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 32);
}

bool Test_SparseTileSize8_4Pipelined_Bosch(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 32, "" , nullptr, true, true, 4);
}

bool Test_SparseScalar_CovType_Int8Type(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_TileSizeVariable_CovType_Int8Type(args, 1);
}

bool Test_SparseTileSize8_CovType_Int8Type(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_TileSizeVariable_CovType_Int8Type(args, 8);
}

bool Test_SparseScalar_Epsilon(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 32);
}

bool Test_SparseTileSize2_Epsilon(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 32);
}

bool Test_SparseTileSize3_Epsilon(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 32);
}

bool Test_SparseTileSize4_Epsilon(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 32);
}

bool Test_SparseTileSize8_Epsilon(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 32);
}

bool Test_SparseTileSize8_Pipelined_Epsilon(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 32, 32, "", nullptr, true, true, 4);
}

bool Test_SparseScalar_Higgs(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32);
}

bool Test_SparseTileSize2_Higgs(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32);
}

bool Test_SparseTileSize3_Higgs(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32);
}

bool Test_SparseTileSize4_Higgs(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32);
}

bool Test_SparseTileSize8_Higgs(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32);
}

bool Test_SparseTileSize8_Pipelined_Higgs(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32, "", nullptr, true, true, 4);
}

bool Test_SparseScalar_Letters_Int8Type(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_TileSizeVariable_Letters_Int8Type(args, 1);
}

bool Test_SparseTileSize8_Letters_Int8Type(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_TileSizeVariable_Letters_Int8Type(args, 8);
}

bool Test_SparseScalar_Year(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32);
}

bool Test_SparseTileSize2_Year(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  int32_t tileSize = 2;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32);
}

bool Test_SparseTileSize3_Year(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  int32_t tileSize = 3;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32);
}

bool Test_SparseTileSize4_Year(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  int32_t tileSize = 4;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32);
}

bool Test_SparseTileSize8_Year(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32);
}

bool Test_SparseTileSize8_Pipelined_Year(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 32, 32, "", nullptr, true, true, 4);
}

// ===---------------------------------------------------=== //
// XGBoost Test Inputs Code Gen Correctness Tests
// ===---------------------------------------------------=== //

bool Test_TileSize8_Abalone_TestInputs(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath);
}

bool Test_TileSize8_Abalone_4Pipelined_TestInputs(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, nullptr, true, true, 4);
}

bool Test_TileSize8_AirlineOHE_TestInputs(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath);
}

bool Test_TileSize8_Airline_TestInputs(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath);
}

bool Test_TileSize8_Bosch_TestInputs(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath);
}

bool Test_TileSize8_Epsilon_TestInputs(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath);
}

bool Test_TileSize8_Higgs_TestInputs(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath);
}

bool Test_TileSize8_Year_TestInputs(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath);
}

bool Test_TileSize8_CovType_TestInputs(TestArgs_t &args) {
  int32_t tileSize = 8;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/covtype_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  return Test_MultiClass_Int32ReturnType(args, modelJSONPath, tileSize, false, 16, 16, csvPath);
}

bool Test_TileSize8_CovType_4Pipelined_TestInputs(TestArgs_t &args) {
  int32_t tileSize = 8;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/covtype_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  return Test_MultiClass_Int32ReturnType(args, modelJSONPath, tileSize, false, 16, 16, csvPath, nullptr, true, true, 4);
}

// ===--------------------------------------------------------=== //
// XGBoost Test Inputs Tiled Schedule Code Gen Correctness Tests
// ===--------------------------------------------------------=== //

bool Test_TileSize8_Abalone_TestInputs_TiledSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, TiledSchedule<2, 4>);
}

bool Test_TileSize8_AirlineOHE_TestInputs_TiledSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath, TiledSchedule<2, 4>);
}

bool Test_TileSize8_Airline_TestInputs_TiledSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, TiledSchedule<2, 4>);
}

bool Test_TileSize8_Bosch_TestInputs_TiledSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath, TiledSchedule<2, 4>);
}

bool Test_TileSize8_Epsilon_TestInputs_TiledSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath, TiledSchedule<2, 4>);
}

bool Test_TileSize8_Higgs_TestInputs_TiledSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, TiledSchedule<2, 4>);
}

bool Test_TileSize8_Year_TestInputs_TiledSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, TiledSchedule<2, 4>);
}

// ===----------------------------------------------------------------=== //
// XGBoost Benchmark Sparse Code Gen Correctness Tests With XGBoost Schedule
// ===----------------------------------------------------------------=== //

bool Test_SparseScalar_Abalone_OneTreeAtATimeSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 16, 16, "", OneTreeAtATimeSchedule);
}

bool Test_SparseTileSize8_Abalone_OneTreeAtATimeSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 16, 16, "", OneTreeAtATimeSchedule);
}

bool Test_SparseScalar_Airline_OneTreeAtATimeSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 16, 16, "", OneTreeAtATimeSchedule);
}

bool Test_SparseTileSize8_Airline_OneTreeAtATimeSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 16, 16, "", OneTreeAtATimeSchedule);
}

bool Test_SparseScalar_AirlineOHE_OneTreeAtATimeSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 16, 16, "", OneTreeAtATimeSchedule);
}

bool Test_SparseTileSize8_AirlineOHE_OneTreeAtATimeSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 16, 16, "", OneTreeAtATimeSchedule);
}

bool Test_SparseScalar_Bosch_OneTreeAtATimeSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 16, 16, "", OneTreeAtATimeSchedule);
}

bool Test_SparseTileSize8_Bosch_OneTreeAtATimeSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 16, 16, "", OneTreeAtATimeSchedule);
}

bool Test_SparseScalar_CovType_OneTreeAtATimeSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  int32_t tileSize = 1;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/covtype_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".csv";
  return Test_MultiClass_Int32ReturnType(args, modelJSONPath, tileSize, false, 16, 16, csvPath, OneTreeAtATimeSchedule);
}

bool Test_SparseTileSize8_CovType_OneTreeAtATimeSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  int32_t tileSize = 8;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/covtype_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".csv";
  return Test_MultiClass_Int32ReturnType(args, modelJSONPath, tileSize, false, 16, 16, csvPath, OneTreeAtATimeSchedule);
}

bool Test_SparseScalar_Letters_OneTreeAtATimeSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  int32_t tileSize = 1;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/letters_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  return Test_MultiClass_Int32ReturnType(args, modelJSONPath, tileSize, false, 16, 16, csvPath, OneTreeAtATimeSchedule);
}

bool Test_SparseTileSize8_Letters_OneTreeAtATimeSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  int32_t tileSize = 8;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/letters_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  return Test_MultiClass_Int32ReturnType(args, modelJSONPath, tileSize, false, 16, 16, csvPath, OneTreeAtATimeSchedule);
}

bool Test_SparseScalar_Epsilon_OneTreeAtATimeSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 16, 16, "", OneTreeAtATimeSchedule);
}

bool Test_SparseTileSize8_Epsilon_OneTreeAtATimeSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, true, 16, 16, "", OneTreeAtATimeSchedule);
}

bool Test_SparseScalar_Higgs_OneTreeAtATimeSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 16, 16, "", OneTreeAtATimeSchedule);
}

bool Test_SparseTileSize8_Higgs_OneTreeAtATimeSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 16, 16, "", OneTreeAtATimeSchedule);
}

bool Test_SparseScalar_Year_OneTreeAtATimeSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  int32_t tileSize = 1;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 16, 16, "", OneTreeAtATimeSchedule);
}

bool Test_SparseTileSize8_Year_OneTreeAtATimeSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel(args, modelJSONPath, tileSize, false, 16, 16, "", OneTreeAtATimeSchedule);
}

// ===-------------------------------------------------------------=== //
// XGBoost Test Inputs Tiled Schedule Sparse Code Gen Correctness Tests
// ===-------------------------------------------------------------=== //

bool Test_SparseTileSize8_Abalone_TestInputs_TiledSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, TiledSchedule<2, 4>);
}

bool Test_SparseTileSize8_AirlineOHE_TestInputs_TiledSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath, TiledSchedule<2, 4>);
}

bool Test_SparseTileSize8_Airline_TestInputs_TiledSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, TiledSchedule<2, 4>);
}

bool Test_SparseTileSize8_Bosch_TestInputs_TiledSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath, TiledSchedule<2, 4>);
}

bool Test_SparseTileSize8_Epsilon_TestInputs_TiledSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath, TiledSchedule<2, 4>);
}

bool Test_SparseTileSize8_Higgs_TestInputs_TiledSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, TiledSchedule<2, 4>);
}

bool Test_SparseTileSize8_Letters_TiledSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  int32_t tileSize = 8;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/letters_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  return Test_MultiClass_Int32ReturnType(args, modelJSONPath, tileSize, false, 16, 16, csvPath, TiledSchedule<2, 4>);
}

bool Test_SparseTileSize8_Year_TestInputs_TiledSchedule(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, TiledSchedule<2, 4>);
}

// ===-------------------------------------------------------------=== //
// XGBoost Test Inputs Probability Based Tiling Correctness Tests
// ===-------------------------------------------------------------=== //

template<typename FloatType, typename FeatureIndexType=int32_t, typename ResultType=FloatType>
bool Test_CodeGenForJSON_VariableBatchSize(TestArgs_t& args, int64_t batchSize, const std::string& modelJsonPath, const std::string& csvPath,
                                           const std::string& statsProfileCSV,
                                           int32_t tileSize, int32_t tileShapeBitWidth, int32_t childIndexBitWidth,
                                           ScheduleManipulator_t scheduleManipulatorFunc=nullptr) {
  TestCSVReader csvReader(csvPath);

  using NodeIndexType = int32_t;
  int32_t floatTypeBitWidth = sizeof(FloatType)*8;
  ScheduleManipulationFunctionWrapper scheduleManipulator(scheduleManipulatorFunc);
  TreeBeard::CompilerOptions options(floatTypeBitWidth, sizeof(ResultType)*8, IsFloatType(ResultType()), sizeof(FeatureIndexType)*8, sizeof(NodeIndexType)*8,
                                     floatTypeBitWidth, batchSize, tileSize, tileShapeBitWidth, childIndexBitWidth,
                                     TreeBeard::TilingType::kHybrid, false, false, scheduleManipulatorFunc ? &scheduleManipulator : nullptr);

  auto modelGlobalsJSONFilePath = TreeBeard::ModelJSONParser<FloatType, FloatType, int32_t, int32_t, FloatType>::ModelGlobalJSONFilePathFromJSONFilePath(modelJsonPath);
  auto module = TreeBeard::ConstructLLVMDialectModuleFromXGBoostJSON<FloatType, ResultType, FeatureIndexType, int32_t, FloatType>(
                                                                     args.context, modelJsonPath, modelGlobalsJSONFilePath, options);

  decisionforest::InferenceRunner inferenceRunner(modelGlobalsJSONFilePath, module, tileSize, sizeof(FloatType)*8, sizeof(FeatureIndexType)*8);
  
  // inferenceRunner.PrintLengthsArray();
  // inferenceRunner.PrintOffsetsArray();
  std::vector<std::vector<FloatType>> inputData;
  std::vector<std::vector<FloatType>> xgBoostPredictions;
  for (size_t i=batchSize  ; i<csvReader.NumberOfRows()-1 ; i += batchSize) {
    std::vector<FloatType> batch, preds;
    for (int32_t j=0 ; j<batchSize ; ++j) {
      auto rowIndex = (i-batchSize) + j;
      auto row = csvReader.GetRowOfType<FloatType>(rowIndex);
      auto xgBoostPrediction = row.back();
      row.pop_back();
      preds.push_back(xgBoostPrediction);
      batch.insert(batch.end(), row.begin(), row.end());
    }
    inputData.push_back(batch);
    xgBoostPredictions.push_back(preds);
  }
  size_t rowSize = csvReader.GetRow(0).size() - 1; // The last entry is the xgboost prediction
  auto currentPredictionsIter = xgBoostPredictions.begin();
  for(auto& batch : inputData) {
    assert (batch.size() % batchSize == 0);
    std::vector<ResultType> result(batchSize, -1);
    inferenceRunner.RunInference<FloatType, ResultType>(batch.data(), result.data(), rowSize, batchSize);
    for(int64_t rowIdx=0 ; rowIdx<batchSize ; ++rowIdx) {
      // This needs to be a vector of doubles because the type is hardcoded for Forest::Predict
      // std::vector<double> row(batch.begin() + rowIdx*rowSize, batch.begin() + (rowIdx+1)*rowSize);
      ResultType expectedResult = (*currentPredictionsIter)[rowIdx];
      
      // FloatType forestPrediction = xgBoostParser.GetForest()->Predict_Float(row);
      // Test_ASSERT(FPEqual<ResultType>(forestPrediction, expectedResult));

      Test_ASSERT(FPEqual<ResultType>(result[rowIdx], expectedResult));
      // std::cout << forestPrediction << "\t" << result[rowIdx] << "\t" << expectedResult << std::endl;
    }
    ++currentPredictionsIter;
  }
  return true;
}

bool Test_SingleTileSize_SingleModel_FloatOnly(TestArgs_t &args, const std::string& modelJSONPath, const std::string& statsProfileCSV, int32_t tileSize, 
                                     bool skipInt8 = false, int32_t tileShapeBitWidth=32, int32_t childIndexBitWidth=1, std::string csvPath="",
                                     ScheduleManipulator_t scheduleManipulator=nullptr) {
  if (csvPath == "")
    csvPath = modelJSONPath + ".csv";
  {
    using FPType = float;
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, csvPath, statsProfileCSV, tileSize, tileShapeBitWidth, childIndexBitWidth, scheduleManipulator)));
  }
  {
    using FPType = float;
    using IntType = int16_t;
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 4, modelJSONPath, csvPath, statsProfileCSV, tileSize, tileShapeBitWidth, childIndexBitWidth, scheduleManipulator)));
  }
  if (!skipInt8)
  {
    using FPType = float;
    using IntType = int8_t;
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 4, modelJSONPath, csvPath, statsProfileCSV, tileSize, tileShapeBitWidth, childIndexBitWidth, scheduleManipulator)));
  }
  return true;
}

bool Test_SingleTileSize_SingleModel_FloatOnly_Int8ReturnType(TestArgs_t &args, const std::string& modelJSONPath, const std::string& statsProfileCSV, int32_t tileSize, 
                                     bool skipInt8 = false, int32_t tileShapeBitWidth=32, int32_t childIndexBitWidth=1, std::string csvPath="",
                                     ScheduleManipulator_t scheduleManipulator=nullptr) {
  if (csvPath == "")
    csvPath = modelJSONPath + ".csv";
  {
    using FPType = float;
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, int32_t, int8_t>(args, 4, modelJSONPath, csvPath, statsProfileCSV, tileSize, tileShapeBitWidth, childIndexBitWidth, scheduleManipulator)));
  }
  {
    using FPType = float;
    using IntType = int16_t;
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType, int8_t>(args, 4, modelJSONPath, csvPath, statsProfileCSV, tileSize, tileShapeBitWidth, childIndexBitWidth, scheduleManipulator)));
  }
  if (!skipInt8)
  {
    using FPType = float;
    using IntType = int8_t;
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType, int8_t>(args, 4, modelJSONPath, csvPath, statsProfileCSV, tileSize, tileShapeBitWidth, childIndexBitWidth, scheduleManipulator)));
  }
  return true;
}

bool Test_ProbabilisticTiling_TileSize8_Abalone(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  auto statsProfileCSV = testModelsDir + "/profiles/abalone.test.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, statsProfileCSV, tileSize, false, 16, 16);
}

bool Test_SparseProbabilisticTiling_TileSize8_Abalone(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation=true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  auto statsProfileCSV = testModelsDir + "/profiles/abalone.test.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, statsProfileCSV, tileSize, false, 16, 16);
}

bool Test_ProbabilisticTiling_TileSize8_Airline(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  auto statsProfileCSV = testModelsDir + "/profiles/airline.test.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, statsProfileCSV, tileSize, false, 16, 16);
}

bool Test_SparseProbabilisticTiling_TileSize8_Airline(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation=true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  auto statsProfileCSV = testModelsDir + "/profiles/airline.test.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, statsProfileCSV, tileSize, false, 16, 16);
}

bool Test_ProbabilisticTiling_TileSize8_AirlineOHE(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  auto statsProfileCSV = testModelsDir + "/profiles/airline-ohe.test.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, statsProfileCSV, tileSize, true, 16, 16);
}

bool Test_SparseProbabilisticTiling_TileSize8_AirlineOHE(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation=true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  auto statsProfileCSV = testModelsDir + "/profiles/airline-ohe.test.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, statsProfileCSV, tileSize, true, 16, 16);
}

bool Test_ProbabilisticTiling_TileSize8_Covtype(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/covtype_xgb_model_save.json";
  auto statsProfileCSV = testModelsDir + "/profiles/covtype.test.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly_Int8ReturnType(args, modelJSONPath, statsProfileCSV, tileSize, false, 16, 16);
}

bool Test_SparseProbabilisticTiling_TileSize8_Covtype(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation=true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/covtype_xgb_model_save.json";
  auto statsProfileCSV = testModelsDir + "/profiles/covtype.test.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly_Int8ReturnType(args, modelJSONPath, statsProfileCSV, tileSize, false, 16, 16);
}

bool Test_ProbabilisticTiling_TileSize8_Epsilon(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  auto statsProfileCSV = testModelsDir + "/profiles/epsilon.test.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, statsProfileCSV, tileSize, true, 16, 16);
}

bool Test_SparseProbabilisticTiling_TileSize8_Epsilon(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation=true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  auto statsProfileCSV = testModelsDir + "/profiles/epsilon.test.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, statsProfileCSV, tileSize, true, 16, 16);
}

bool Test_ProbabilisticTiling_TileSize8_Higgs(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  auto statsProfileCSV = testModelsDir + "/profiles/higgs.test.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, statsProfileCSV, tileSize, false, 16, 16);
}

bool Test_SparseProbabilisticTiling_TileSize8_Higgs(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation=true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  auto statsProfileCSV = testModelsDir + "/profiles/higgs.test.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, statsProfileCSV, tileSize, false, 16, 16);
}

bool Test_ProbabilisticTiling_TileSize8_Year(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  auto statsProfileCSV = testModelsDir + "/profiles/year_prediction_msd.test.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, statsProfileCSV, tileSize, false, 16, 16);
}

bool Test_SparseProbabilisticTiling_TileSize8_Year(TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation=true;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  auto statsProfileCSV = testModelsDir + "/profiles/year_prediction_msd.test.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, statsProfileCSV, tileSize, false, 16, 16);
}

// ===-------------------------------------------------------------=== //
// Random XGBoost Tiled Tree Padding Tests 
// ===-------------------------------------------------------------=== //

bool Test_RandomXGBoostJSONs_1Tree_BatchSize4_EqualDepth_TileSize8(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize(args, 4, 8, 16, 1, true);
}

bool Test_RandomXGBoostJSONs_2Trees_BatchSize4_EqualDepth_TileSize8(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize(args, 4, 8, 16, 1, true);
}

bool Test_RandomXGBoostJSONs_4Trees_BatchSize4_EqualDepth_TileSize8(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize(args, 4, 8, 16, 1, true);
}

bool Test_TileSize8_Abalone_TestInputs_MakeLeavesSameDepth(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, nullptr, true);
}

bool Test_TileSize8_AirlineOHE_TestInputs_MakeLeavesSameDepth(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath, nullptr, true);
}

bool Test_TileSize8_Airline_TestInputs_MakeLeavesSameDepth(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, nullptr, true);
}

bool Test_TileSize8_Bosch_TestInputs_MakeLeavesSameDepth(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath, nullptr, true);
}

bool Test_TileSize8_Epsilon_TestInputs_MakeLeavesSameDepth(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath, nullptr, true);
}

bool Test_TileSize8_Higgs_TestInputs_MakeLeavesSameDepth(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, nullptr, true);
}

bool Test_TileSize8_Year_TestInputs_MakeLeavesSameDepth(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, nullptr, true);
}

bool Test_TileSize8_CovType_TestInputs_MakeLeavesSameDepth(TestArgs_t &args) {
  int32_t tileSize = 8;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/covtype_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  return Test_MultiClass_Int32ReturnType(args, modelJSONPath, tileSize, false, 16, 16, csvPath, nullptr, true);
}

// ===-------------------------------------------------------------=== //
// Random XGBoost Remove Extra Hop Tests
// ===-------------------------------------------------------------=== //

struct SetAndResetRemoveExtraHop {
  SetAndResetRemoveExtraHop() {
    mlir::decisionforest::UseSparseTreeRepresentation = true;
    mlir::decisionforest::RemoveExtraHopInSparseRepresentation = true;
  }
  ~SetAndResetRemoveExtraHop() {
    mlir::decisionforest::UseSparseTreeRepresentation = false;
    mlir::decisionforest::RemoveExtraHopInSparseRepresentation = false;
  }
};

bool Test_RandomXGBoostJSONs_1Tree_BatchSize4_RemoveExtraHop_TileSize8(TestArgs_t& args) {
  SetAndResetRemoveExtraHop setAndReset;
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize(args, 4, 8, 16, 16, false);
}

bool Test_RandomXGBoostJSONs_2Trees_BatchSize4_RemoveExtraHop_TileSize8(TestArgs_t& args) {
  SetAndResetRemoveExtraHop setAndReset;
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize(args, 4, 8, 16, 16, false);
}

bool Test_RandomXGBoostJSONs_4Trees_BatchSize4_RemoveExtraHop_TileSize8(TestArgs_t& args) {
  SetAndResetRemoveExtraHop setAndReset;
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize(args, 4, 8, 16, 16, false);
}

bool Test_TileSize8_Abalone_TestInputs_RemoveExtraHop(TestArgs_t &args) {
  SetAndResetRemoveExtraHop setAndReset;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath);
}

bool Test_TileSize8_AirlineOHE_TestInputs_RemoveExtraHop(TestArgs_t &args) {
  SetAndResetRemoveExtraHop setAndReset;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath);
}

bool Test_TileSize8_Airline_TestInputs_RemoveExtraHop(TestArgs_t &args) {
  SetAndResetRemoveExtraHop setAndReset;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath);
}

bool Test_TileSize8_Bosch_TestInputs_RemoveExtraHop(TestArgs_t &args) {
  SetAndResetRemoveExtraHop setAndReset;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath);
}

bool Test_TileSize8_Epsilon_TestInputs_RemoveExtraHop(TestArgs_t &args) {
  SetAndResetRemoveExtraHop setAndReset;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath);
}

bool Test_TileSize8_Higgs_TestInputs_RemoveExtraHop(TestArgs_t &args) {
  SetAndResetRemoveExtraHop setAndReset;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath);
}

bool Test_TileSize8_Year_TestInputs_RemoveExtraHop(TestArgs_t &args) {
  SetAndResetRemoveExtraHop setAndReset;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath);
}

bool Test_TileSize8_CovType_TestInputs_RemoveExtraHop(TestArgs_t &args) {
  SetAndResetRemoveExtraHop setAndReset;
  int32_t tileSize = 8;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/covtype_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  return Test_MultiClass_Int32ReturnType(args, modelJSONPath, tileSize, false, 16, 16, csvPath);
}

bool Test_TileSize8_Abalone_TestInputs_ReorderTrees(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, nullptr, true, true);
}

// ===--------------------------------------------------------=== //
// XGBoost Test Inputs Split Schedule Code Gen Correctness Tests
// ===--------------------------------------------------------=== //

bool Test_TileSize8_Abalone_TestInputs_SplitTreeLoopSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, SplitTreeDimensionSchedule<500>);
}

bool Test_TileSize8_AirlineOHE_TestInputs_SplitTreeLoopSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath, SplitTreeDimensionSchedule<500>);
}

bool Test_TileSize8_Airline_TestInputs_SplitTreeLoopSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, SplitTreeDimensionSchedule<50>);
}

bool Test_TileSize8_Bosch_TestInputs_SplitTreeLoopSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath, SplitTreeDimensionSchedule<50>);
}

bool Test_TileSize8_Epsilon_TestInputs_SplitTreeLoopSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath, SplitTreeDimensionSchedule<50>);
}

bool Test_TileSize8_Higgs_TestInputs_SplitTreeLoopSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, SplitTreeDimensionSchedule<50>);
}

bool Test_TileSize8_Year_TestInputs_SplitTreeLoopSchedule(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, SplitTreeDimensionSchedule<50>);
}

// ===--------------------------------------------------------=== //
// XGBoost Test Inputs Swap and Split Schedule Code Gen Correctness Tests
// ===--------------------------------------------------------=== //

bool Test_TileSize8_Abalone_TestInputs_SwapAndSplitTreeIndex(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, SwapAndSplitTreeDimensionSchedule<500>);
}

bool Test_TileSize8_AirlineOHE_TestInputs_SwapAndSplitTreeIndex(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath, SwapAndSplitTreeDimensionSchedule<500>);
}

bool Test_TileSize8_Airline_TestInputs_SwapAndSplitTreeIndex(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, SwapAndSplitTreeDimensionSchedule<50>);
}

bool Test_TileSize8_Bosch_TestInputs_SwapAndSplitTreeIndex(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/bosch_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath, SwapAndSplitTreeDimensionSchedule<50>);
}

bool Test_TileSize8_Epsilon_TestInputs_SwapAndSplitTreeIndex(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, true, 16, 16, csvPath, SwapAndSplitTreeDimensionSchedule<50>);
}

bool Test_TileSize8_Higgs_TestInputs_SwapAndSplitTreeIndex(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, SwapAndSplitTreeDimensionSchedule<50>);
}

bool Test_TileSize8_Year_TestInputs_SwapAndSplitTreeIndex(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8;
  return Test_SingleTileSize_SingleModel_FloatOnly(args, modelJSONPath, tileSize, false, 16, 16, csvPath, SwapAndSplitTreeDimensionSchedule<50>);
}

// ===--------------------------------------------------------=== //
// XGBoost Test Inputs Parallel Batch Schedule Code Gen Correctness Tests
// ===--------------------------------------------------------=== //

bool Test_TileSize8_Abalone_TestInputs_ParallelBatch(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8, tileShapeBitWidth=16, childIndexBitWidth=1;
  Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<float, int8_t>(args, 200, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, 
                                                                    childIndexBitWidth, false, false, [](decisionforest::Schedule* schedule) {
    schedule->Parallel(schedule->GetBatchIndex());
  })));
  return true;
}

bool Test_TileSize8_Airline_TestInputs_ParallelBatch(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8, tileShapeBitWidth=16, childIndexBitWidth=1;
  Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<float, int16_t>(args, 200, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, 
                                                                    childIndexBitWidth, false, false, [](decisionforest::Schedule* schedule) {
    schedule->Parallel(schedule->GetBatchIndex());
  })));
  return true;
}

bool Test_TileSize8_AirlineOHE_TestInputs_ParallelBatch(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8, tileShapeBitWidth=16, childIndexBitWidth=1;
  Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<float, int16_t>(args, 200, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, 
                                                                    childIndexBitWidth, false, false, [](decisionforest::Schedule* schedule) {
    schedule->Parallel(schedule->GetBatchIndex());
  })));
  return true;
}

bool Test_TileSize8_Covtype_TestInputs_ParallelBatch(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/covtype_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8, tileShapeBitWidth=16, childIndexBitWidth=1;
  Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<float, int16_t, int8_t>(args, 200, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, 
                                                                    childIndexBitWidth, false, false, [](decisionforest::Schedule* schedule) {
    schedule->Parallel(schedule->GetBatchIndex());
  })));
  return true;
}

bool Test_TileSize8_Letters_TestInputs_ParallelBatch(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/letters_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8, tileShapeBitWidth=16, childIndexBitWidth=1;
  Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<float, int16_t, int8_t>(args, 200, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, 
                                                                    childIndexBitWidth, false, false, [](decisionforest::Schedule* schedule) {
    schedule->Parallel(schedule->GetBatchIndex());
  })));
  return true;
}

bool Test_TileSize8_Epsilon_TestInputs_ParallelBatch(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8, tileShapeBitWidth=16, childIndexBitWidth=1;
  Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<float, int16_t>(args, 200, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, 
                                                                    childIndexBitWidth, false, false, [](decisionforest::Schedule* schedule) {
    schedule->Parallel(schedule->GetBatchIndex());
  })));
  return true;
}

bool Test_TileSize8_Higgs_TestInputs_ParallelBatch(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8, tileShapeBitWidth=16, childIndexBitWidth=1;
  Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<float, int16_t>(args, 200, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, 
                                                                    childIndexBitWidth, false, false, [](decisionforest::Schedule* schedule) {
    schedule->Parallel(schedule->GetBatchIndex());
  })));
  return true;
}

bool Test_TileSize8_Year_TestInputs_ParallelBatch(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  int32_t tileSize = 8, tileShapeBitWidth=16, childIndexBitWidth=1;
  Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<float, int16_t>(args, 200, modelJSONPath, csvPath, tileSize, tileShapeBitWidth, 
                                                                    childIndexBitWidth, false, false, [](decisionforest::Schedule* schedule) {
    schedule->Parallel(schedule->GetBatchIndex());
  })));
  return true;
}

} // test
} // TreeBeard