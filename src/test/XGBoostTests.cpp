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

using namespace mlir;

namespace TreeBeard
{
namespace test
{

template<typename FloatType>
bool Test_CodeGenForJSON_VariableBatchSize(TestArgs_t& args, int64_t batchSize, const std::string& modelJsonPath, int32_t tileSize) {
  TestCSVReader csvReader(modelJsonPath + ".csv");
  TreeBeard::XGBoostJSONParser<FloatType, FloatType, int32_t, int32_t, FloatType> xgBoostParser(args.context, modelJsonPath, batchSize);
  xgBoostParser.Parse();
  auto module = xgBoostParser.GetEvaluationFunction();
  // module->dump();
  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(args.context, module);
  mlir::decisionforest::DoUniformTiling(args.context, module, tileSize);
  mlir::decisionforest::LowerEnsembleToMemrefs(args.context, module);
  mlir::decisionforest::ConvertNodeTypeToIndexType(args.context, module);
  // module->dump();
  mlir::decisionforest::LowerToLLVM(args.context, module);
  // module->dump();
  // mlir::decisionforest::dumpLLVMIR(module);
  decisionforest::InferenceRunner inferenceRunner(module, tileSize, sizeof(FloatType)*8, sizeof(int32_t)*8);
  
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
    std::vector<FloatType> result(batchSize, -1);
    inferenceRunner.RunInference<FloatType, FloatType>(batch.data(), result.data(), rowSize, batchSize);
    for(int64_t rowIdx=0 ; rowIdx<batchSize ; ++rowIdx) {
      // This needs to be a vector of doubles because the type is hardcoded for Forest::Predict
      std::vector<double> row(batch.begin() + rowIdx*rowSize, batch.begin() + (rowIdx+1)*rowSize);
      FloatType expectedResult = (*currentPredictionsIter)[rowIdx];
      FloatType forestPrediction = xgBoostParser.GetForest()->Predict(row);

      Test_ASSERT(FPEqual<FloatType>(forestPrediction, expectedResult));
      Test_ASSERT(FPEqual<FloatType>(result[rowIdx], expectedResult));
    }
    ++currentPredictionsIter;
  }
  return true;
}

template<typename FloatType>
bool Test_RandomXGBoostJSONs_VariableTrees_VariableBatchSize(TestArgs_t& args, int32_t batchSize, 
                                                             const std::string& modelDirRelativePath, int32_t tileSize) {
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
    Test_ASSERT(Test_CodeGenForJSON_VariableBatchSize<FloatType>(args, batchSize, modelJSONPath, tileSize));
  }
  return true;
}

template<typename FloatType=double>
bool Test_RandomXGBoostJSONs_1Tree_VariableBatchSize(TestArgs_t& args, int32_t batchSize, int32_t tileSize=1) {
  return Test_RandomXGBoostJSONs_VariableTrees_VariableBatchSize<FloatType>(args, batchSize, "xgb_models/test/Random_1Tree", tileSize);
}

template<typename FloatType=double>
bool Test_RandomXGBoostJSONs_2Trees_VariableBatchSize(TestArgs_t& args, int32_t batchSize, int32_t tileSize=1) {
  return Test_RandomXGBoostJSONs_VariableTrees_VariableBatchSize<FloatType>(args, batchSize, "xgb_models/test/Random_2Tree", tileSize);
}

template<typename FloatType=double>
bool Test_RandomXGBoostJSONs_4Trees_VariableBatchSize(TestArgs_t& args, int32_t batchSize, int32_t tileSize=1) {
  return Test_RandomXGBoostJSONs_VariableTrees_VariableBatchSize<FloatType>(args, batchSize, "xgb_models/test/Random_4Tree", tileSize);
}

bool Test_RandomXGBoostJSONs_1Tree_BatchSize1(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize(args, 1);
}

bool Test_RandomXGBoostJSONs_1Tree_BatchSize2(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize(args, 2);
}

bool Test_RandomXGBoostJSONs_1Tree_BatchSize4(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize(args, 4);
}

bool Test_RandomXGBoostJSONs_2Trees_BatchSize1(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize(args, 1);
}

bool Test_RandomXGBoostJSONs_2Trees_BatchSize2(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize(args, 2);
}

bool Test_RandomXGBoostJSONs_2Trees_BatchSize4(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize(args, 4);
}

bool Test_RandomXGBoostJSONs_4Trees_BatchSize1(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize(args, 1);
}

bool Test_RandomXGBoostJSONs_4Trees_BatchSize2(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize(args, 2);
}

bool Test_RandomXGBoostJSONs_4Trees_BatchSize4(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize(args, 4);
}

bool Test_RandomXGBoostJSONs_1Tree_BatchSize1_Float(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<float>(args, 1);
}

bool Test_RandomXGBoostJSONs_1Tree_BatchSize2_Float(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<float>(args, 2);
}

bool Test_RandomXGBoostJSONs_1Tree_BatchSize4_Float(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<float>(args, 4);
}

bool Test_RandomXGBoostJSONs_2Trees_BatchSize1_Float(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<float>(args, 1);
}

bool Test_RandomXGBoostJSONs_2Trees_BatchSize2_Float(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<float>(args, 2);
}

bool Test_RandomXGBoostJSONs_2Trees_BatchSize4_Float(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<float>(args, 4);
}

bool Test_RandomXGBoostJSONs_4Trees_BatchSize1_Float(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<float>(args, 1);
}

bool Test_RandomXGBoostJSONs_4Trees_BatchSize2_Float(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<float>(args, 2);
}

bool Test_RandomXGBoostJSONs_4Trees_BatchSize4_Float(TestArgs_t& args) {
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<float>(args, 4);
}

// --------------------------------------------------------------------------
// XGBoost Uniform Tiling Tests
// --------------------------------------------------------------------------

bool Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize1(TestArgs_t& args) {
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

bool Test_Scalar_Abalone(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  {
    using FPType = double;
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, 1);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, 1);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, 1);
  }
  {
    using FPType = float;
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, 1);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, 1);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, 1);
  }
  return true;
}

bool Test_TileSize2_Abalone(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  int32_t tileSize = 2;
  {
    using FPType = double;
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, tileSize);
  }
  {
    using FPType = float;
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, tileSize);
  }
  return true;
}

bool Test_TileSize3_Abalone(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  int32_t tileSize = 3;
  {
    using FPType = double;
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, tileSize);
  }
  {
    using FPType = float;
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, tileSize);
  }
  return true;
}

bool Test_TileSize4_Abalone(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  int32_t tileSize = 4;
  {
    using FPType = double;
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, tileSize);
  }
  {
    using FPType = float;
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, tileSize);
  }
  return true;
}

bool Test_Scalar_Airline(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  {
    using FPType = double;
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, 1);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, 1);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, 1);
  }
  {
    using FPType = float;
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, 1);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, 1);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, 1);
  }
  return true;
}

bool Test_TileSize2_Airline(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 2;
  {
    using FPType = double;
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, tileSize);
  }
  {
    using FPType = float;
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, tileSize);
  }
  return true;
}

bool Test_TileSize3_Airline(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 3;
  {
    using FPType = double;
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, tileSize);
  }
  {
    using FPType = float;
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, tileSize);
  }
  return true;
}

bool Test_TileSize4_Airline(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  int32_t tileSize = 4;
  {
    using FPType = double;
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, tileSize);
  }
  {
    using FPType = float;
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, tileSize);
    Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, tileSize);
  }
  return true;
}

} // test
} // TreeBeard