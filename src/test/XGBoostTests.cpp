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

// ===---------------------------------------------------=== //
// XGBoost Scalar Inference Tests
// ===---------------------------------------------------=== //

template<typename FloatType, typename FeatureIndexType=int32_t>
bool Test_CodeGenForJSON_VariableBatchSize(TestArgs_t& args, int64_t batchSize, const std::string& modelJsonPath, 
                                           int32_t tileSize, int32_t tileShapeBitWidth, int32_t childIndexBitWidth) {
  TestCSVReader csvReader(modelJsonPath + ".csv");
  TreeBeard::XGBoostJSONParser<FloatType, FloatType, FeatureIndexType, int32_t, FloatType> xgBoostParser(args.context, modelJsonPath, batchSize);
  xgBoostParser.Parse();
  xgBoostParser.SetChildIndexBitWidth(childIndexBitWidth);
  auto module = xgBoostParser.GetEvaluationFunction();
  // module->dump();
  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(args.context, module);
  mlir::decisionforest::DoUniformTiling(args.context, module, tileSize, tileShapeBitWidth);
  mlir::decisionforest::LowerEnsembleToMemrefs(args.context, module);
  mlir::decisionforest::ConvertNodeTypeToIndexType(args.context, module);
  // module->dump();
  mlir::decisionforest::LowerToLLVM(args.context, module);
  // module->dump();
  // mlir::decisionforest::dumpLLVMIR(module);
  decisionforest::InferenceRunner inferenceRunner(module, tileSize, sizeof(FloatType)*8, sizeof(FeatureIndexType)*8);
  
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
                                                             const std::string& modelDirRelativePath, 
                                                             int32_t tileSize, int32_t tileShapeBitWidth, int32_t childIndexBitWidth) {
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
    Test_ASSERT(Test_CodeGenForJSON_VariableBatchSize<FloatType>(args, batchSize, modelJSONPath, tileSize, tileShapeBitWidth, childIndexBitWidth));
  }
  return true;
}

template<typename FloatType=double>
bool Test_RandomXGBoostJSONs_1Tree_VariableBatchSize(TestArgs_t& args, int32_t batchSize, int32_t tileSize=1, int32_t tileShapeBitWidth=32, int32_t childIndexBitWidth=1) {
  return Test_RandomXGBoostJSONs_VariableTrees_VariableBatchSize<FloatType>(args, batchSize, "xgb_models/test/Random_1Tree", tileSize, tileShapeBitWidth, childIndexBitWidth);
}

template<typename FloatType=double>
bool Test_RandomXGBoostJSONs_2Trees_VariableBatchSize(TestArgs_t& args, int32_t batchSize, int32_t tileSize=1, int32_t tileShapeBitWidth=32, int32_t childIndexBitWidth=1) {
  return Test_RandomXGBoostJSONs_VariableTrees_VariableBatchSize<FloatType>(args, batchSize, "xgb_models/test/Random_2Tree", tileSize, tileShapeBitWidth, childIndexBitWidth);
}

template<typename FloatType=double>
bool Test_RandomXGBoostJSONs_4Trees_VariableBatchSize(TestArgs_t& args, int32_t batchSize, int32_t tileSize=1, int32_t tileShapeBitWidth=32, int32_t childIndexBitWidth=1) {
  return Test_RandomXGBoostJSONs_VariableTrees_VariableBatchSize<FloatType>(args, batchSize, "xgb_models/test/Random_4Tree", tileSize, tileShapeBitWidth, childIndexBitWidth);
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
// XGBoost Sparse Code Gen Tests
// --------------------------------------------------------------------------

bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize1(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize(args, 1, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize2(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize(args, 2, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize4(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize(args, 4, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize1(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize(args, 1, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize2(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize(args, 2, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize4(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize(args, 4, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize1(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize(args, 1, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize2(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize(args, 2, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize4(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize(args, 4, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize1_Float(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<float>(args, 1, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize2_Float(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<float>(args, 2, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize4_Float(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_1Tree_VariableBatchSize<float>(args, 4, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize1_Float(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<float>(args, 1, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize2_Float(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<float>(args, 2, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize4_Float(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_2Trees_VariableBatchSize<float>(args, 4, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize1_Float(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_RandomXGBoostJSONs_4Trees_VariableBatchSize<float>(args, 1, 1, 32, 32);
}

bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize2_Float(TestArgs_t& args) {
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
// XGBoost Benchmark Correctness Tests
// ===---------------------------------------------------=== //

bool Test_SingleTileSize_SingleModel(TestArgs_t &args, const std::string& modelJSONPath, int32_t tileSize, 
                                     bool skipInt8 = false, int32_t tileShapeBitWidth=32, int32_t childIndexBitWidth=1) {
  {
    using FPType = double;
    Test_ASSERT(Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, tileSize, tileShapeBitWidth, childIndexBitWidth));
    Test_ASSERT(Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, tileSize, tileShapeBitWidth, childIndexBitWidth));
    Test_ASSERT(Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, tileSize, tileShapeBitWidth, childIndexBitWidth));
  }
  {
    using FPType = double;
    using IntType = int16_t;
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 1, modelJSONPath, tileSize, tileShapeBitWidth, childIndexBitWidth)));
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 2, modelJSONPath, tileSize, tileShapeBitWidth, childIndexBitWidth)));
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 4, modelJSONPath, tileSize, tileShapeBitWidth, childIndexBitWidth)));
  }
  if (!skipInt8)
  {
    using FPType = double;
    using IntType = int8_t;
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 1, modelJSONPath, tileSize, tileShapeBitWidth, childIndexBitWidth)));
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 2, modelJSONPath, tileSize, tileShapeBitWidth, childIndexBitWidth)));
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 4, modelJSONPath, tileSize, tileShapeBitWidth, childIndexBitWidth)));
  }

  {
    using FPType = float;
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 1, modelJSONPath, tileSize, tileShapeBitWidth, childIndexBitWidth)));
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 2, modelJSONPath, tileSize, tileShapeBitWidth, childIndexBitWidth)));
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType>(args, 4, modelJSONPath, tileSize, tileShapeBitWidth, childIndexBitWidth)));
  }
  {
    using FPType = float;
    using IntType = int16_t;
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 1, modelJSONPath, tileSize, tileShapeBitWidth, childIndexBitWidth)));
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 2, modelJSONPath, tileSize, tileShapeBitWidth, childIndexBitWidth)));
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 4, modelJSONPath, tileSize, tileShapeBitWidth, childIndexBitWidth)));
  }
  if (!skipInt8)
  {
    using FPType = float;
    using IntType = int8_t;
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 1, modelJSONPath, tileSize, tileShapeBitWidth, childIndexBitWidth)));
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 2, modelJSONPath, tileSize, tileShapeBitWidth, childIndexBitWidth)));
    Test_ASSERT((Test_CodeGenForJSON_VariableBatchSize<FPType, IntType>(args, 4, modelJSONPath, tileSize, tileShapeBitWidth, childIndexBitWidth)));
  }
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

} // test
} // TreeBeard