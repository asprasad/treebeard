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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"

#include "xgboostparser.h"
#include "ExecutionHelpers.h"
#include "TreeTilingDescriptor.h"
#include "TreeTilingUtils.h"
#include "ForestTestUtils.h"
#include "CompileUtils.h"
#include "TiledTree.h"
#include "ModelSerializers.h"
#include "Representations.h"

using namespace mlir;
using namespace mlir::decisionforest;

namespace TreeBeard
{
namespace test
{

// ===---------------------------------------------------=== //
// XGBoost Random Models Hybrid Tiling + Peeling Tests
// ===---------------------------------------------------=== //

struct SetAndResetPeelTreeWalk {
  SetAndResetPeelTreeWalk() {
    mlir::decisionforest::UseSparseTreeRepresentation = true;
    mlir::decisionforest::PeeledCodeGenForProbabiltyBasedTiling = true;
  }
  ~SetAndResetPeelTreeWalk() {
    mlir::decisionforest::UseSparseTreeRepresentation = false;
    mlir::decisionforest::PeeledCodeGenForProbabiltyBasedTiling = false;
  }
};

void InitializeRandomLeafHitCounts(decisionforest::DecisionForest<>& forest) {
  for (size_t i=0 ; i<forest.NumTrees() ; ++i) {
    auto& tree = forest.GetTree(i);
    auto& nodes = tree.GetNodes();
    int32_t numLeaves = 0;
    for (auto& node : nodes) {
      numLeaves += node.IsLeaf() ? 1 : 0;
    }
    int32_t hitLeaves = 0.05 * numLeaves;
    int32_t leafIndex = 0;
    for (auto& node : nodes) {
      if (!node.IsLeaf())
        continue;
      if (leafIndex < hitLeaves)
        const_cast<decisionforest::DecisionTree<>::Node&>(node).hitCount = 100;
      else
        const_cast<decisionforest::DecisionTree<>::Node&>(node).hitCount = 0;
      
      ++leafIndex;
    }
  }
}

void InitializeLeafHitCountsForUnifAndProb(decisionforest::DecisionForest<>& forest) {
  size_t numUnifTrees = forest.NumTrees()/2;

  // Make half the trees uniformly tiled (not skewed)
  for (size_t i=0 ; i<numUnifTrees ; ++i) {
    auto& tree = forest.GetTree(i);
    auto& nodes = tree.GetNodes();
    for (auto& node : nodes) {
      const_cast<decisionforest::DecisionTree<>::Node&>(node).hitCount = 1;
    }
  }

  // And the other half skewed so they are prob tiled
  for (size_t i=numUnifTrees ; i<forest.NumTrees() ; ++i) {
    auto& tree = forest.GetTree(i);
    auto& nodes = tree.GetNodes();
    int32_t numLeaves = 0;
    for (auto& node : nodes) {
      numLeaves += node.IsLeaf() ? 1 : 0;
    }
    int32_t hitLeaves = 0.05 * numLeaves;
    int32_t leafIndex = 0;
    for (auto& node : nodes) {
      if (!node.IsLeaf())
        continue;
      if (leafIndex < hitLeaves)
        const_cast<decisionforest::DecisionTree<>::Node&>(node).hitCount = 100;
      else
        const_cast<decisionforest::DecisionTree<>::Node&>(node).hitCount = 0;
      ++leafIndex;
    }
  }
}

template<typename FloatType, typename FeatureIndexType=int32_t, typename ResultType=FloatType>
bool Test_CodeGenForJSON_VariableBatchSize(TestArgs_t& args, int64_t batchSize, const std::string& modelJsonPath, const std::string& csvPath, 
                                           int32_t tileSize, int32_t tileShapeBitWidth, int32_t childIndexBitWidth, bool allTreesBiased) {
  TestCSVReader csvReader(csvPath);

  using NodeIndexType = int32_t;
  int32_t floatTypeBitWidth = sizeof(FloatType)*8;
  TreeBeard::CompilerOptions options(floatTypeBitWidth, sizeof(ResultType)*8, IsFloatType(ResultType()), sizeof(FeatureIndexType)*8, sizeof(NodeIndexType)*8,
                                     floatTypeBitWidth, batchSize, tileSize, tileShapeBitWidth, childIndexBitWidth,
                                     TreeBeard::TilingType::kHybrid, false /*makeAllLeavesSameDepth*/, true, nullptr);
  
  auto modelGlobalsJSONFilePath = TreeBeard::ModelJSONParser<FloatType, FloatType, int32_t, int32_t, FloatType>::ModelGlobalJSONFilePathFromJSONFilePath(modelJsonPath);
  auto serializer = decisionforest::ConstructModelSerializer(modelGlobalsJSONFilePath);
  // auto module = TreeBeard::ConstructLLVMDialectModuleFromXGBoostJSON<FloatType, ResultType, FeatureIndexType>(args.context, modelJsonPath, modelGlobalsJSONFilePath, options);
  TreeBeard::XGBoostJSONParser<FloatType, 
                               ResultType,
                               FeatureIndexType,
                               NodeIndexType,
                               FloatType> xgBoostParser(args.context, modelJsonPath, serializer, options.batchSize);
  xgBoostParser.Parse();
  xgBoostParser.SetChildIndexBitWidth(options.childIndexBitWidth);

  if (allTreesBiased) {
    // Initialize the hit counts of trees so that they are all biased
    InitializeRandomLeafHitCounts(*xgBoostParser.GetForest());
  }
  else {
    InitializeLeafHitCountsForUnifAndProb(*xgBoostParser.GetForest());
  }
  auto module = xgBoostParser.GetEvaluationFunction();

  mlir::decisionforest::DoHybridTiling(args.context, module, options.tileSize, options.tileShapeBitWidth);
  mlir::decisionforest::DoReorderTreesByDepth(args.context, module, -1);
  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(args.context, module);
  auto representation = decisionforest::ConstructRepresentation();
  mlir::decisionforest::LowerEnsembleToMemrefs(args.context, 
                                               module,
                                               serializer,
                                               representation);
  mlir::decisionforest::ConvertNodeTypeToIndexType(args.context, module);
  // module->dump();
  mlir::decisionforest::LowerToLLVM(args.context, module, representation);


  decisionforest::InferenceRunner inferenceRunner(modelGlobalsJSONFilePath, module, tileSize, sizeof(FloatType)*8, sizeof(FeatureIndexType)*8);
  
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
      ResultType expectedResult = (*currentPredictionsIter)[rowIdx];
      Test_ASSERT(FPEqual<ResultType>(result[rowIdx], expectedResult));
    }
    ++currentPredictionsIter;
  }
  return true;
}

template<typename FloatType>
bool Test_HybridTilingAndPeeling_RandomXGBoostJSONs(TestArgs_t& args, int32_t batchSize, 
                                                    const std::string& modelDirRelativePath, 
                                                    int32_t tileSize, int32_t tileShapeBitWidth, int32_t childIndexBitWidth,
                                                    bool allTreesBiased) {
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
                                                                 tileSize, tileShapeBitWidth, childIndexBitWidth, allTreesBiased));
  }
  return true;
}

template<typename FloatType=double>
bool Test_HybridTilingAndPeeling_RandomXGBoostJSONs_1Tree(TestArgs_t& args, int32_t batchSize, int32_t tileSize, int32_t tileShapeBitWidth,
                                                          int32_t childIndexBitWidth) {
  SetAndResetPeelTreeWalk setPeelingOn;
  return Test_HybridTilingAndPeeling_RandomXGBoostJSONs<FloatType>(args, batchSize, "xgb_models/test/Random_1Tree", 
                                                                   tileSize, tileShapeBitWidth, childIndexBitWidth, true);
}

template<typename FloatType=double>
bool Test_HybridTilingAndPeeling_RandomXGBoostJSONs_2Trees(TestArgs_t& args, int32_t batchSize, int32_t tileSize, int32_t tileShapeBitWidth,
                                                          int32_t childIndexBitWidth, bool allTreesBiased) {
  SetAndResetPeelTreeWalk setPeelingOn;
  return Test_HybridTilingAndPeeling_RandomXGBoostJSONs<FloatType>(args, batchSize, "xgb_models/test/Random_2Tree", 
                                                                   tileSize, tileShapeBitWidth, childIndexBitWidth, allTreesBiased);
}

template<typename FloatType=double>
bool Test_HybridTilingAndPeeling_RandomXGBoostJSONs_4Trees(TestArgs_t& args, int32_t batchSize, int32_t tileSize, int32_t tileShapeBitWidth,
                                                          int32_t childIndexBitWidth, bool allTreesBiased) {
  SetAndResetPeelTreeWalk setPeelingOn;
  return Test_HybridTilingAndPeeling_RandomXGBoostJSONs<FloatType>(args, batchSize, "xgb_models/test/Random_4Tree", 
                                                                   tileSize, tileShapeBitWidth, childIndexBitWidth, allTreesBiased);
}

bool Test_HybridTilingAndPeeling_RandomXGBoostJSONs_1Tree_FloatBatchSize4(TestArgs_t& args) {
  return Test_HybridTilingAndPeeling_RandomXGBoostJSONs_1Tree<float>(args, 4, 8, 16, 16);
}

bool Test_HybridTilingAndPeeling_RandomXGBoostJSONs_2Tree_FloatBatchSize4(TestArgs_t& args) {
  return Test_HybridTilingAndPeeling_RandomXGBoostJSONs_2Trees<float>(args, 4, 8, 16, 16, true);
}

bool Test_HybridTilingAndPeeling_RandomXGBoostJSONs_4Tree_FloatBatchSize4(TestArgs_t& args) {
  return Test_HybridTilingAndPeeling_RandomXGBoostJSONs_4Trees<float>(args, 4, 8, 16, 16, true);
}

bool Test_UniformAndHybridTilingAndPeeling_RandomXGBoostJSONs_2Tree_FloatBatchSize4(TestArgs_t& args) {
  return Test_HybridTilingAndPeeling_RandomXGBoostJSONs_2Trees<float>(args, 4, 8, 16, 16, false);
}

bool Test_UniformAndHybridTilingAndPeeling_RandomXGBoostJSONs_4Tree_FloatBatchSize4(TestArgs_t& args) {
  return Test_HybridTilingAndPeeling_RandomXGBoostJSONs_4Trees<float>(args, 4, 8, 16, 16, false);
}

// ===---------------------------------------------------=== //
// XGBoost Benchmarks Models Hybrid Tiling + Peeling Tests
// ===---------------------------------------------------=== //

template<typename FloatType, typename FeatureIndexType=int32_t, typename ResultType=FloatType>
bool TestXGBoostBenchmark_CodeGenForJSON_VariableBatchSize(TestArgs_t& args, int64_t batchSize, const std::string& modelJsonPath, const std::string& csvPath,
                                                           const std::string& statsProfileCSV,
                                                           int32_t tileSize, int32_t tileShapeBitWidth, int32_t childIndexBitWidth) {
  TestCSVReader csvReader(csvPath);

  using NodeIndexType = int32_t;
  int32_t floatTypeBitWidth = sizeof(FloatType)*8;
  TreeBeard::CompilerOptions options(floatTypeBitWidth, sizeof(ResultType)*8, IsFloatType(ResultType()), sizeof(FeatureIndexType)*8, sizeof(NodeIndexType)*8,
                                     floatTypeBitWidth, batchSize, tileSize, tileShapeBitWidth, childIndexBitWidth,
                                     TreeBeard::TilingType::kHybrid, false /*makeAllLeavesSameDepth*/, true, nullptr);
  
  auto modelGlobalsJSONFilePath = TreeBeard::ModelJSONParser<FloatType, FloatType, int32_t, int32_t, FloatType>::ModelGlobalJSONFilePathFromJSONFilePath(modelJsonPath);
  auto serializer = decisionforest::ConstructModelSerializer(modelGlobalsJSONFilePath);
  // auto module = TreeBeard::ConstructLLVMDialectModuleFromXGBoostJSON<FloatType, ResultType, FeatureIndexType>(args.context, modelJsonPath, modelGlobalsJSONFilePath, options);
  TreeBeard::XGBoostJSONParser<FloatType, 
                               ResultType,
                               FeatureIndexType,
                               NodeIndexType,
                               FloatType> xgBoostParser(args.context, modelJsonPath, serializer, statsProfileCSV, options.batchSize);
  xgBoostParser.Parse();
  xgBoostParser.SetChildIndexBitWidth(options.childIndexBitWidth);
  auto module = xgBoostParser.GetEvaluationFunction();

  mlir::decisionforest::DoHybridTiling(args.context, module, options.tileSize, options.tileShapeBitWidth);
  mlir::decisionforest::DoReorderTreesByDepth(args.context, module, -1);
  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(args.context, module);
  auto representation = decisionforest::ConstructRepresentation();
  mlir::decisionforest::LowerEnsembleToMemrefs(args.context,
                                               module,
                                               serializer,
                                               representation);
  mlir::decisionforest::ConvertNodeTypeToIndexType(args.context, module);
  // module->dump();
  mlir::decisionforest::LowerToLLVM(args.context, module, representation);

  decisionforest::InferenceRunner inferenceRunner(modelGlobalsJSONFilePath, module, tileSize, sizeof(FloatType)*8, sizeof(FeatureIndexType)*8);
  
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
      ResultType expectedResult = (*currentPredictionsIter)[rowIdx];
      Test_ASSERT(FPEqual<ResultType>(expectedResult, result[rowIdx]));

    }
    ++currentPredictionsIter;
  }
  return true;
}

bool Test_PeeledHybridProbabilisticTiling_TileSize8_Abalone(TestArgs_t &args) {
  SetAndResetPeelTreeWalk setPeelWalk;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  auto statsProfileCSV = testModelsDir + "/profiles/abalone.test.csv";
  int32_t tileSize = 8;
  return TestXGBoostBenchmark_CodeGenForJSON_VariableBatchSize<float, int32_t>(args, 1, modelJSONPath, csvPath, statsProfileCSV, tileSize, 16, 16);
}

bool Test_PeeledHybridProbabilisticTiling_TileSize8_Airline(TestArgs_t &args) {
  SetAndResetPeelTreeWalk setPeelWalk;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  auto statsProfileCSV = testModelsDir + "/profiles/airline.test.csv";
  int32_t tileSize = 8;
  return TestXGBoostBenchmark_CodeGenForJSON_VariableBatchSize<float, int16_t>(args, 1, modelJSONPath, csvPath, statsProfileCSV, tileSize, 16, 16);
}

bool Test_PeeledHybridProbabilisticTiling_TileSize8_AirlineOHE(TestArgs_t &args) {
  SetAndResetPeelTreeWalk setPeelWalk;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  auto statsProfileCSV = testModelsDir + "/profiles/airline-ohe.test.csv";
  int32_t tileSize = 8;
  return TestXGBoostBenchmark_CodeGenForJSON_VariableBatchSize<float, int16_t>(args, 1, modelJSONPath, csvPath, statsProfileCSV, tileSize, 16, 16);
}

bool Test_PeeledHybridProbabilisticTiling_TileSize8_Covtype(TestArgs_t &args) {
  SetAndResetPeelTreeWalk setPeelWalk;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/covtype_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  auto statsProfileCSV = testModelsDir + "/profiles/covtype.test.csv";
  int32_t tileSize = 8;
  return TestXGBoostBenchmark_CodeGenForJSON_VariableBatchSize<float, int16_t, int8_t>(args, 1, modelJSONPath, csvPath, statsProfileCSV, tileSize, 16, 16);
}

bool Test_PeeledHybridProbabilisticTiling_TileSize8_Letters(TestArgs_t &args) {
  SetAndResetPeelTreeWalk setPeelWalk;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/letters_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  auto statsProfileCSV = testModelsDir + "/profiles/letters.test.csv";
  int32_t tileSize = 8;
  return TestXGBoostBenchmark_CodeGenForJSON_VariableBatchSize<float, int16_t, int8_t>(args, 1, modelJSONPath, csvPath, statsProfileCSV, tileSize, 16, 16);
}

bool Test_PeeledHybridProbabilisticTiling_TileSize8_Epsilon(TestArgs_t &args) {
  SetAndResetPeelTreeWalk setPeelWalk;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  auto statsProfileCSV = testModelsDir + "/profiles/epsilon.test.csv";
  int32_t tileSize = 8;
  return TestXGBoostBenchmark_CodeGenForJSON_VariableBatchSize<float, int16_t>(args, 1, modelJSONPath, csvPath, statsProfileCSV, tileSize, 16, 16);
}

bool Test_PeeledHybridProbabilisticTiling_TileSize8_Higgs(TestArgs_t &args) {
  SetAndResetPeelTreeWalk setPeelWalk;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  auto statsProfileCSV = testModelsDir + "/profiles/higgs.test.csv";
  int32_t tileSize = 8;
  return TestXGBoostBenchmark_CodeGenForJSON_VariableBatchSize<float, int16_t>(args, 1, modelJSONPath, csvPath, statsProfileCSV, tileSize, 16, 16);
}

bool Test_PeeledHybridProbabilisticTiling_TileSize8_Year(TestArgs_t &args) {
  SetAndResetPeelTreeWalk setPeelWalk;
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  auto statsProfileCSV = testModelsDir + "/profiles/year_prediction_msd.test.csv";
  int32_t tileSize = 8;
  return TestXGBoostBenchmark_CodeGenForJSON_VariableBatchSize<float, int16_t>(args, 1, modelJSONPath, csvPath, statsProfileCSV, tileSize, 16, 16);
}

}
}