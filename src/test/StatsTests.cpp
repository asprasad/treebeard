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
#include "StatsUtils.h"
#include "ModelSerializers.h"

using namespace mlir;

namespace TreeBeard
{
namespace test
{

decisionforest::DecisionForest ConstructForestAndRunInference(const std::string& modelJSONPath, const std::string& csvPath, int32_t numRows) {
  // std::string csvPath = "/home/ashwin/ML/scikit-learn_bench/xgb_models/airline_xgb_model_save.json.test.csv";
  mlir::MLIRContext context;
  TreeBeard::XGBoostJSONParser<> xgBoostParser(context, modelJSONPath, decisionforest::ConstructModelSerializer(""), 1);
  xgBoostParser.ConstructForest();
  auto decisionForest = xgBoostParser.GetForest();

  TreeBeard::test::TestCSVReader csvReader(csvPath);
  // std::cerr << "Done reading csv..\n";
  // std::cerr << "Running inference\n";
  for (size_t i=0  ; i<csvReader.NumberOfRows() ; ++i ) {
    auto row = csvReader.GetRowOfType<double>(i);
    if(row.size() <= 1)
      continue;
    row.pop_back();
    decisionForest->Predict(row);
  }
  return *decisionForest;
}

bool Test_XGBoostModel_StatGenerationAndReading(const std::string& modelJSON, const std::string& inputCSV, const std::string& statsCSV) {
  TreeBeard::Profile::ComputeForestProbabilityProfile(modelJSON, inputCSV, statsCSV, -1);

  mlir::MLIRContext context;
  TreeBeard::XGBoostJSONParser<> xgBoostParser(context, modelJSON, decisionforest::ConstructModelSerializer(""), 1);
  xgBoostParser.ConstructForest();
  auto decisionForest = xgBoostParser.GetForest();

  TreeBeard::Profile::ReadProbabilityProfile(*decisionForest, statsCSV);

  auto computedForest = ConstructForestAndRunInference(modelJSON, inputCSV, -1);
  for (size_t i=0 ; i<decisionForest->NumTrees() ; ++i) {
    auto& nodes1 = decisionForest->GetTree(i).GetNodes();
    auto& nodes2 = computedForest.GetTree(i).GetNodes();
    Test_ASSERT(nodes1.size() == nodes2.size());
    for (size_t j=0 ; j<nodes1.size() ; ++j) {
      if (abs(nodes1.at(j).hitCount - nodes2.at(j).hitCount)>1) {
        std::cerr << "Tree : " << i << " Node : " << j << " " << nodes1.at(j).hitCount << " " << nodes2.at(j).hitCount << std::endl;
        Test_ASSERT(false);
      }
      Test_ASSERT(nodes1.at(j).depth == nodes2.at(j).depth || nodes2.at(j).depth==-1 || nodes1.at(j).depth==-1);
    }
  }
  return true;
}

bool Test_AbaloneStatGenerationAndReading(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/abalone_xgb_model_save.json";
  auto csvPath = modelJSONPath  + ".test.sampled.csv";
  auto statsCSVPath = modelJSONPath  + ".test.sampled.stats.csv";
  return Test_XGBoostModel_StatGenerationAndReading(modelJSONPath, csvPath, statsCSVPath);
}

bool Test_AirlineStatGenerationAndReading(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline_xgb_model_save.json";
  auto csvPath = modelJSONPath  + ".test.sampled.csv";
  auto statsCSVPath = modelJSONPath  + ".test.sampled.stats.csv";
  return Test_XGBoostModel_StatGenerationAndReading(modelJSONPath, csvPath, statsCSVPath);
}

bool Test_AirlineOHEStatGenerationAndReading(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/airline-ohe_xgb_model_save.json";
  auto csvPath = modelJSONPath  + ".test.sampled.csv";
  auto statsCSVPath = modelJSONPath  + ".test.sampled.stats.csv";
  return Test_XGBoostModel_StatGenerationAndReading(modelJSONPath, csvPath, statsCSVPath);
}

bool Test_CovtypeStatGenerationAndReading(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/covtype_xgb_model_save.json";
  auto csvPath = modelJSONPath  + ".test.sampled.csv";
  auto statsCSVPath = modelJSONPath  + ".test.sampled.stats.csv";
  return Test_XGBoostModel_StatGenerationAndReading(modelJSONPath, csvPath, statsCSVPath);
}

bool Test_EpsilonStatGenerationAndReading(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/epsilon_xgb_model_save.json";
  auto csvPath = modelJSONPath  + ".test.sampled.csv";
  auto statsCSVPath = modelJSONPath  + ".test.sampled.stats.csv";
  return Test_XGBoostModel_StatGenerationAndReading(modelJSONPath, csvPath, statsCSVPath);
}

bool Test_HiggsStatGenerationAndReading(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/higgs_xgb_model_save.json";
  auto csvPath = modelJSONPath  + ".test.sampled.csv";
  auto statsCSVPath = modelJSONPath  + ".test.sampled.stats.csv";
  return Test_XGBoostModel_StatGenerationAndReading(modelJSONPath, csvPath, statsCSVPath);
}

bool Test_YearStatGenerationAndReading(TestArgs_t &args) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models";
  auto modelJSONPath = testModelsDir + "/year_prediction_msd_xgb_model_save.json";
  auto csvPath = modelJSONPath  + ".test.sampled.csv";
  auto statsCSVPath = modelJSONPath  + ".test.sampled.stats.csv";
  return Test_XGBoostModel_StatGenerationAndReading(modelJSONPath, csvPath, statsCSVPath);
}


}
}