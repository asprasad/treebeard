#include <string>
#include "DecisionForest.h"
#include "xgboostparser.h"
#include "TestUtilsCommon.h"

namespace TreeBeard
{
namespace Profile
{
void ComputeForestInferenceStats(const std::string& modelJSONPath, const std::string& csvPath, int32_t numRows) {
  // std::string csvPath = "/home/ashwin/ML/scikit-learn_bench/xgb_models/airline_xgb_model_save.json.test.csv";
  mlir::MLIRContext context;
  TreeBeard::XGBoostJSONParser<> xgBoostParser(context, modelJSONPath, "", 1);
  xgBoostParser.Parse();
  auto decisionForest = xgBoostParser.GetForest();

  TreeBeard::test::TestCSVReader csvReader(csvPath);
  std::cerr << "Done reading csv..\n";
  std::cerr << "Running inference\n";
  for (size_t i=0  ; i<csvReader.NumberOfRows() ; ++i ) {
    auto row = csvReader.GetRowOfType<double>(i);
    row.pop_back();
    decisionForest->Predict(row);
    if (i % (csvReader.NumberOfRows()/25) == 0)
      std::cerr << "-" << std::flush;
  }
  std::cout << decisionForest->NumTrees() << ", " << csvReader.NumberOfRows() <<  std::endl;
  for (size_t i=0 ; i<decisionForest->NumTrees() ; ++i) {
    auto& tree = decisionForest->GetTree(i);
    std::vector<mlir::decisionforest::DecisionTree<>::Node> leaves;
    for (auto& node : tree.GetNodes()) {
      if (node.IsLeaf())
        leaves.push_back(node);
    }
    std::sort(leaves.begin(), leaves.end(), [](mlir::decisionforest::DecisionTree<>::Node& n1, mlir::decisionforest::DecisionTree<>::Node& n2) {
      return n1.hitCount > n2.hitCount;
    });
    std::cout << leaves[0].hitCount << ", "  << leaves[0].depth;
    for (size_t j=1 ; j<leaves.size() ; ++j) {
      std::cout << ", " << leaves[j].hitCount << ", "  << leaves[j].depth;
    }
    std::cout << std::endl;
  }
}

void ComputeForestInferenceStatsOnSampledTestInput(const std::string& model, int32_t numRows) {
  auto repoPath = TreeBeard::test::GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models/";
  auto modelJSONPath = testModelsDir + model + "_xgb_model_save.json";
  auto csvPath = modelJSONPath + ".test.sampled.csv";
  ComputeForestInferenceStats(modelJSONPath, csvPath, numRows);
}

void ComputeForestInferenceStatsOnModel(const std::string& model, const std::string& csvPath, int32_t numRows) {
  auto repoPath = TreeBeard::test::GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models/";
  auto modelJSONPath = testModelsDir + model + "_xgb_model_save.json";
  ComputeForestInferenceStats(modelJSONPath, csvPath, numRows);
}

}
}