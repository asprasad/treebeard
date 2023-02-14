#include <string>
#include "DecisionForest.h"
#include "xgboostparser.h"
#include "TestUtilsCommon.h"
#include "ModelSerializers.h"

namespace
{

void ComputeForestInferenceStatsImpl(const std::string& modelJSONPath, const std::string& csvPath, int32_t numRows, std::ostream& outputStream, bool sortLeaves) {
  // std::string csvPath = "/home/ashwin/ML/scikit-learn_bench/xgb_models/airline_xgb_model_save.json.test.csv";
  mlir::MLIRContext context;
  TreeBeard::XGBoostJSONParser<> xgBoostParser(context, modelJSONPath, mlir::decisionforest::ConstructModelSerializer(""), 1);
  xgBoostParser.Parse();
  auto decisionForest = xgBoostParser.GetForest();

  TreeBeard::test::TestCSVReader csvReader(csvPath);
  // std::cerr << "Done reading csv..\n";
  // std::cerr << "Running inference\n";
  for (size_t i=0  ; i<csvReader.NumberOfRows() ; ++i ) {
    auto row = csvReader.GetRowOfType<double>(i);
    row.pop_back();
    decisionForest->Predict(row);
    // if (i % (csvReader.NumberOfRows()/25) == 0)
    //   std::cerr << "-" << std::flush;
  }
  outputStream << decisionForest->NumTrees() << ", " << csvReader.NumberOfRows() <<  std::endl;
  for (size_t i=0 ; i<decisionForest->NumTrees() ; ++i) {
    auto& tree = decisionForest->GetTree(i);
    std::vector<mlir::decisionforest::DecisionTree<>::Node> leaves;
    for (auto& node : tree.GetNodes()) {
      if (node.IsLeaf())
        leaves.push_back(node);
    }
    if (sortLeaves)
      std::sort(leaves.begin(), leaves.end(), [](mlir::decisionforest::DecisionTree<>::Node& n1, mlir::decisionforest::DecisionTree<>::Node& n2) {
        return n1.hitCount > n2.hitCount;
      });
    outputStream << leaves[0].hitCount << ", "  << leaves[0].depth;
    for (size_t j=1 ; j<leaves.size() ; ++j) {
      outputStream << ", " << leaves[j].hitCount << ", "  << leaves[j].depth;
    }
    outputStream << std::endl;
  }
}

}

namespace TreeBeard
{
namespace Profile
{

void ComputeForestInferenceStats(const std::string& modelJSONPath, const std::string& csvPath, int32_t numRows) {
  ComputeForestInferenceStatsImpl(modelJSONPath, csvPath, numRows, std::cout, true);
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

void ComputeForestProbabilityProfile(const std::string& modelJSONPath, const std::string& csvPath, const std::string& statsCSVPath, int32_t numRows) {
  std::ofstream fout(statsCSVPath);
  ComputeForestInferenceStatsImpl(modelJSONPath, csvPath, numRows, fout, false);
}

void ComputeForestProbabilityProfileForXGBoostModel(const std::string& modelName, const std::string& csvPath, const std::string& statsCSVPath, int32_t numRows) {
  std::ofstream fout(statsCSVPath);
  auto repoPath = TreeBeard::test::GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/xgb_models/";
  auto modelJSONPath = testModelsDir + modelName + "_xgb_model_save.json";
  ComputeForestInferenceStatsImpl(modelJSONPath, csvPath, numRows, fout, false);
}

void ReadProbabilityProfile(mlir::decisionforest::DecisionForest<>& decisionForest, const std::string& statsCSVFile) {
  TreeBeard::test::TestCSVReader csvReader(statsCSVFile);
  // std::cerr << "Done reading stats file..\n";

  auto firstRow = csvReader.GetRow(0);
  assert (firstRow.size() == 2);
  size_t numTrees = (size_t)firstRow.at(0);
  assert (numTrees == decisionForest.NumTrees());
  // int32_t totalInputRows = (int32_t)firstRow.at(1);
  // assert(decisionForest.NumTrees() == (csvReader.NumberOfRows() - 1));
  for(size_t i=1 ; i<(1 + numTrees); ++i) {
    auto row = csvReader.GetRow(i);
    auto nodes = decisionForest.GetTree(i-1).GetNodes();
    assert (row.size() % 2 == 0);

    std::vector<size_t> leafIndices;
    for (size_t j=0 ; j<nodes.size() ; ++j) {
      if (nodes.at(j).IsLeaf())
        leafIndices.push_back(j);
    }
    assert (leafIndices.size() == row.size()/2);
    for (size_t j=0 ; j<row.size() ; j+=2) {
      int32_t hitCount = (int32_t)row.at(j);
      int32_t depth = (int32_t)row.at(j+1);

      nodes.at(leafIndices.at(j/2)).hitCount = hitCount;
      nodes.at(leafIndices.at(j/2)).depth = depth;
    }
    decisionForest.GetTree(i-1).SetNodes(nodes);
  }
}

}
}