#include "DecisionForest.h"
#include "../test/TestUtilsCommon.h"
#include "json.hpp"
#include <fstream>
#include <queue>
#include <map>

using namespace mlir;
using namespace mlir::decisionforest;

using json = nlohmann::json;

namespace TreeBeard
{
namespace test
{
namespace
{
int32_t GenerateRandomTreeHelper(DecisionTree<>& tree, int32_t numFeatures, int32_t depth, int32_t maxDepth, 
                             RandomIntGenerator& featureIdxGen, RandomRealGenerator& thresholdGen, RandomRealGenerator& nodeTypeGen, double leafCutoff) {
  bool isLeaf = nodeTypeGen() < leafCutoff || depth == maxDepth;

  if (isLeaf) {
    // Generate a leaf
    auto threshold = thresholdGen();
    auto leaf = tree.NewNode(threshold, -1);
    return leaf;
  }
  else {
    // Create new node
    auto threshold = thresholdGen();
    auto featureIdx = featureIdxGen();
    auto node = tree.NewNode(threshold, featureIdx);

    // Recurse
    auto leftChild = GenerateRandomTreeHelper(tree, numFeatures, depth+1, maxDepth, featureIdxGen, thresholdGen, nodeTypeGen, (double)depth/maxDepth);
    auto rightChild = GenerateRandomTreeHelper(tree, numFeatures, depth+1, maxDepth, featureIdxGen, thresholdGen, nodeTypeGen, (double)depth/maxDepth);
    tree.SetNodeParent(leftChild, node);
    tree.SetNodeParent(rightChild, node);
    tree.SetNodeRightChild(node, rightChild);
    tree.SetNodeLeftChild(node, leftChild);
    
    return node;
  }
}
}

DecisionTree<> GenerateRandomTree(int32_t numFeatures, double thresholdMin, double thresholdMax, int32_t maxDepth) {
  DecisionTree<> tree;
  RandomIntGenerator featureIdxGen = [=]() { return GetRandomInt(0, numFeatures-1); };
  RandomRealGenerator thresholdGen = [=]() { return GetRandomReal(thresholdMin, thresholdMax); };
  RandomRealGenerator nodeTypeGen = [=]() { return GetRandomReal(0.0, 1.0); };
  GenerateRandomTreeHelper(tree, numFeatures, 0, maxDepth, featureIdxGen, thresholdGen, nodeTypeGen, 0.0);
  return tree;
}

DecisionForest<> GenerateRandomDecisionForest(int32_t numTrees, int32_t numFeatures, double thresholdMin, double thresholdMax, int32_t maxDepth) {
  DecisionForest<> forest;
  for (int32_t i=0; i<numFeatures ; ++i)
    forest.AddFeature(std::to_string(i) /*name*/, "float" /*type*/);
  for (int32_t i=0 ; i<numTrees ; ++i) {
    auto& tree = forest.NewTree();
    tree = GenerateRandomTree(numFeatures, thresholdMin, thresholdMax, maxDepth);
  }
  return forest;
}

json CreateTreeJSON(DecisionTree<>& tree, int32_t id, int32_t numFeatures) {
  json treeJSON;
  treeJSON["id"] = id;
  treeJSON["tree_param"]["num_deleted"] = "0";
  treeJSON["tree_param"]["num_feature"] = std::to_string(numFeatures);
  treeJSON["tree_param"]["num_nodes"] = std::to_string(tree.GetNodes().size());
  treeJSON["tree_param"]["size_leaf_vector"] = "0";

  treeJSON["loss_changes"] = std::vector<double>(tree.GetNodes().size(), 0.0);
  treeJSON["sum_hessian"] = std::vector<double>(tree.GetNodes().size(), 0.0);
  treeJSON["base_weights"] = std::vector<double>(tree.GetNodes().size(), 0.0);
  treeJSON["default_left"] = std::vector<bool>(tree.GetNodes().size(), false);
  treeJSON["categories"] = "[]"_json;
  treeJSON["categories_nodes"] = "[]"_json;
  treeJSON["categories_segments"] = "[]"_json;
  treeJSON["categories_sizes"] = "[]"_json;

  std::vector<int32_t> leftChildren, rightChildren, parents, splitIndices;
  std::vector<double> splitConditions;
  mlir::decisionforest::LevelOrderTraversal levelOrder(tree.GetNodes());
  const auto& nodes = levelOrder.LevelOrderNodes();
  for (size_t i=0 ; i<nodes.size() ; ++i) {
    const auto& node = nodes[i];
    leftChildren.push_back(node.leftChild);
    rightChildren.push_back(node.rightChild);
    if (node.featureIndex != -1)
      splitIndices.push_back(node.featureIndex);
    else
      splitIndices.push_back(0);
    splitConditions.push_back(node.threshold);
    if (node.parent == DecisionTree<>::INVALID_NODE_INDEX)
      parents.push_back(2147483647); //Some XGBoost weirdness for the parent of the root
    else
      parents.push_back(node.parent);
  }
  treeJSON["left_children"] = leftChildren;
  treeJSON["right_children"] = rightChildren;
  treeJSON["parents"] = parents;
  treeJSON["split_indices"] = splitIndices;
  treeJSON["split_conditions"] = splitConditions;

  return treeJSON;
}

json CreateModelJSON(DecisionForest<>& forest) {
  json modelJSON;
  modelJSON["gbtree_model_param"]["num_trees"] = std::to_string(forest.NumTrees());
  modelJSON["gbtree_model_param"]["size_leaf_vector"] = "0";
  modelJSON["tree_info"] = std::vector<int32_t>(forest.NumTrees(), 0);
  for (size_t i=0 ; i<forest.NumTrees() ; ++i)
    modelJSON["trees"].push_back(CreateTreeJSON(forest.GetTree(i), (int32_t)i, (int32_t)forest.GetFeatures().size()));
  return modelJSON;
}

void SaveToXGBoostJSON(DecisionForest<>& forest, const std::string& filename) {
  json forestJSON;
  forestJSON["version"] = { 1, 4, 0 };

  json learnerJSON;
  learnerJSON["attributes"] = "{ }"_json;
  learnerJSON["feature_types"] = std::vector<std::string>(forest.GetFeatures().size(), "float");

  json learnerModelParamJSON;
  learnerModelParamJSON["base_score"] = "5E-1";
  learnerModelParamJSON["num_class"] = "0";
  learnerModelParamJSON["num_feature"] = std::to_string(forest.GetFeatures().size());
  learnerJSON["learner_model_param"] = learnerModelParamJSON;

  json objectiveJSON;
  
  objectiveJSON["name"] = "reg:squarederror";
  objectiveJSON["reg_loss_param"]["scale_pos_weight"] = "1";
  learnerJSON["objective"] = objectiveJSON;
  learnerJSON["gradient_booster"]["model"] = CreateModelJSON(forest);
  learnerJSON["gradient_booster"]["name"] = "gbtree";
  forestJSON["learner"] = learnerJSON;

  std::ofstream fout(filename);
  fout << forestJSON;
}

void GenerateRandomModelJSONs(const std::string& dirname, int32_t numberOfModels, int32_t maxNumTrees, 
                              int32_t maxNumFeatures, double thresholdMin, double thresholdMax, int32_t maxDepth) {
    for(int32_t i=0 ; i<numberOfModels ; ++i) {
      std::string filename = "TestModel_Size" + std::to_string(maxNumTrees) + "_" + std::to_string(i+1) + ".json";
      std::string filepath = dirname + "/" + filename;
      auto numTrees = maxNumTrees; // GetRandomInt(1, maxNumTrees);
      std::cout << "Number of trees : " << numTrees << std::endl;
      auto numFeatures = GetRandomInt(1, maxNumFeatures);

      auto forest = GenerateRandomDecisionForest(numTrees, numFeatures, thresholdMin, thresholdMax, maxDepth);
      SaveToXGBoostJSON(forest, filepath);
    }

}

} // test
} // treebeard