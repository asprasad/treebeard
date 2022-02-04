#ifndef _XGBOOST_PARSER_H_
#define _XGBOOST_PARSER_H_

#include "modeljsonparser.h"
#include <fstream>

namespace TreeBeard
{

template<typename ThresholdType=double, typename ReturnType=double, typename FeatureIndexType=int32_t, 
         typename NodeIndexType=int32_t, typename InputElementType=double>
class XGBoostJSONParser : public ModelJSONParser<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType, InputElementType>
{
    json m_json;
    void ConstructSingleTree(json& treeJSON);
    void ConstructTreesFromBooster(json& boosterJSON);
    static constexpr double_t INITIAL_VALUE = 0;

public:
    XGBoostJSONParser(mlir::MLIRContext& context, const std::string& filename, const std::string& modelGlobalsJSONFilePath, int32_t batchSize)
        :ModelJSONParser<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType, InputElementType>(filename, modelGlobalsJSONFilePath, context, batchSize, INITIAL_VALUE)
    {
        std::ifstream fin(filename);
        assert (fin);
        fin >> m_json;
    }
    void Parse() override;
};
/*
TOP LEVEL : 
  "required": [
    "version",
    "learner"
  ]
LEARNER (Also includes an objective that is ignored here)
 "learner": {
      "type": "object",
      "properties": {
        "feature_names": {
          "type": "array",
          "items": {
              "type": "string"
          }
        },
        "feature_types": {
          "type": "array",
          "items": {
              "type": "string"
          }
        },
        "gradient_booster": {
            {
              "$ref": "#/definitions/gbtree"
            }
        }
    }
*/
inline double TransformBaseScore(const std::string& objectiveName, double val) {
  if (objectiveName == "binary:logistic")
    return -log(1.0/val - 1.0);
  else if (objectiveName == "reg:squarederror" || objectiveName=="multi:softmax")
    return val;
  else
    assert(false && "Unknown objective type");
  return val;
}

inline mlir::decisionforest::PredictionTransformation GetPredictionTransformType(const std::string& objectiveName) {
  if (objectiveName == "binary:logistic")
    return mlir::decisionforest::PredictionTransformation::kSigmoid;
  else if (objectiveName == "reg:squarederror")
    return mlir::decisionforest::PredictionTransformation::kIdentity;
  else if (objectiveName=="multi:softmax")
    return mlir::decisionforest::PredictionTransformation::kSoftMax;
  else
    assert(false && "Unknown objective type");
  return mlir::decisionforest::PredictionTransformation::kUnknown;
}

template<typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType, typename InputElementType>
void XGBoostJSONParser<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType, InputElementType>::Parse()
{
    auto& learnerJSON = m_json["learner"];
    auto& featureNamesJSON = learnerJSON["feature_names"];
    auto& featureTypesJSON = learnerJSON["feature_types"];
    
    // Set the base score for current objective type
    auto baseScoreStr = learnerJSON["learner_model_param"]["base_score"].get<std::string>();
    auto baseScore = std::stod(baseScoreStr);
    auto objectiveName = learnerJSON["objective"]["name"].get<std::string>();
    this->SetInitialOffset(TransformBaseScore(objectiveName, baseScore));
    this->SetNumberOfClasses(std::stoi(learnerJSON["learner_model_param"]["num_class"].get<std::string>()));
    this->m_forest->SetPredictionTransformation(GetPredictionTransformType(objectiveName));
    
    // Assert is not valid since feature_names is not required. 
    // assert(featureNamesJSON.size() == featureTypesJSON.size());
    for (size_t i = 0; i<featureTypesJSON.size() ; ++i)
    {
        std::string name;
        if (featureNamesJSON.size() == 0)
            name = std::to_string(i);
        else
            name = featureNamesJSON[i].get<std::string>();
        auto featureType = featureTypesJSON[i].get<std::string>();
        this->AddFeature(name, featureType); //TODO hardcoded feature type
    }

    ConstructTreesFromBooster(learnerJSON["gradient_booster"]);
}
// We asumme the gradient booster is a gbtree
/*
    "required": [
    "name",
    "model"
    ]
*/
// Name is a const string string "gbtree"

template<typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType, typename InputElementType>
void XGBoostJSONParser<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType, InputElementType>::ConstructTreesFromBooster(json& boosterJSON)
{
    auto& modelJSON = boosterJSON["model"];
    size_t numTrees = static_cast<size_t>(std::stoi(modelJSON["gbtree_model_param"]["num_trees"].get<std::string>()));
    auto& treesJSON = modelJSON["trees"];
    auto& treeInfoJSON = modelJSON["tree_info"];

    assert (numTrees == treesJSON.size());
    assert (numTrees == treeInfoJSON.size());

    int32_t treeIndex = 0;
    for (auto& treeJSON : treesJSON)
    {
        this->NewTree();
        ConstructSingleTree(treeJSON);
        this->SetTreeGroupId(treeInfoJSON[treeIndex++]);
        this->EndTree();
    }
}

template<typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType, typename InputElementType>
void XGBoostJSONParser<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType, InputElementType>::ConstructSingleTree(json& treeJSON)
{
    // TODO what is "base_weights", "categories", "categories_nodes", 
    // "categories_segments", "categories_sizes"?
    size_t numNodes = treeJSON["base_weights"].size();
    // TODO ignoring "default_left"
    // auto treeID = treeJSON["id"].get<int>();
    
    auto& left_children = treeJSON["left_children"];
    auto& right_childen = treeJSON["right_children"];
    auto& parents = treeJSON["parents"];
    auto& split_conditions = treeJSON["split_conditions"];
    auto& split_indices = treeJSON["split_indices"];
    // auto& split_type = treeJSON["split_type"]; // 0 is Numerical and 1 is Categorical
    auto num_features = std::stoi(treeJSON["tree_param"]["num_feature"].get<std::string>());
    auto num_nodes = static_cast<size_t>(std::stoi(treeJSON["tree_param"]["num_nodes"].get<std::string>()));
    assert (numNodes == num_nodes);
    assert (left_children.size() == num_nodes);
    assert (left_children.size() == right_childen.size() && 
            left_children.size() == parents.size());
    this->SetTreeNumberOfFeatures(num_features);

    std::vector<NodeIndexType> nodes;
    for (size_t i=0 ; i< num_nodes ; ++i)
    {
        // assert (split_type[i].get<int>() == 0); // only numerical splits for now
        auto node = this->NewNode(split_conditions[i].get<ThresholdType>(), split_indices[i].get<FeatureIndexType>());
        nodes.push_back(node);
    }
    for (size_t i=0 ; i< num_nodes ; ++i)
    {
        auto leftChildIndex = left_children[i].get<int>();
        if (leftChildIndex != -1)
            this->SetNodeLeftChild(nodes[i], nodes[leftChildIndex]);
        auto rightChildIndex = right_childen[i].get<int>();
        if (rightChildIndex != -1)
            this->SetNodeRightChild(nodes[i], nodes[rightChildIndex]);
        if (parents[i].get<int>() == 2147483647)
            this->SetNodeParent(nodes[i],  -1);
        else
            this->SetNodeParent(nodes[i], nodes[parents[i].get<int>()]);
    }
}

}


#endif //_XGBOOST_PARSER_H_