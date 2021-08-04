#ifndef _XGBOOST_PARSER_H_
#define _XGBOOST_PARSER_H_

#include "modeljsonparser.h"
#include <fstream>

namespace TreeHeavy
{

template<typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
class XGBoostJSONParser : public ModelJSONParser<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>
{
    json m_json;
    void ConstructSingleTree(json& treeJSON);
    void ConstructTreesFromBooster(json& boosterJSON);

public:
    XGBoostJSONParser(const std::string& filename)
    {
        std::ifstream fin(filename);
        fin >> m_json;
    }
    void Parse() override;
};

template<typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
void XGBoostJSONParser<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::Parse()
{
    auto& learnerJSON = m_json["learner"];
    auto& featureNamesJSON = learnerJSON["feature_names"];
    auto& featureTypesJSON = learnerJSON["feature_types"];

    assert(featureNamesJSON.size() == featureTypesJSON.size());
    for (size_t i = 0; i<featureNamesJSON.size() ; ++i)
    {
        auto name = featureNamesJSON[i].get<std::string>();
        auto featureType = featureTypesJSON[i].get<std::string>();
        this->AddFeature(name, static_cast<FeatureType>(0)); //TODO hardcoded feature type
    }

    ConstructTreesFromBooster(learnerJSON["gradient_booster"]);
}

template<typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
void XGBoostJSONParser<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::ConstructTreesFromBooster(json& boosterJSON)
{
    auto& modelJSON = boosterJSON["model"];
    size_t numTrees = static_cast<size_t>(std::stoi(modelJSON["gbtree_model_param"]["num_trees"].get<std::string>()));
    auto& treesJSON = modelJSON["trees"];
    assert (numTrees == treesJSON.size());
    for (auto& treeJSON : treesJSON)
    {
        this->NewTree();
        ConstructSingleTree(treeJSON);
        this->EndTree();
    }
}

template<typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
void XGBoostJSONParser<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::ConstructSingleTree(json& treeJSON)
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
    auto& split_type = treeJSON["split_type"]; // 0 is Numerical and 1 is Categorical
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
        assert (split_type[i].get<int>() == 0); // only numerical splits for now
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