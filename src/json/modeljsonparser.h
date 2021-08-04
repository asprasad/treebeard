#ifndef _MODEL_JSON_PARSER_H_
#define _MODEL_JSON_PARSER_H_

#include <string>
#include <vector>
#include "json.hpp"
#include "DecisionForest.h"

using json = nlohmann::json;

// TODO replace with the actual MLIR function/module type
typedef void* MLIRFuntion;

/*
//++
// The abstract base class from which all JSON model parsers derive.
// Also implements callbacks that construct the actual MLIR structures.
//--
*/
namespace TreeHeavy
{
enum ReductionType { kAdd, kVoting };
enum FeatureType { kNumerical, kCategorical };

template<typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
class ModelJSONParser
{
protected:
    using DecisionForestType = mlir::decisionforest::DecisionForest<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>;
    using DecisionTreeType = typename DecisionForestType::DecisionTreeType;

    DecisionForestType *m_forest;
    DecisionTreeType *m_currentTree;

    void SetReductionType(ReductionType reductionType) { }
    void AddFeature(const std::string& featureName, FeatureType type) { }
    void NewTree() { m_currentTree = &(m_forest->NewTree()); }
    void EndTree() { m_currentTree = nullptr; }
    void SetTreeNumberOfFeatures(size_t numFeatures) { m_currentTree->SetNumberOfFeatures(numFeatures); }
    void SetTreeScalingFactor(ThresholdType scale) { m_currentTree->SetTreeScalingFactor(scale); }

    // Create a new node in the current tree
    NodeIndexType NewNode(ThresholdType threshold, FeatureIndexType featureIndex) { return m_currentTree->NewNode(threshold, featureIndex); }
    // Set the parent of a node
    void SetNodeParent(NodeIndexType node, NodeIndexType parent) { m_currentTree->SetNodeParent(node, parent); }
    // Set right child of a node
    void SetNodeRightChild(NodeIndexType node, NodeIndexType child) { m_currentTree->SetNodeRightChild(node, child); }
    // Set left child of a node
    void SetNodeLeftChild(NodeIndexType node, NodeIndexType child) { m_currentTree->SetNodeLeftChild(node, child); }

public:
    ModelJSONParser()
        : m_forest(new DecisionForestType), m_currentTree(nullptr)
    { }
    virtual void Parse() = 0;

    MLIRFuntion GetEvaluationFunction() { return nullptr; }
};
}

#endif //_MODEL_JSON_PARSER_H_