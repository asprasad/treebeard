#ifndef _MODEL_JSON_PARSER_H_
#define _MODEL_JSON_PARSER_H_

#include <string>
#include <vector>
#include "json.hpp"

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

template<typename ThresholdType, typename FeatureIndexType, typename NodeIndexType>
class ModelJSONParser
{
protected:
    void SetReductionType(ReductionType reductionType) { }
    void AddFeature(const std::string& featureName, FeatureType type) { }
    void NewTree() { }
    void EndTree() { }
    void SetTreeScalingFactor(ThresholdType scale) { }

    // Create a new node in the current tree
    NodeIndexType NewNode(ThresholdType threshold, FeatureIndexType featureIndex) { return 0; }
    // Set the parent of a node
    void SetNodeParent(NodeIndexType node, NodeIndexType parent) { }
    // Set right child of a node
    void SetNodeRightChild(NodeIndexType node, NodeIndexType child) { }
    // Set left child of a node
    void SetNodeLeftChild(NodeIndexType node, NodeIndexType child) { }

public:
    ModelJSONParser() { }
    virtual void Parse() = 0;

    MLIRFuntion GetEvaluationFunction() { return nullptr; }
};
}

#endif //_MODEL_JSON_PARSER_H_