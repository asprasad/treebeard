#ifndef _DECISIONFOREST_H_
#define _DECISIONFOREST_H_

#include <vector>
#include <string>
#include <sstream>
namespace mlir
{
namespace decisionforest
{

enum ReductionType { kAdd, kVoting };
enum FeatureType { kNumerical, kCategorical };

template <typename ThresholdType=double, typename ReturnType=double, typename FeatureIndexType=int32_t, typename NodeIndexType=int32_t>
class DecisionTree
{
public:
    static constexpr NodeIndexType INVALID_NODE_INDEX = -1;
    struct Node
    {
        ThresholdType threshold;
        FeatureIndexType featureIndex;
        NodeIndexType parent;
        NodeIndexType leftChild;
        NodeIndexType rightChild;
        FeatureType featureType; // TODO For now assuming everything is numerical

        bool operator==(const Node& that) const
        {
            return threshold==that.threshold && featureIndex==that.featureIndex && parent==that.parent &&
                   leftChild==that.leftChild && rightChild==that.rightChild && featureType==that.featureType;
        }
    };
    void SetNumberOfFeatures(size_t numFeatures) { m_numFeatures = numFeatures; }
    void SetTreeScalingFactor(ThresholdType scale) { m_scale = scale; }

    // Create a new node in the current tree
    NodeIndexType NewNode(ThresholdType threshold, FeatureIndexType featureIndex)
    { 
        Node node{threshold, featureIndex, INVALID_NODE_INDEX, INVALID_NODE_INDEX, INVALID_NODE_INDEX, kNumerical};
        m_nodes.push_back(node);
        return m_nodes.size() - 1;
    }
    // Set the parent of a node
    void SetNodeParent(NodeIndexType node, NodeIndexType parent) { m_nodes[node].parent = parent; }
    // Set right child of a node
    void SetNodeRightChild(NodeIndexType node, NodeIndexType child) { m_nodes[node].rightChild = child; }
    // Set left child of a node
    void SetNodeLeftChild(NodeIndexType node, NodeIndexType child) { m_nodes[node].leftChild = child; }

    std::string Serialize() const;

    bool operator==(const DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>& that) const
    {
        return m_nodes==that.m_nodes && m_numFeatures==that.m_numFeatures && m_scale==that.m_scale;
    }
private:
    std::vector<Node> m_nodes;
    size_t m_numFeatures;
    ThresholdType m_scale;
};

template <typename ThresholdType=double, typename ReturnType=double, typename FeatureIndexType=int32_t, typename NodeIndexType=int32_t>
class DecisionForest
{
public:
    struct Feature
    {
        std::string name;
        std::string type;
    };

    using DecisionTreeType = DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>;
    void SetReductionType(ReductionType reductionType) { m_reductionType = reductionType; }
    void AddFeature(const std::string& featureName, const std::string& type)
    {
        Feature f{featureName, type};
        m_features.push_back(f);
    }
    DecisionTreeType& NewTree()
    { 
        m_trees.push_back(DecisionTreeType());
        return m_trees.back();
    }
    void EndTree() { }
    size_t NumTrees() { return m_trees.size(); }
    const std::vector<Feature>& GetFeatures() const { return m_features; }
    std::string Serialize() const;
    std::string PrintToString() const;
    bool operator==(const DecisionForest<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>& that) const
    {
        return m_reductionType==that.m_reductionType && m_trees==that.m_trees;
    }
private:
    std::vector<Feature> m_features;
    std::vector<DecisionTreeType> m_trees;
    ReductionType m_reductionType;
};

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
std::string DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::Serialize() const
{
    std::stringstream strStream;
    strStream << m_numFeatures << m_scale;
    for (auto& node : m_nodes)
    {
        strStream << node.threshold;
        strStream << node.featureIndex;
        strStream << node.parent;
        strStream << node.leftChild;
        strStream << node.rightChild;
        strStream << node.featureType; 
    }
    return strStream.str();
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
std::string DecisionForest<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::Serialize() const
{
    std::stringstream strStream;
    strStream << m_reductionType << m_trees.size();
    for (auto& tree : m_trees)
        strStream << tree.Serialize();
    return strStream.str();
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
std::string DecisionForest<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::PrintToString() const
{
    std::stringstream strStream;
    strStream << "ReductionType = " << m_reductionType << ", #Trees = " << m_trees.size();
    return strStream.str();
}

} // decisionforest
} // mlir
#endif // _DECISIONFOREST_H_