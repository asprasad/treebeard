#ifndef _DECISIONFOREST_H_
#define _DECISIONFOREST_H_

#include <vector>

namespace mlir
{
namespace decisionforest
{

enum ReductionType { kAdd, kVoting };
enum FeatureType { kNumerical, kCategorical };

template <typename ThresholdType=double, typename ReturnType=double, typename FeatureIndexType=size_t, typename NodeIndexType=size_t>
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
private:
    std::vector<Node> m_nodes;
    size_t m_numFeatures;
    ThresholdType m_scale;
};

template <typename ThresholdType=double, typename ReturnType=double, typename FeatureIndexType=size_t, typename NodeIndexType=size_t>
class DecisionForest
{
public:
    using DecisionTreeType = DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>;
    void SetReductionType(ReductionType reductionType) { m_reductionType = reductionType; }
    void AddFeature(const std::string& featureName, FeatureType type) { }
    DecisionTreeType& NewTree()
    { 
        m_trees.push_back(DecisionTreeType());
        return m_trees.back();
    }
    void EndTree() { }
private:
    std::vector<DecisionTreeType> m_trees;
    ReductionType m_reductionType;
};

}
}
#endif // _DECISIONFOREST_H_