#ifndef _DECISIONFOREST_H_
#define _DECISIONFOREST_H_

#include <vector>
#include <set>
#include <string>
#include <sstream>
#include <numeric>
#include "TreeTilingDescriptor.h"

namespace mlir
{
namespace decisionforest
{

enum class ReductionType { kAdd, kVoting };
enum class FeatureType { kNumerical, kCategorical };

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

        bool IsLeaf() const
        {
            return leftChild == INVALID_NODE_INDEX && rightChild == INVALID_NODE_INDEX;
        }
    };
    void SetNumberOfFeatures(size_t numFeatures) { m_numFeatures = numFeatures; }
    void SetTreeScalingFactor(ThresholdType scale) { m_scale = scale; }

    // Create a new node in the current tree
    NodeIndexType NewNode(ThresholdType threshold, FeatureIndexType featureIndex)
    { 
        Node node{threshold, featureIndex, INVALID_NODE_INDEX, INVALID_NODE_INDEX, INVALID_NODE_INDEX, FeatureType::kNumerical};
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
    std::string PrintToString() const;

    bool operator==(const DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>& that) const
    {
        return m_nodes==that.m_nodes && m_numFeatures==that.m_numFeatures && m_scale==that.m_scale;
    }

    ReturnType PredictTree(std::vector<ThresholdType>& data) const;
    TreeTilingDescriptor& TilingDescriptor() { return m_tilingDescriptor; }
    const TreeTilingDescriptor& TilingDescriptor() const { return m_tilingDescriptor; }

    int32_t GetTreeDepth() {
        return GetTreeDepthHelper(0);
    }
    // The number of entries that would be needed if this tree is serialized 
    // into a dense array reprsentation
    int32_t GetDenseSerializationVectorLength() {
        assert (m_tilingDescriptor.MaxTileSize() == 1 && "Larger tile sizes unimplemented");
        return std::pow(2, GetTreeDepth()) - 1;
    }
    std::vector<ThresholdType> GetThresholdArray();
    std::vector<FeatureIndexType> GetFeatureIndexArray();
    int32_t GetNumberOfTiles();
private:
    std::vector<Node> m_nodes;
    size_t m_numFeatures;
    ThresholdType m_scale;
    TreeTilingDescriptor m_tilingDescriptor;

    int32_t GetTreeDepthHelper(size_t node) const;
    
    template <typename AttribType, typename GetterType>
    void GetNodeAttributeArray(std::vector<AttribType>& thresholdVec, size_t vecIndex, size_t nodeIndex, GetterType get);
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
    DecisionTreeType& GetTree(int64_t index) { return m_trees[index]; }
    const std::vector<Feature>& GetFeatures() const { return m_features; }
    std::string Serialize() const;
    std::string PrintToString() const;
    ReturnType Predict(std::vector<ThresholdType>& data) const;
    bool operator==(const DecisionForest<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>& that) const {
        return m_reductionType==that.m_reductionType && m_trees==that.m_trees;
    }

    int32_t GetDenseSerializationVectorLength() const {
        int32_t size = 0;
        for (auto& tree : m_trees)
            size += tree.GetDenseSerializationVectorLength();
        return size; 
    }

    std::set<int32_t> GetTileSizes() const {
        std::set<int32_t> tileSizes;
        for (auto& tree : m_trees) {
            tileSizes.insert(m_trees.TilingDescriptor().MaxTileSize());
        }
        return tileSizes;
    }

    // Get the serialized representation of the forest. Used to lower ensemble constants into memrefs.
    // TODO Currently assumes that all nodes are numerical
    void GetDenseSerialization(std::vector<ThresholdType>& thresholds, std::vector<FeatureIndexType>& featureIndices,
                               std::vector<int32_t>& offsets);
private:
    std::vector<Feature> m_features;
    std::vector<DecisionTreeType> m_trees;
    ReductionType m_reductionType = ReductionType::kAdd;
};

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
int32_t DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::GetTreeDepthHelper(size_t node) const
{
    const Node& n = this->m_nodes[node];
    if (n.IsLeaf())
        return 1;
    return 1 + std::max(GetTreeDepthHelper(n.leftChild), GetTreeDepthHelper(n.rightChild));
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
template <typename AttribType, typename GetterType>
void DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::GetNodeAttributeArray(std::vector<AttribType>& attributeVec,
                                                                                                     size_t vecIndex, size_t nodeIndex, GetterType get)
{
    Node& node = m_nodes[nodeIndex];
    assert(vecIndex < attributeVec.size());
    // TODO What is the type we set on leaf nodes?
    assert(node.featureType == FeatureType::kNumerical || node.IsLeaf());
    attributeVec[vecIndex] = get(node);

    if (node.IsLeaf())
        return;
    GetNodeAttributeArray<AttribType, GetterType>(attributeVec, 2*vecIndex+1, node.leftChild, get);
    GetNodeAttributeArray<AttribType, GetterType>(attributeVec, 2*vecIndex+2, node.rightChild, get);
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
std::vector<ThresholdType> DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::GetThresholdArray()
{
    int32_t depth = GetTreeDepth();
    size_t vectorLength = static_cast<size_t>(std::pow(2, depth)) - 1;
    std::vector<ThresholdType> thresholdVec(vectorLength, 0.0);
    assert (m_tilingDescriptor.MaxTileSize() == 1 && "Only size 1 tiles currently supported");

    GetNodeAttributeArray(thresholdVec, 0, 0, [](Node& n) { return n.threshold; });
    return thresholdVec;
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
std::vector<FeatureIndexType> DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::GetFeatureIndexArray()
{
    int32_t depth = GetTreeDepth();
    size_t vectorLength = static_cast<size_t>(std::pow(2, depth)) - 1;
    std::vector<FeatureIndexType> featureIndexVec(vectorLength, -1);
    assert (m_tilingDescriptor.MaxTileSize() == 1 && "Only size 1 tiles currently supported");

    GetNodeAttributeArray(featureIndexVec, 0, 0, [](Node& n) { return n.featureIndex; });
    return featureIndexVec;
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
int32_t DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::GetNumberOfTiles()
{
    assert (m_tilingDescriptor.MaxTileSize() == 1 && "Only size 1 tiles currently supported");
    int32_t depth = GetTreeDepth();
    size_t numTiles = static_cast<size_t>(std::pow(2, depth)) - 1;
    return numTiles;
}

// TODO This needs to also include the tiling of the tree
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
        strStream << (int32_t)node.featureType; 
    }
    return strStream.str();
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
std::string DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::PrintToString() const
{
    std::stringstream strStream;
    strStream << "NumberOfFeatures = " << m_numFeatures << ", Scale = " << m_scale << ", NumberOfNodes = " << m_nodes.size();
    return strStream.str();
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
ReturnType DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::PredictTree(std::vector<ThresholdType>& data) const
{
    // go over the features
    assert(m_nodes.size() > 0);
    const Node* node = &m_nodes[0]; // root node
    while (!node->IsLeaf())
    {
      if (node->threshold < data[node->featureIndex])
        node = &m_nodes[node->leftChild];
      else
        node = &m_nodes[node->rightChild];
    }    
    return node->threshold;
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
std::string DecisionForest<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::Serialize() const
{
    std::stringstream strStream;
    strStream << (int32_t)m_reductionType << m_trees.size();
    for (auto& tree : m_trees)
        strStream << tree.Serialize();
    return strStream.str();
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
std::string DecisionForest<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::PrintToString() const
{
    std::stringstream strStream;
    strStream << "ReductionType = " << (int32_t)m_reductionType << ", #Trees = " << m_trees.size();
    return strStream.str();
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
ReturnType DecisionForest<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::Predict(std::vector<ThresholdType>& data) const
{
    std::vector<ReturnType> predictions;
    for (auto& tree: m_trees)
        predictions.push_back(tree.PredictTree(data));
    
    assert(m_reductionType == ReductionType::kAdd);
    return std::accumulate(predictions.begin(), predictions.end(), 0.0);
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
void DecisionForest<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::GetDenseSerialization(
                    std::vector<ThresholdType>& thresholds, std::vector<FeatureIndexType>& featureIndices,
                    std::vector<int32_t>& offsets)
{
    int32_t currentOffset = 0;
    for (auto& tree : m_trees) {
        assert(currentOffset == static_cast<int32_t>(thresholds.size()));
        offsets.push_back(currentOffset);
        
        auto treeThresholds = tree.GetThresholdArray();
        auto treeFeatureIndices = tree.GetFeatureIndexArray();
        
        thresholds.insert(thresholds.end(), treeThresholds.begin(), treeThresholds.end());
        featureIndices.insert(featureIndices.end(), treeFeatureIndices.begin(), treeFeatureIndices.end());

        assert (treeThresholds.size() == treeFeatureIndices.size());
        currentOffset += treeThresholds.size();
    }
}

} // decisionforest
} // mlir
#endif // _DECISIONFOREST_H_