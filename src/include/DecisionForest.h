#ifndef _DECISIONFOREST_H_
#define _DECISIONFOREST_H_

#include <vector>
#include <set>
#include <string>
#include <sstream>
#include <numeric>
#include <cassert>
#include <cmath>
#include <queue>
#include <map>
#include <iostream>
#include "TreeTilingDescriptor.h"
#include <numeric>
#include <algorithm>
#include <memory>

namespace mlir
{
namespace decisionforest
{

class TiledTree;

enum class PredictionTransformation { kIdentity, kSigmoid, kSoftMax, kUnknown };
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
        int32_t hitCount = 0;
        int32_t depth = -1;
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
    // Create a new node in the current tree, and add it to a specified tile
    NodeIndexType NewNode(ThresholdType threshold, FeatureIndexType featureIndex, int32_t tileID)
    { 
        auto nodeIndex = NewNode(threshold, featureIndex);
        m_tilingDescriptor.TileIDs().push_back(tileID);
        return nodeIndex;
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
        return m_nodes==that.m_nodes && m_numFeatures==that.m_numFeatures 
               && m_scale==that.m_scale && m_tilingDescriptor==that.m_tilingDescriptor;
    }

    ReturnType PredictTree(std::vector<ThresholdType>& data) const;
    float PredictTree_Float(std::vector<ThresholdType>& data) const;

    TreeTilingDescriptor& TilingDescriptor() { return m_tilingDescriptor; }
    const TreeTilingDescriptor& TilingDescriptor() const { return m_tilingDescriptor; }
    void SetTilingDescriptor(const TreeTilingDescriptor& descriptor) { m_tilingDescriptor = descriptor; }

    int32_t GetTreeDepth() {
        return GetTreeDepthHelper(0);
    }

    int32_t NumLeaves() {
        int numNodes = 0;
        for (auto& node : m_nodes)
            if (node.IsLeaf())
                ++numNodes;
        return numNodes;
    }

    std::vector<ThresholdType> GetThresholdArray();
    std::vector<FeatureIndexType> GetFeatureIndexArray();
    
    // Helpers for sparse representation
    std::vector<ThresholdType> GetSparseThresholdArray();
    std::vector<FeatureIndexType> GetSparseFeatureIndexArray();
    std::vector<int32_t> GetChildIndexArray();
    
    int32_t GetNumberOfTiles();
    void WriteToDOTFile(std::ostream& fout);
    void WriteToDOTFile(const std::string& filename);

    const std::vector<Node>& GetNodes() { return m_nodes; }
    void SetNodes(const std::vector<Node>& nodes) { m_nodes=nodes; }

    void SetClassId(int32_t classId) { m_classId = classId; }
    int32_t GetClassId() const { return m_classId; }
    int32_t NumFeatures();

    void InitializeInternalNodeHitCounts();
    int32_t GetSubtreeHitCount(int32_t nodeIndex);
    TiledTree* GetTiledTree();
private:
    std::vector<Node> m_nodes;
    size_t m_numFeatures = 0;
    ThresholdType m_scale;
    TreeTilingDescriptor m_tilingDescriptor;
    // TODO It looks like some tests aren't setting this property at all! 
    // Adding an initialization to make sure we aren't accessing unitialized memory
    int32_t m_classId = 0;
    
    std::shared_ptr<TiledTree> m_tiledTree = nullptr;

    int32_t GetTreeDepthHelper(size_t node) const;
    
    template <typename AttribType, typename GetterType>
    void GetNodeAttributeArray(std::vector<AttribType>& thresholdVec, size_t vecIndex, size_t nodeIndex, GetterType get);
};

template <typename ThresholdType=double, typename ReturnType=double, typename FeatureIndexType=int32_t, typename NodeIndexType=int32_t>
class DecisionForest
{
public:
    DecisionForest(ReturnType initialValue) : m_initialValue(initialValue), m_predictionTransform(PredictionTransformation::kUnknown), m_numClasses(0) {}
    DecisionForest() : DecisionForest(0.0) {}
    // DecisionForest(const DecisionForest<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>&) = delete;

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
        m_trees.push_back(std::make_shared<DecisionTreeType>());
        return *(m_trees.back());
    }
    void EndTree() { }
    size_t NumTrees() { return m_trees.size(); }
    DecisionTreeType& GetTree(int64_t index) { return *(m_trees.at(index)); }
    const std::vector<Feature>& GetFeatures() const { return m_features; }
    ReturnType GetInitialOffset() const { return m_initialValue; }
    void SetInitialOffset(ReturnType val) { m_initialValue = val; }
    std::string Serialize() const;
    std::string PrintToString() const;
    ReturnType Predict(std::vector<ThresholdType>& data) const;
    float Predict_Float(std::vector<ThresholdType>& data) const;
    
    bool operator==(const DecisionForest<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>& that) const {
        if (m_reductionType!=that.m_reductionType)
            return false;
        if (m_trees.size()!=that.m_trees.size())
            return false;
        for (size_t i=0 ; i<m_trees.size() ; ++i)
            if (!(*m_trees[i] == *(that.m_trees[i])))
                return false;
        return true;
    }
    
    void SetPredictionTransformation(PredictionTransformation val) { m_predictionTransform = val; }
    PredictionTransformation GetPredictionTransformation() const { return m_predictionTransform; }

    void SetNumClasses(int32_t numClasses) { m_numClasses = numClasses; }
    int32_t GetNumClasses() { return m_numClasses; }
    bool IsMultiClassClassifier() { return m_numClasses > 0; }

    std::vector<std::shared_ptr<DecisionTreeType>>& GetTrees() { return m_trees; }
private:
    std::vector<Feature> m_features;
    std::vector<std::shared_ptr<DecisionTreeType>> m_trees;
    ReductionType m_reductionType = ReductionType::kAdd;
    ReturnType m_initialValue;
    PredictionTransformation m_predictionTransform;
    int32_t m_numClasses;
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

    GetNodeAttributeArray(featureIndexVec, 0, 0, [](Node& n) { return n.IsLeaf() ? -1 : n.featureIndex; });
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
    int32_t depth = 0;
    while (!node->IsLeaf())
    {
      // std::cout << "\tf" << node->featureIndex << "(" << data[node->featureIndex] << ")" << " < " << node->threshold << std::endl;
      if (data[node->featureIndex] < node->threshold)
        node = &m_nodes[node->leftChild];
      else
        node = &m_nodes[node->rightChild];
      ++depth;
    }
    const_cast<Node*>(node)->hitCount++;
    const_cast<Node*>(node)->depth = depth;
    return node->threshold;
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
float DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::PredictTree_Float(std::vector<ThresholdType>& data) const
{
    // go over the features
    assert(m_nodes.size() > 0);
    const Node* node = &m_nodes[0]; // root node
    while (!node->IsLeaf())
    {
      // std::cout << "\tf" << node->featureIndex << "(" << data[node->featureIndex] << ")" << " < " << node->threshold << std::endl;
      if ((float)data[node->featureIndex] < (float)node->threshold)
        node = &m_nodes[node->leftChild];
      else
        node = &m_nodes[node->rightChild];
    }    
    return (float)node->threshold;
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
void DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::WriteToDOTFile(std::ostream& fout)
{
  fout << "digraph {\n";
  for (size_t i=0 ; i<m_nodes.size() ; ++i) {
    int64_t parentIndex = m_nodes[i].parent;
    fout << "\t\"node" << i << "\" [ label = \"Id:" << i << ", Thres:" << m_nodes[i].threshold << ", FeatIdx:" << m_nodes[i].featureIndex << "\"];\n";
    if (parentIndex != INVALID_NODE_INDEX) {
      auto& parentNode = m_nodes.at(parentIndex);
      std::string color = (parentNode.leftChild == static_cast<int32_t>(i)) ? "green" : "red";
      fout << "\t\"node" << parentIndex << "\" -> \"node" << i << "\" [color=" << color << "];\n";
    }
  }
  fout << "}\n";
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
void DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::WriteToDOTFile(const std::string& filename)
{
    std::ofstream fout(filename);
    WriteToDOTFile(fout);
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
int32_t DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::NumFeatures()
{
    std::set<int32_t> featureSet;
    for (auto& node : m_nodes) {
        if (!node.IsLeaf())
            featureSet.insert(node.featureIndex);
    }
    return featureSet.size();
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
void DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::InitializeInternalNodeHitCounts() {
    GetSubtreeHitCount(0);
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
int32_t DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::GetSubtreeHitCount(int32_t nodeIndex) {
    Node& node = m_nodes.at(nodeIndex);
    if (node.IsLeaf())
        return node.hitCount;
    node.hitCount = GetSubtreeHitCount(node.leftChild) + GetSubtreeHitCount(node.rightChild);
    return node.hitCount;
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
std::string DecisionForest<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::Serialize() const
{
    std::stringstream strStream;
    strStream << (int32_t)m_reductionType << m_trees.size() << m_initialValue;
    for (auto& tree : m_trees)
        strStream << tree->Serialize();
    return strStream.str();
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
std::string DecisionForest<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::PrintToString() const
{
    std::stringstream strStream;
    strStream << "ReductionType = " << (int32_t)m_reductionType << ", #Trees = " << m_trees.size() << ", InitialValue=" << m_initialValue;
    return strStream.str();
}

template<typename FPType>
FPType sigmoid(FPType val) {
  return 1.0/(1.0 + std::exp(-val));
}

template<typename FPType, typename ReturnType>
ReturnType argmax(std::vector<FPType>& classProbabilities) {
    // Note: Commented out code implements the actual formula of softmax. 
    // For our purpose, we dont' really need this since softmax is monotonically increasing.
    // std::transform(
    //     classProbabilities.begin(),
    //     classProbabilities.end(),
    //     classProbabilities.begin(),
    //     [](const FPType& n) { return std::exp(n); });
    
    // FPType sum = std::accumulate(classProbabilities.begin(), classProbabilities.end(), 0);

    // std::transform(
    //     classProbabilities.begin(),
    //     classProbabilities.end(),
    //     classProbabilities.begin(),
    //     [=](const FPType& n) { return std::exp(n) / sum; });
    
    return std::distance(classProbabilities.begin(), std::max_element(classProbabilities.begin(), classProbabilities.end()));
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
ReturnType DecisionForest<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::Predict(std::vector<ThresholdType>& data) const
{
    std::map<int32_t, std::vector<ThresholdType>> predictions;
    for (auto& tree: m_trees) {
        auto prediction = tree->PredictTree(data);
        // std::cout << "Tree " << predictions.size() << " prediction : " << prediction << std::endl;
        predictions[tree->GetClassId()].push_back(prediction);
    }
    
    assert(m_reductionType == ReductionType::kAdd);

    if (m_numClasses == 0) {
        auto rawPrediction = std::accumulate(predictions[0].begin(), predictions[0].end(), m_initialValue);
        // std::cout << "Raw prediction : " << rawPrediction << std::endl;
        if (m_predictionTransform == PredictionTransformation::kIdentity)
            return rawPrediction;
        else if (m_predictionTransform == PredictionTransformation::kSigmoid)
            return sigmoid(rawPrediction);
        else
            assert(false);
        return -1;
    }
    else {
        std::vector<ThresholdType> classProbabilities(m_numClasses, 0);
        std::for_each(
            predictions.begin(),
            predictions.end(),
            [&](std::pair<const int32_t, std::vector<ThresholdType>>& pair) {
                classProbabilities[pair.first] = std::accumulate(pair.second.begin(), pair.second.end(), m_initialValue);
            });
        
        return argmax<ThresholdType, ReturnType>(classProbabilities);
    }
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
float DecisionForest<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::Predict_Float(std::vector<ThresholdType>& data) const
{
    std::map<int32_t, std::vector<float>> predictions;
    for (auto& tree: m_trees) {
        auto prediction = tree->PredictTree(data);
        // std::cout << "Tree " << predictions.size() << " prediction : " << prediction << std::endl;
        predictions[tree->GetClassId()].push_back(prediction);
    }
    
    assert(m_reductionType == ReductionType::kAdd);

    if (m_numClasses == 0) {
        auto rawPrediction = std::accumulate(predictions[0].begin(), predictions[0].end(), m_initialValue);
        if (m_predictionTransform == PredictionTransformation::kIdentity)
        return rawPrediction;
        else if (m_predictionTransform == PredictionTransformation::kSigmoid)
        return sigmoid(rawPrediction);
        else
        assert(false);
        return -1;
    }
    else {
        assert(m_predictionTransform == PredictionTransformation::kSoftMax);
        std::vector<float> classProbabilities(m_numClasses, 0);
        std::for_each(
            predictions.begin(),
            predictions.end(),
            [&](std::pair<const int32_t, std::vector<float>>& pair) {
                classProbabilities[pair.first] = std::accumulate(pair.second.begin(), pair.second.end(), m_initialValue);
            });
        
        return argmax<float, ReturnType>(classProbabilities);
    }
}

// Level Order Sorter
using LevelOrderSorterNodeType = mlir::decisionforest::DecisionTree<>::Node;

class LevelOrderTraversal {
  using QueueEntry = std::pair<int32_t, LevelOrderSorterNodeType>;
  std::vector<LevelOrderSorterNodeType> m_levelOrder;
  std::queue<QueueEntry> m_queue;
  std::map<int32_t, int32_t> m_nodeIndexMap;
  
  void DoLevelOrderTraversal(const std::vector<LevelOrderSorterNodeType>& nodes) {
    int32_t invalidIndex = DecisionTree<>::INVALID_NODE_INDEX;
    m_nodeIndexMap[invalidIndex] = invalidIndex;
    // Assume the root is the first node.
    assert (nodes[0].parent == -1);
    m_queue.push(QueueEntry(0, nodes[0]));
    while(!m_queue.empty()) {
      auto entry = m_queue.front();
      m_queue.pop();
      auto index = entry.first;
      auto& node = entry.second;
      m_levelOrder.push_back(node);
      assert (m_nodeIndexMap.find(index) == m_nodeIndexMap.end());
      m_nodeIndexMap[index] = m_levelOrder.size() - 1;
      if (node.IsLeaf())
        continue;
      if (node.leftChild != DecisionTree<>::INVALID_NODE_INDEX)
        m_queue.push(QueueEntry(node.leftChild, nodes.at(node.leftChild)));
      if (node.rightChild != DecisionTree<>::INVALID_NODE_INDEX)
        m_queue.push(QueueEntry(node.rightChild, nodes.at(node.rightChild)));
    }
  }

  int32_t GetNewIndex(int32_t oldIndex) {
    auto iter = m_nodeIndexMap.find(oldIndex);
    assert (iter != m_nodeIndexMap.end());
    return iter->second;
  }

  void RewriteIndices() {
    for (auto& node : m_levelOrder) {
      node.parent = GetNewIndex(node.parent);
      node.leftChild = GetNewIndex(node.leftChild);
      node.rightChild = GetNewIndex(node.rightChild);
    }
  }
public:
  LevelOrderTraversal(const std::vector<LevelOrderSorterNodeType>& nodes) {
    DoLevelOrderTraversal(nodes);
    RewriteIndices();
  }
  std::vector<LevelOrderSorterNodeType>& LevelOrderNodes() { return m_levelOrder; }
};

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
std::vector<ThresholdType> DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::GetSparseThresholdArray() {
    LevelOrderTraversal levelOrder(GetNodes());
    auto& sortedNodes = levelOrder.LevelOrderNodes();
    std::vector<ThresholdType> thresholdVec(sortedNodes.size());
    size_t i=0;
    for (auto& node : sortedNodes) {
        thresholdVec.at(i) = node.threshold;
        ++i;
    }
    return thresholdVec;
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
std::vector<FeatureIndexType> DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::GetSparseFeatureIndexArray() {
    LevelOrderTraversal levelOrder(GetNodes());
    auto& sortedNodes = levelOrder.LevelOrderNodes();
    std::vector<FeatureIndexType> featureIndexVec(sortedNodes.size());
    size_t i=0;
    for (auto& node : sortedNodes) {
        featureIndexVec.at(i) = node.IsLeaf() ? -1 : node.featureIndex;
        ++i;
    }
    return featureIndexVec;
}

template <typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
std::vector<int32_t> DecisionTree<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>::GetChildIndexArray() {
    LevelOrderTraversal levelOrder(GetNodes());
    auto& sortedNodes = levelOrder.LevelOrderNodes();
    std::vector<int32_t> childIndexVec(sortedNodes.size());
    size_t i=0;
    for (auto& node : sortedNodes) {
        childIndexVec.at(i) = node.leftChild;
        ++i;
    }
    return childIndexVec;
}


} // decisionforest
} // mlir
#endif // _DECISIONFOREST_H_