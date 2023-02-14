#ifndef _FORESTTESTUTILS_H_
#define _FORESTTESTUTILS_H_

using namespace mlir;

// Some utlities that are used by the tests
#pragma pack(push, 1)
template<typename ThresholdType, typename IndexType>
struct NumericalTileType_Packed {
  ThresholdType threshold;
  IndexType index;
  bool operator==(const NumericalTileType_Packed<ThresholdType, IndexType>& other) const {
    return threshold==other.threshold && index==other.index;
  }
};
#pragma pack(pop)

template<typename ThresholdType, typename IndexType>
struct NumericalTileType_Natural {
  ThresholdType threshold;
  IndexType index;
  bool operator==(const NumericalTileType_Natural<ThresholdType, IndexType>& other) const {
    return threshold==other.threshold && index==other.index;
  }
};

#pragma pack(push, 1)
template<typename ThresholdType, typename IndexType, int32_t VectorSize>
struct NumericalVectorTileType_Packed {
  ThresholdType threshold[VectorSize];
  IndexType index[VectorSize];
  int32_t tileShapeID;
  bool operator==(const NumericalVectorTileType_Packed<ThresholdType, IndexType, VectorSize>& other) const {
    for (int32_t i=0; i<VectorSize ; ++i)
      if (threshold[i]!=other.threshold[i] || index[i]!=other.index[i])
        return false;
    return tileShapeID==other.tileShapeID;
  }
};
#pragma pack(pop)


template<typename TileType>
std::vector<TileType> AddLeftHeavyTree(mlir::decisionforest::DecisionForest<>& forest) {
  // Add tree one
  auto& firstTree = forest.NewTree();
  auto rootNode = firstTree.NewNode(0.5, 2);
  // Add left child
  {
    auto node = firstTree.NewNode(0.3, 4);
    firstTree.SetNodeParent(node, rootNode);
    firstTree.SetNodeLeftChild(rootNode, node);
    {
      // Leaf
      auto subTreeRoot = node;
      auto leftChild = firstTree.NewNode(0.1, -1);
      firstTree.SetNodeParent(leftChild, subTreeRoot);
      firstTree.SetNodeLeftChild(subTreeRoot, leftChild);
      
      // Leaf
      auto rightChild = firstTree.NewNode(0.2, -1);
      firstTree.SetNodeParent(rightChild, subTreeRoot);
      firstTree.SetNodeRightChild(subTreeRoot, rightChild);
    }
  }
  // Add right child (leaf)
  {
    auto node = firstTree.NewNode(0.8, -1);
    firstTree.SetNodeParent(node, rootNode);
    firstTree.SetNodeRightChild(rootNode, node);
  }
  assert (firstTree.GetTreeDepth() == 3);
  std::vector<TileType> expectedArray{ {0.5, 2}, {0.3, 4}, {0.8, -1}, {0.1, -1}, {0.2, -1}, {0, -1}, {0, -1}};
  return expectedArray;
}

inline void InitializeRightHeavyTree(decisionforest::DecisionTree<>& firstTree) {
  auto rootNode = firstTree.NewNode(0.5, 2);
  // Add right child
  {
    auto node = firstTree.NewNode(0.3, 4);
    firstTree.SetNodeParent(node, rootNode);
    firstTree.SetNodeRightChild(rootNode, node);
    {
      // Leaf
      auto subTreeRoot = node;
      auto leftChild = firstTree.NewNode(0.15, -1);
      firstTree.SetNodeParent(leftChild, subTreeRoot);
      firstTree.SetNodeLeftChild(subTreeRoot, leftChild);
      
      // Leaf
      auto rightChild = firstTree.NewNode(0.25, -1);
      firstTree.SetNodeParent(rightChild, subTreeRoot);
      firstTree.SetNodeRightChild(subTreeRoot, rightChild);
    }
  }
  // Add left child (leaf)
  {
    auto node = firstTree.NewNode(0.85, -1);
    firstTree.SetNodeParent(node, rootNode);
    firstTree.SetNodeLeftChild(rootNode, node);
  }
}

template<typename TileType>
std::vector<TileType> AddRightHeavyTree(mlir::decisionforest::DecisionForest<>& forest) {
  // Add tree one
  auto& firstTree = forest.NewTree();
  InitializeRightHeavyTree(firstTree);
  assert (firstTree.GetTreeDepth() == 3);
  std::vector<TileType> expectedArray{ {0.5, 2}, {0.85, -1}, {0.3, 4}, {0, -1}, {0, -1}, {0.15, -1}, {0.25, -1} };
  return expectedArray;
}

inline void InitializeBalancedTree(decisionforest::DecisionTree<>& firstTree) {
  auto rootNode = firstTree.NewNode(0.5, 2);
  // Add right child
  {
    auto node = firstTree.NewNode(0.3, 4);
    firstTree.SetNodeParent(node, rootNode);
    firstTree.SetNodeRightChild(rootNode, node);
    {
      // Leaf
      auto subTreeRoot = node;
      auto leftChild = firstTree.NewNode(0.15, -1);
      firstTree.SetNodeParent(leftChild, subTreeRoot);
      firstTree.SetNodeLeftChild(subTreeRoot, leftChild);
      
      // Leaf
      auto rightChild = firstTree.NewNode(0.25, -1);
      firstTree.SetNodeParent(rightChild, subTreeRoot);
      firstTree.SetNodeRightChild(subTreeRoot, rightChild);
    }
  }
  // Add left child
  {
    auto node = firstTree.NewNode(0.1, 1);
    firstTree.SetNodeParent(node, rootNode);
    firstTree.SetNodeLeftChild(rootNode, node);
    {
      // Leaf
      auto subTreeRoot = node;
      auto leftChild = firstTree.NewNode(0.75, -1);
      firstTree.SetNodeParent(leftChild, subTreeRoot);
      firstTree.SetNodeLeftChild(subTreeRoot, leftChild);
      
      // Leaf
      auto rightChild = firstTree.NewNode(0.85, -1);
      firstTree.SetNodeParent(rightChild, subTreeRoot);
      firstTree.SetNodeRightChild(subTreeRoot, rightChild);
    }
  }
}

template<typename TileType>
std::vector<TileType> AddBalancedTree(mlir::decisionforest::DecisionForest<>& forest) {
  // Add tree one
  auto& firstTree = forest.NewTree();
  InitializeBalancedTree(firstTree);
  assert (firstTree.GetTreeDepth() == 3);
  std::vector<TileType> expectedArray{ {0.5, 2}, {0.1, 1}, {0.3, 4}, {0.75, -1}, {0.85, -1}, {0.15, -1}, {0.25, -1} };
  return expectedArray;
}

template<typename TileType>
std::vector<TileType> AddRightAndLeftHeavyTrees(decisionforest::DecisionForest<>& forest) {
  auto expectedArray = AddRightHeavyTree<TileType>(forest);
  auto expectedArray2 = AddLeftHeavyTree<TileType>(forest);
  expectedArray.insert(std::end(expectedArray), std::begin(expectedArray2), std::end(expectedArray2));
  return expectedArray;
}

template<typename TileType>
void AddFeaturesToForest(decisionforest::DecisionForest<>& forest, std::vector<TileType>& serializedForest, std::string featureType) {
  int32_t numFeatures = -1;
  for (auto& tile : serializedForest) {
    if (tile.index > numFeatures)
      numFeatures = tile.index;
  }
  numFeatures++; // Index --> Length
  for (int32_t i=0 ; i<numFeatures ; ++i) {
    std::stringstream strStream;
    strStream << "x_" << i;
    forest.AddFeature(strStream.str(), featureType);
  }
}

using DoubleInt32Tile = NumericalTileType_Packed<double, int32_t>;
typedef std::vector<DoubleInt32Tile> (*ForestConstructor_t)(decisionforest::DecisionForest<>& forest);

template<typename ThresholdType=double, typename ReturnType=double, 
         typename FeatureIndexType=int32_t, typename NodeIndexType=int32_t, typename InputElementType=double>
class FixedTreeIRConstructor : public TreeBeard::ModelJSONParser<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType, InputElementType> {
  std::vector<DoubleInt32Tile> m_treeSerialization;
  ForestConstructor_t m_constructForest;
public:
  FixedTreeIRConstructor(MLIRContext& context,
                         std::shared_ptr<decisionforest::IModelSerializer> serializer,
                         int32_t batchSize,
                         ForestConstructor_t constructForest)
    : TreeBeard::ModelJSONParser<ThresholdType,
                                 ReturnType,
                                 FeatureIndexType,
                                 NodeIndexType,
                                 InputElementType>(TreeBeard::test::GetGlobalJSONNameForTests(), 
                                                   serializer,
                                                   context,
                                                   batchSize),
                                 m_constructForest(constructForest)
  {  }

  void Parse() override {
    m_treeSerialization = m_constructForest(*this->m_forest);
    this->m_forest->SetPredictionTransformation(decisionforest::PredictionTransformation::kIdentity);
    AddFeaturesToForest(*this->m_forest, m_treeSerialization, "float");
  }
  
  decisionforest::DecisionForest<>& GetForest() { return *this->m_forest; }
};

template <typename BaseClass>
class InferenceRunnerForTestTemplate : public BaseClass {
public:
  using BaseClass::BaseClass;

  int32_t ExecuteFunction(const std::string& funcName, std::vector<void*>& args) {
    auto& engine = this->m_engine;
    auto invocationResult = engine->invokePacked(funcName, args);
    if (invocationResult) {
      llvm::errs() << "JIT invocation failed\n";
      assert (false);
      return -1;
    }
    return 0;
  }
};

using InferenceRunnerForTest = InferenceRunnerForTestTemplate<decisionforest::InferenceRunner>;

template<typename T>
decisionforest::Memref<T, 1> VectorToMemref(std::vector<T>& vec) {
  return decisionforest::Memref<T, 1>{vec.data(), vec.data(), 0, {static_cast<int64_t>(vec.size())}, {1}};
}

namespace TreeBeard
{
namespace test
{

std::vector<std::vector<double>> GetBatchSize1Data();
std::vector<std::vector<double>> GetBatchSize2Data();

} // namespace test
} // namespace Treebeard

#endif // _FORESTTESTUTILS_H_