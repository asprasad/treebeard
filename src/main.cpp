#include <iostream>
#include "json/xgboostparser.h"
#include "include/TreeTilingUtils.h"

// #include "mlir/Dialect.h"

// #include "mlir/DecisionTreeAttributes.h"
// #include "mlir/DecisionTreeTypes.h"

using namespace std;

namespace mlir
{
namespace decisionforest
{
void LowerFromHighLevelToMidLevelIR(mlir::MLIRContext& context, mlir::ModuleOp module);
void LowerEnsembleToMemrefs(mlir::MLIRContext& context, mlir::ModuleOp module);
}
}

#pragma pack(push, 1)
struct TileType {
  double threshold;
  int32_t index;
  bool operator==(const TileType& other) const {
    return threshold==other.threshold && index==other.index;
  }
};
#pragma pack(pop)

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

std::vector<TileType> AddRightHeavyTree(mlir::decisionforest::DecisionForest<>& forest) {
  // Add tree one
  auto& firstTree = forest.NewTree();
  auto rootNode = firstTree.NewNode(0.5, 2);
  // Add right child
  {
    auto node = firstTree.NewNode(0.3, 4);
    firstTree.SetNodeParent(node, rootNode);
    firstTree.SetNodeRightChild(rootNode, node);
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
  // Add left child (leaf)
  {
    auto node = firstTree.NewNode(0.8, -1);
    firstTree.SetNodeParent(node, rootNode);
    firstTree.SetNodeLeftChild(rootNode, node);
  }
  assert (firstTree.GetTreeDepth() == 3);
  std::vector<TileType> expectedArray{ {0.5, 2}, {0.8, -1}, {0.3, 4}, {0, -1}, {0, -1}, {0.1, -1}, {0.2, -1} };
  return expectedArray;
}

bool Test_BufferInitializationWithOneTree_RightHeavy(mlir::MLIRContext& context) {
  mlir::decisionforest::DecisionForest<> forest;
  auto expectedArray = AddRightHeavyTree(forest);

  // Construct basic tree types for the tree
  // (Type resultType, const TreeTilingDescriptor& tilingDescriptor, Type thresholdType, Type featureIndexType
  auto doubleType = mlir::Float64Type::get(&context);
  auto int32Type = mlir::IntegerType::get(&context, 32);
  mlir::decisionforest::TreeTilingDescriptor tilingDescriptor;
  auto treeType = mlir::decisionforest::TreeType::get(doubleType, tilingDescriptor, doubleType, int32Type);
  //(Type resultType, size_t numTrees, Type rowType, ReductionType reductionType, Type treeType)
  auto forestType = mlir::decisionforest::TreeEnsembleType::get(doubleType, 1, doubleType /*HACK type doesn't matter for this test*/,
                                                                mlir::decisionforest::ReductionType::kAdd, treeType);
  mlir::decisionforest::PersistDecisionForest(forest, forestType);
  std::vector<TileType> serializedTree(std::pow(2, 3) - 1); //Depth of the tree is 3, so this is the size of the dense array
  // InitializeBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth, std::vector<int32_t>& treeOffsets)
  std::vector<int32_t> offsets(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeBuffer(serializedTree.data(), 1, 64, 32, offsets);
  assert(expectedArray == serializedTree);
  assert(offsets[0] == 0);

  mlir::decisionforest::ClearPersistedForest();

  return true;
}

bool Test_BufferInitializationWithOneTree_LeftHeavy(mlir::MLIRContext& context) {
  mlir::decisionforest::DecisionForest<> forest;
  auto expectedArray = AddLeftHeavyTree(forest);  

  // Construct basic tree types for the tree
  // (Type resultType, const TreeTilingDescriptor& tilingDescriptor, Type thresholdType, Type featureIndexType
  auto doubleType = mlir::Float64Type::get(&context);
  auto int32Type = mlir::IntegerType::get(&context, 32);
  mlir::decisionforest::TreeTilingDescriptor tilingDescriptor;
  auto treeType = mlir::decisionforest::TreeType::get(doubleType, tilingDescriptor, doubleType, int32Type);
  //(Type resultType, size_t numTrees, Type rowType, ReductionType reductionType, Type treeType)
  auto forestType = mlir::decisionforest::TreeEnsembleType::get(doubleType, 1, doubleType /*HACK type doesn't matter for this test*/,
                                                                mlir::decisionforest::ReductionType::kAdd, treeType);
  mlir::decisionforest::PersistDecisionForest(forest, forestType);
  std::vector<TileType> serializedTree(std::pow(2, 3) - 1); //Depth of the tree is 3, so this is the size of the dense array

  // InitializeBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth, std::vector<int32_t>& treeOffsets)
  std::vector<int32_t> offsets(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeBuffer(serializedTree.data(), 1, 64, 32, offsets);
  assert(expectedArray == serializedTree);
  assert(offsets[0] == 0);

  mlir::decisionforest::ClearPersistedForest();

  return true;
}

bool Test_BufferInitializationWithTwoTrees(mlir::MLIRContext& context) {
  mlir::decisionforest::DecisionForest<> forest;
  auto expectedArray = AddRightHeavyTree(forest);
  auto expectedArray2 = AddLeftHeavyTree(forest);
  expectedArray.insert(std::end(expectedArray), std::begin(expectedArray2), std::end(expectedArray2));

  // Construct basic tree types for the tree
  // (Type resultType, const TreeTilingDescriptor& tilingDescriptor, Type thresholdType, Type featureIndexType
  auto doubleType = mlir::Float64Type::get(&context);
  auto int32Type = mlir::IntegerType::get(&context, 32);
  mlir::decisionforest::TreeTilingDescriptor tilingDescriptor;
  auto treeType = mlir::decisionforest::TreeType::get(doubleType, tilingDescriptor, doubleType, int32Type);
  //(Type resultType, size_t numTrees, Type rowType, ReductionType reductionType, Type treeType)
  auto forestType = mlir::decisionforest::TreeEnsembleType::get(doubleType, 2, doubleType /*HACK type doesn't matter for this test*/,
                                                                mlir::decisionforest::ReductionType::kAdd, treeType);
  mlir::decisionforest::PersistDecisionForest(forest, forestType);
  std::vector<TileType> serializedTree(2*(std::pow(2, 3) - 1)); //Depth of the tree is 3, so this is the size of the dense array
  // InitializeBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth, std::vector<int32_t>& treeOffsets)
  std::vector<int32_t> offsets(2, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeBuffer(serializedTree.data(), 1, 64, 32, offsets);
  assert(expectedArray == serializedTree);
  assert(offsets[0] == 0);
  assert(offsets[1] == 7);

  mlir::decisionforest::ClearPersistedForest();

  return true;
}

void RunForestSerializationTests(mlir::MLIRContext& context) {
  Test_BufferInitializationWithOneTree_LeftHeavy(context);
  Test_BufferInitializationWithOneTree_RightHeavy(context);
  Test_BufferInitializationWithTwoTrees(context);
}

int main(int argc, char *argv[]) {
  cout << "Tree-heavy: A compiler for gradient boosting tree inference.\n";
  
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::decisionforest::DecisionForestDialect>();
  context.getOrLoadDialect<mlir::StandardOpsDialect>();

  RunForestSerializationTests(context);

  const int32_t batchSize = 16;
  TreeHeavy::XGBoostJSONParser<> xgBoostParser(context, argv[1], batchSize);
  xgBoostParser.Parse();
  auto module = xgBoostParser.GetEvaluationFunction();
  // module->dump();

  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(context, module);

  // module->dump();

  mlir::decisionforest::LowerEnsembleToMemrefs(context, module);
  module->dump();

  std::vector<double> data(8);
  auto decisionForest = xgBoostParser.GetForest();
  cout << "Ensemble prediction: " << decisionForest->Predict(data) << endl;
}