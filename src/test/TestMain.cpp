#include <vector>
#include <sstream>
#include "TreeTilingUtils.h"
#include "TestUtilsCommon.h"
#include "ExecutionHelpers.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "xgboostparser.h"
#include "TiledTree.h"
using namespace mlir;

namespace TreeBeard
{
namespace test
{

// Codegen tests
bool Test_LoadTileFeatureIndicesOp_DoubleInt32_TileSize1(TestArgs_t& args);
bool Test_LoadTileThresholdOp_DoubleInt32_TileSize1(TestArgs_t& args);
bool Test_LoadTileThresholdOp_Subview_DoubleInt32_TileSize1(TestArgs_t& args);
bool Test_LoadTileFeatureIndicesOp_Subview_DoubleInt32_TileSize1(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize1(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize2(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize4(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_2Trees_BatchSize1(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_2Trees_BatchSize2(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_2Trees_BatchSize4(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize1(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize2(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize4(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize1_Float(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize2_Float(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize4_Float(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_2Trees_BatchSize1_Float(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_2Trees_BatchSize2_Float(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_2Trees_BatchSize4_Float(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize1_Float(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize2_Float(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize4_Float(TestArgs_t& args);

void InitializeVectorWithRandValues(std::vector<double>& vec) {
  for(size_t i=0 ; i<vec.size() ; ++i)
    vec[i] = (double)rand()/RAND_MAX;
}

#pragma pack(push, 1)
template<typename ThresholdType, typename IndexType>
struct NumericalTileType {
  ThresholdType threshold;
  IndexType index;
  bool operator==(const NumericalTileType<ThresholdType, IndexType>& other) const {
    return threshold==other.threshold && index==other.index;
  }
};
#pragma pack(pop)

#pragma pack(push, 1)
template<typename ThresholdType, typename IndexType, int32_t VectorSize>
struct NumericalVectorTileType {
  ThresholdType threshold[VectorSize];
  IndexType index[VectorSize];
  int16_t tileShapeID;
  bool operator==(const NumericalVectorTileType<ThresholdType, IndexType, VectorSize>& other) const {
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

template<typename TileType>
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
  assert (firstTree.GetTreeDepth() == 3);
  std::vector<TileType> expectedArray{ {0.5, 2}, {0.85, -1}, {0.3, 4}, {0, -1}, {0, -1}, {0.15, -1}, {0.25, -1} };
  return expectedArray;
}

template<typename TileType>
std::vector<TileType> AddBalancedTree(mlir::decisionforest::DecisionForest<>& forest) {
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

template<typename ThresholdType, typename IndexType>
bool Test_BufferInit_RightHeavy(TestArgs_t& args) {
  using TileType = NumericalTileType<ThresholdType, IndexType>;
  auto& context = args.context;
  mlir::OpBuilder builder(&context);
  mlir::decisionforest::DecisionForest<> forest;
  auto expectedArray = AddRightHeavyTree<TileType>(forest);

  // Construct basic tree types for the tree
  // (Type resultType, const TreeTilingDescriptor& tilingDescriptor, Type thresholdType, Type featureIndexType
  auto thresholdType = TreeBeard::GetMLIRType(ThresholdType(), builder);
  auto indexType = TreeBeard::GetMLIRType(IndexType(), builder);
  mlir::decisionforest::TreeTilingDescriptor tilingDescriptor;
  auto treeType = mlir::decisionforest::TreeType::get(thresholdType, tilingDescriptor, thresholdType, indexType);
  //(Type resultType, size_t numTrees, Type rowType, ReductionType reductionType, Type treeType)
  auto forestType = mlir::decisionforest::TreeEnsembleType::get(thresholdType, 1, thresholdType /*HACK type doesn't matter for this test*/,
                                                                mlir::decisionforest::ReductionType::kAdd, treeType);
  mlir::decisionforest::PersistDecisionForest(forest, forestType);
  std::vector<TileType> serializedTree(std::pow(2, 3) - 1); //Depth of the tree is 3, so this is the size of the dense array
  // InitializeBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth, std::vector<int32_t>& treeOffsets)
  std::vector<int32_t> offsets(1, -1);
  int32_t thresholdSize = sizeof(ThresholdType)*8;
  int32_t indexSize = sizeof(IndexType)*8;
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeBuffer(serializedTree.data(), 1, thresholdSize, indexSize, offsets);
  Test_ASSERT(expectedArray == serializedTree);
  Test_ASSERT(offsets[0] == 0);
  
  std::vector<int64_t> offsetVec(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(offsetVec.data(), 1, thresholdSize, indexSize);
  Test_ASSERT(offsetVec[0] == 0);
  
  std::vector<int64_t> lengthVec(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(lengthVec.data(), 1, thresholdSize, indexSize);
  Test_ASSERT(lengthVec[0] == 7);

  mlir::decisionforest::ClearPersistedForest();

  return true;
}

bool Test_BufferInitializationWithOneTree_RightHeavy(TestArgs_t& args) {
  return Test_BufferInit_RightHeavy<double, int32_t>(args);
}

bool Test_BufferInitializationWithOneTree_RightHeavy_Int16(TestArgs_t& args) {
  return Test_BufferInit_RightHeavy<double, int16_t>(args);
}

bool Test_BufferInitializationWithOneTree_RightHeavy_Int8(TestArgs_t& args) {
  return Test_BufferInit_RightHeavy<double, int8_t>(args);
}

bool Test_BufferInitializationWithOneTree_RightHeavy_Float(TestArgs_t& args) {
  return Test_BufferInit_RightHeavy<float, int32_t>(args);
}

bool Test_BufferInitializationWithOneTree_RightHeavy_FloatInt16(TestArgs_t& args) {
  return Test_BufferInit_RightHeavy<float, int16_t>(args);
}

bool Test_BufferInitializationWithOneTree_RightHeavy_FloatInt8(TestArgs_t& args) {
  return Test_BufferInit_RightHeavy<float, int8_t>(args);
}

template<typename ThresholdType, typename IndexType>
bool Test_BufferInitialization_TwoTrees(TestArgs_t& args) {
  using TileType = NumericalTileType<ThresholdType, IndexType>;
  auto& context = args.context;
  mlir::OpBuilder builder(&context);
  mlir::decisionforest::DecisionForest<> forest;
  auto expectedArray = AddRightHeavyTree<TileType>(forest);
  auto expectedArray2 = AddLeftHeavyTree<TileType>(forest);
  expectedArray.insert(std::end(expectedArray), std::begin(expectedArray2), std::end(expectedArray2));

  // Construct basic tree types for the tree
  // (Type resultType, const TreeTilingDescriptor& tilingDescriptor, Type thresholdType, Type featureIndexType
  auto thresholdType = TreeBeard::GetMLIRType(ThresholdType(), builder);
  auto indexType = TreeBeard::GetMLIRType(IndexType(), builder);
  mlir::decisionforest::TreeTilingDescriptor tilingDescriptor;
  auto treeType = mlir::decisionforest::TreeType::get(thresholdType, tilingDescriptor, thresholdType, indexType);
  //(Type resultType, size_t numTrees, Type rowType, ReductionType reductionType, Type treeType)
  auto forestType = mlir::decisionforest::TreeEnsembleType::get(thresholdType, 1, thresholdType /*HACK type doesn't matter for this test*/,
                                                                mlir::decisionforest::ReductionType::kAdd, treeType);
  mlir::decisionforest::PersistDecisionForest(forest, forestType);
  std::vector<TileType> serializedTree(2*(std::pow(2, 3) - 1)); //Depth of the tree is 3, so this is the size of the dense array
  // InitializeBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth, std::vector<int32_t>& treeOffsets)
  int32_t thresholdSize = sizeof(ThresholdType)*8;
  int32_t indexSize = sizeof(IndexType)*8;
  std::vector<int32_t> offsets(2, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeBuffer(serializedTree.data(), 1, thresholdSize, indexSize, offsets);
  Test_ASSERT(expectedArray == serializedTree);
  Test_ASSERT(offsets[0] == 0);
  Test_ASSERT(offsets[1] == 7);

  std::vector<int64_t> offsetVec(2, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(offsetVec.data(), 1, thresholdSize, indexSize);
  Test_ASSERT(offsetVec[0] == 0);
  Test_ASSERT(offsetVec[1] == 7);

  std::vector<int64_t> lengthVec(2, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(lengthVec.data(), 1, thresholdSize, indexSize);
  Test_ASSERT(lengthVec[0] == 7);
  Test_ASSERT(lengthVec[1] == 7);

  mlir::decisionforest::ClearPersistedForest();

  return true;
}

using DoubleInt32Tile = NumericalTileType<double, int32_t>;

bool Test_BufferInitializationWithOneTree_LeftHeavy(TestArgs_t& args) {
  mlir::MLIRContext& context = args.context;
  mlir::decisionforest::DecisionForest<> forest;
  auto expectedArray = AddLeftHeavyTree<DoubleInt32Tile>(forest);  

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
  std::vector<DoubleInt32Tile> serializedTree(std::pow(2, 3) - 1); //Depth of the tree is 3, so this is the size of the dense array

  // InitializeBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth, std::vector<int32_t>& treeOffsets)
  std::vector<int32_t> offsets(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeBuffer(serializedTree.data(), 1, 64, 32, offsets);
  Test_ASSERT(expectedArray == serializedTree);
  Test_ASSERT(offsets[0] == 0);

  std::vector<int64_t> offsetVec(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(offsetVec.data(), 1, 64, 32);
  Test_ASSERT(offsetVec[0] == 0);

  std::vector<int64_t> lengthVec(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(lengthVec.data(), 1, 64, 32);
  Test_ASSERT(lengthVec[0] == 7);

  mlir::decisionforest::ClearPersistedForest();

  return true;
}

bool Test_BufferInitializationWithTwoTrees(TestArgs_t& args) {
  return Test_BufferInitialization_TwoTrees<double, int32_t>(args);
}

bool Test_BufferInitializationWithTwoTrees_Int16(TestArgs_t& args) {
  return Test_BufferInitialization_TwoTrees<double, int16_t>(args);
}

bool Test_BufferInitializationWithTwoTrees_Int8(TestArgs_t& args) {
  return Test_BufferInitialization_TwoTrees<double, int8_t>(args);
}

bool Test_BufferInitializationWithTwoTrees_Float(TestArgs_t& args) {
  return Test_BufferInitialization_TwoTrees<float, int32_t>(args);
}

bool Test_BufferInitializationWithTwoTrees_FloatInt16(TestArgs_t& args) {
  return Test_BufferInitialization_TwoTrees<float, int16_t>(args);
}

bool Test_BufferInitializationWithTwoTrees_FloatInt8(TestArgs_t& args) {
  return Test_BufferInitialization_TwoTrees<float, int8_t>(args);
}

// IR Tests
using ThresholdType = double;
using ReturnType = double;
using FeatureIndexType = int32_t;
using NodeIndexType = int32_t;
using InputElementType = double;

typedef std::vector<DoubleInt32Tile> (*ForestConstructor_t)(decisionforest::DecisionForest<>& forest);

class FixedTreeIRConstructor : public TreeBeard::ModelJSONParser<double, double, int32_t, int32_t, double> {
  std::vector<DoubleInt32Tile> m_treeSerialization;
  ForestConstructor_t m_constructForest;
public:
  FixedTreeIRConstructor(MLIRContext& context, int32_t batchSize, ForestConstructor_t constructForest)
    : TreeBeard::ModelJSONParser<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType, InputElementType>(context, batchSize), m_constructForest(constructForest)
  {  }
  void Parse() override {
    m_treeSerialization = m_constructForest(*m_forest);
    AddFeaturesToForest(*m_forest, m_treeSerialization, "float");
  }
  decisionforest::DecisionForest<>& GetForest() { return *m_forest; }
};

bool Test_ForestCodeGen_BatchSize1(TestArgs_t& args, ForestConstructor_t forestConstructor, std::vector< std::vector<double> >& inputData) {
  FixedTreeIRConstructor irConstructor(args.context, 1, forestConstructor);
  irConstructor.Parse();
  auto module = irConstructor.GetEvaluationFunction();
  // module->dump();
  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(args.context, module);
  // module->dump();
  mlir::decisionforest::LowerEnsembleToMemrefs(args.context, module);
  // module->dump();
  mlir::decisionforest::ConvertNodeTypeToIndexType(args.context, module);
  // module->dump();
  mlir::decisionforest::LowerToLLVM(args.context, module);
  // module->dump();
  // mlir::decisionforest::dumpLLVMIR(module);
  decisionforest::InferenceRunner inferenceRunner(module, 1, 64, 32);
  
  // inferenceRunner.PrintLengthsArray();
  // inferenceRunner.PrintOffsetsArray();
  
  for(auto& row : inputData) {
    double result = -1;
    inferenceRunner.RunInference<double, double>(row.data(), &result, row.size(), 1);
    double expectedResult = irConstructor.GetForest().Predict(row);
    Test_ASSERT(FPEqual(result, expectedResult));
  }
  return true;
}

bool Test_ForestCodeGen_VariableBatchSize(TestArgs_t& args, ForestConstructor_t forestConstructor, 
                                          int64_t batchSize, std::vector< std::vector<double> >& inputData) {
  FixedTreeIRConstructor irConstructor(args.context, batchSize, forestConstructor);
  irConstructor.Parse();
  auto module = irConstructor.GetEvaluationFunction();
  // module->dump();
  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(args.context, module);
  mlir::decisionforest::LowerEnsembleToMemrefs(args.context, module);
  mlir::decisionforest::ConvertNodeTypeToIndexType(args.context, module);
  // module->dump();
  mlir::decisionforest::LowerToLLVM(args.context, module);
  // module->dump();
  // mlir::decisionforest::dumpLLVMIR(module);
  decisionforest::InferenceRunner inferenceRunner(module, 1, 64, 32);
  
  // inferenceRunner.PrintLengthsArray();
  // inferenceRunner.PrintOffsetsArray();
  
  for(auto& batch : inputData) {
    assert (batch.size() % batchSize == 0);
    size_t rowSize = batch.size()/batchSize;
    std::vector<double> result(batchSize, -1);
    inferenceRunner.RunInference<double, double>(batch.data(), result.data(), batch.size()/batchSize, batchSize);
    for(int64_t rowIdx=0 ; rowIdx<batchSize ; ++rowIdx) {
      std::vector<double> row(batch.begin() + rowIdx*rowSize, batch.begin() + (rowIdx+1)*rowSize);
      double expectedResult = irConstructor.GetForest().Predict(row);
      Test_ASSERT(FPEqual(result[rowIdx], expectedResult));
    }
  }
  return true;
}

std::vector<std::vector<double>> GetBatchSize1Data() {
  std::vector<double> inputData1 = {0.1, 0.2, 0.5, 0.3, 0.25};
  std::vector<double> inputData2 = {0.1, 0.2, 0.6, 0.3, 0.25};
  std::vector<std::vector<double>> data = {inputData1, inputData2};
  return data;
}

std::vector<std::vector<double>> GetBatchSize2Data() {
  std::vector<double> inputData1 = {0.1, 0.2, 0.5, 0.3, 0.25,
                                    0.1, 0.2, 0.6, 0.3, 0.25};
  std::vector<std::vector<double>> data = { inputData1 };
  return data;
}

bool Test_CodeGeneration_LeftHeavy_BatchSize1(TestArgs_t& args) {
  auto data = GetBatchSize1Data();
  return Test_ForestCodeGen_BatchSize1(args, AddLeftHeavyTree<DoubleInt32Tile>, data);
}

bool Test_CodeGeneration_RightHeavy_BatchSize1(TestArgs_t& args) {
  auto data = GetBatchSize1Data();
  return Test_ForestCodeGen_BatchSize1(args, AddRightHeavyTree<DoubleInt32Tile>, data);
}

bool Test_CodeGeneration_RightAndLeftHeavy_BatchSize1(TestArgs_t& args) {
  auto data = GetBatchSize1Data();
  return Test_ForestCodeGen_BatchSize1(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, data);
}

bool Test_CodeGeneration_LeftHeavy_BatchSize2(TestArgs_t& args) {
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(args, AddLeftHeavyTree<DoubleInt32Tile>, 2, data);
}

bool Test_CodeGeneration_RightHeavy_BatchSize2(TestArgs_t& args) {
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(args, AddRightHeavyTree<DoubleInt32Tile>, 2, data);
}

bool Test_CodeGeneration_AddRightAndLeftHeavyTrees_BatchSize2(TestArgs_t& args) {
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, 2, data);
}

// Tests for Tiled Buffer Initialization
template<typename ThresholdType, typename IndexType>
bool Test_BufferInit_SingleTree_Tiled(TestArgs_t& args, ForestConstructor_t forestConstructor, std::vector<int32_t>& tileIDs) {
  using VectorTileType = NumericalVectorTileType<ThresholdType, IndexType, 3>;
  auto& context = args.context;
  mlir::OpBuilder builder(&context);
  mlir::decisionforest::DecisionForest<> forest;
  forestConstructor(forest);

  // Construct basic tree types for the tree
  // (Type resultType, const TreeTilingDescriptor& tilingDescriptor, Type thresholdType, Type featureIndexType
  auto thresholdType = TreeBeard::GetMLIRType(ThresholdType(), builder);
  auto indexType = TreeBeard::GetMLIRType(IndexType(), builder);
  
  int32_t tileSize = 3;
  decisionforest::TreeTilingDescriptor tilingDescriptor(tileSize /*tile size*/, 4 /*num tiles*/, tileIDs, decisionforest::TilingType::kRegular);
  forest.GetTree(0).SetTilingDescriptor(tilingDescriptor);

  auto treeType = mlir::decisionforest::TreeType::get(thresholdType, tilingDescriptor, thresholdType, indexType);
  //(Type resultType, size_t numTrees, Type rowType, ReductionType reductionType, Type treeType)
  std::vector<Type> treeTypes = {treeType};
  auto forestType = mlir::decisionforest::TreeEnsembleType::get(thresholdType, 1, thresholdType /*HACK type doesn't matter for this test*/,
                                                                mlir::decisionforest::ReductionType::kAdd, treeTypes);
  mlir::decisionforest::PersistDecisionForest(forest, forestType);

  mlir::decisionforest::TiledTree tiledTree(forest.GetTree(0));
  auto numTiles = tiledTree.GetNumberOfTiles();
  std::vector<VectorTileType> serializedTree(numTiles);
  auto thresholds = tiledTree.SerializeThresholds();
  auto featureIndices = tiledTree.SerializeFeatureIndices();
  auto tileShapeIDs = tiledTree.SerializeTileShapeIDs();

  std::vector<int32_t> offsets(1, -1);
  int32_t thresholdSize = sizeof(ThresholdType)*8;
  int32_t indexSize = sizeof(IndexType)*8;
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeBuffer(serializedTree.data(), tileSize, thresholdSize, indexSize, offsets);
  for(int32_t i=0 ; i<numTiles ; ++i) {
    for (int32_t j=0 ; j<tileSize ; ++j) {
      Test_ASSERT(FPEqual(serializedTree[i].threshold[j], thresholds[i*tileSize + j]));
      Test_ASSERT(serializedTree[i].index[j] == featureIndices[i*tileSize + j]);
    }
    std::cout << tileShapeIDs[i] << std::endl;
    Test_ASSERT(tileShapeIDs[i] == serializedTree[i].tileShapeID);
  }
  Test_ASSERT(offsets[0] == 0);
  
  std::vector<int64_t> offsetVec(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(offsetVec.data(), tileSize, thresholdSize, indexSize);
  Test_ASSERT(offsetVec[0] == 0);
  
  std::vector<int64_t> lengthVec(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(lengthVec.data(), tileSize, thresholdSize, indexSize);
  Test_ASSERT(lengthVec[0] == numTiles);

  mlir::decisionforest::ClearPersistedForest();
  std::cout << "**********\n";
  return true;
}

bool Test_BufferInitializationWithOneTree_RightHeavy_Tiled(TestArgs_t& args) {
  using TileType = NumericalTileType<double, int32_t>;
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  return Test_BufferInit_SingleTree_Tiled<double, int32_t>(args, AddRightHeavyTree<TileType>, tileIDs);
}

bool Test_BufferInitializationWithOneTree_LeftHeavy_Tiled(TestArgs_t& args) {
  using TileType = NumericalTileType<double, int32_t>;
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  return Test_BufferInit_SingleTree_Tiled<double, int32_t>(args, AddLeftHeavyTree<TileType>, tileIDs);
}

bool Test_BufferInitializationWithOneTree_Balanced_Tiled(TestArgs_t& args) {
  using TileType = NumericalTileType<double, int32_t>;
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 0, 3, 4 };
  return Test_BufferInit_SingleTree_Tiled<double, int32_t>(args, AddBalancedTree<TileType>, tileIDs);
}

bool Test_TiledTreeConstruction_LeftHeavy_Simple(TestArgs_t& args) {
  decisionforest::DecisionForest<> forest;
  AddLeftHeavyTree<DoubleInt32Tile>(forest);
  auto& tree = forest.GetTree(0);
  
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 };
  decisionforest::TreeTilingDescriptor tilingDescriptor(3, 4, tileIDs, decisionforest::TilingType::kRegular);
  tree.SetTilingDescriptor(tilingDescriptor);

  decisionforest::TiledTree tiledTree(tree);
  tiledTree.WriteDOTFile("/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/debug/tiledTree.dot");
  auto thresholds = tiledTree.SerializeThresholds();
  auto featureIndices = tiledTree.SerializeFeatureIndices();
  return true;
}

bool Test_TiledTreeConstruction_RightHeavy_Simple(TestArgs_t& args) {
  decisionforest::DecisionForest<> forest;
  AddRightHeavyTree<DoubleInt32Tile>(forest);
  auto& tree = forest.GetTree(0);
  
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 };
  decisionforest::TreeTilingDescriptor tilingDescriptor(3, 4, tileIDs, decisionforest::TilingType::kRegular);
  tree.SetTilingDescriptor(tilingDescriptor);

  decisionforest::TiledTree tiledTree(tree);
  tiledTree.WriteDOTFile("/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/debug/tiledTree.dot");
  auto thresholds = tiledTree.SerializeThresholds();
  auto featureIndices = tiledTree.SerializeFeatureIndices();
  return true;
}

bool Test_TiledTreeConstruction_Balanced_Simple(TestArgs_t& args) {
  decisionforest::DecisionForest<> forest;
  AddBalancedTree<DoubleInt32Tile>(forest);
  auto& tree = forest.GetTree(0);
  
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 0, 3, 4 };
  decisionforest::TreeTilingDescriptor tilingDescriptor(3, 5, tileIDs, decisionforest::TilingType::kRegular);
  tree.SetTilingDescriptor(tilingDescriptor);

  decisionforest::TiledTree tiledTree(tree);
  tiledTree.WriteDOTFile("/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/debug/tiledTree.dot");
  auto thresholds = tiledTree.SerializeThresholds();
  auto featureIndices = tiledTree.SerializeFeatureIndices();
  return true;
}

void TestTileStringGen() {
    mlir::decisionforest::TileShapeToTileIDMap tileMap(3);
    tileMap.ComputeTileLookUpTable();
}

TestDescriptor testList[] = {
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_LeftHeavy),
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy_Int16),
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy_Int8),
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy_Float),
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy_FloatInt16),
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy_FloatInt8),
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy),
  TEST_LIST_ENTRY(Test_BufferInitializationWithTwoTrees),
  TEST_LIST_ENTRY(Test_BufferInitializationWithTwoTrees_Int16),
  TEST_LIST_ENTRY(Test_BufferInitializationWithTwoTrees_Int8),
  TEST_LIST_ENTRY(Test_BufferInitializationWithTwoTrees_Float),
  TEST_LIST_ENTRY(Test_BufferInitializationWithTwoTrees_FloatInt16),
  TEST_LIST_ENTRY(Test_BufferInitializationWithTwoTrees_FloatInt8),
  TEST_LIST_ENTRY(Test_CodeGeneration_LeftHeavy_BatchSize1),
  TEST_LIST_ENTRY(Test_CodeGeneration_RightHeavy_BatchSize1),
  TEST_LIST_ENTRY(Test_CodeGeneration_RightAndLeftHeavy_BatchSize1),
  TEST_LIST_ENTRY(Test_CodeGeneration_LeftHeavy_BatchSize2),
  TEST_LIST_ENTRY(Test_CodeGeneration_RightHeavy_BatchSize2),
  TEST_LIST_ENTRY(Test_CodeGeneration_AddRightAndLeftHeavyTrees_BatchSize2),
  TEST_LIST_ENTRY(Test_LoadTileFeatureIndicesOp_DoubleInt32_TileSize1),
  TEST_LIST_ENTRY(Test_LoadTileThresholdOp_DoubleInt32_TileSize1),
  TEST_LIST_ENTRY(Test_LoadTileThresholdOp_Subview_DoubleInt32_TileSize1),
  TEST_LIST_ENTRY(Test_LoadTileFeatureIndicesOp_Subview_DoubleInt32_TileSize1),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_1Tree_BatchSize4),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_1Tree_BatchSize2),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_1Tree_BatchSize1),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_2Trees_BatchSize1),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_2Trees_BatchSize2),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_2Trees_BatchSize4),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_4Trees_BatchSize1),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_4Trees_BatchSize2),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_4Trees_BatchSize4),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_1Tree_BatchSize1_Float),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_1Tree_BatchSize2_Float),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_1Tree_BatchSize4_Float),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_2Trees_BatchSize1_Float),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_2Trees_BatchSize2_Float),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_2Trees_BatchSize4_Float),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_4Trees_BatchSize1_Float),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_4Trees_BatchSize2_Float),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_4Trees_BatchSize4_Float),
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy_Tiled),
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_LeftHeavy_Tiled),
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_Balanced_Tiled)
};

// TestDescriptor testList[] = {
//    TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy_Tiled),
//    TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_LeftHeavy_Tiled),
//    TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_Balanced_Tiled)
// };

const size_t numTests = sizeof(testList) / sizeof(testList[0]);

// void PrintExceptionInfo(std::exception_ptr eptr) {
// 	try {
// 		if (eptr) {
// 			std::rethrow_exception(eptr);
// 		}
// 	}
// 	catch (const std::exception& e) {
// 		std::cout << "\"" << e.what() << "\"\n";
// 	}
// }

bool RunTest(TestDescriptor test, TestArgs_t& args) {
	std::string errStr;
	std::cout << "Running test " << test.m_testName << ".... ";
	bool pass = false;
  // try
	{
		pass = test.m_testFunc(args);
	}
	// catch s(...)
	// {
	// 	std::exception_ptr eptr = std::current_exception();
	// 	std::cout << "Crashed with exception ";
	// 	PrintExceptionInfo(eptr);
	// 	pass = false;
	// }
	std::cout << (pass ? "Passed" : "Failed") << std::endl;
	return pass;
}

void RunTests() {
 	bool overallPass = true;

  std::cout << "Running Treebeard Tests " << std::endl << std::endl;

  for (size_t i = 0; i < numTests; ++i) {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::decisionforest::DecisionForestDialect>();
    context.getOrLoadDialect<mlir::StandardOpsDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    TestArgs_t args = { context };    
    bool pass = RunTest(testList[i], args);
    overallPass = overallPass && pass;
  }
  std::cout << (overallPass ? "\nTest Suite Passed" : "\nTest Suite Failed") << std::endl << std::endl;
}

} // test
} // TreeBeard