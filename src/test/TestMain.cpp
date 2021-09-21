#include <vector>
#include "xgboostparser.h"
#include "TreeTilingUtils.h"
#include "TestUtilsCommon.h"

namespace test
{
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

bool Test_BufferInitializationWithOneTree_RightHeavy(TestArgs_t& args) {
  auto& context = args.context;
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

bool Test_BufferInitializationWithOneTree_LeftHeavy(TestArgs_t& args) {
  mlir::MLIRContext& context = args.context;
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
  mlir::MLIRContext& context = args.context;
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
  Test_ASSERT(expectedArray == serializedTree);
  Test_ASSERT(offsets[0] == 0);
  Test_ASSERT(offsets[1] == 7);

  std::vector<int64_t> offsetVec(2, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(offsetVec.data(), 1, 64, 32);
  Test_ASSERT(offsetVec[0] == 0);
  Test_ASSERT(offsetVec[1] == 7);

  std::vector<int64_t> lengthVec(2, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(lengthVec.data(), 1, 64, 32);
  Test_ASSERT(lengthVec[0] == 7);
  Test_ASSERT(lengthVec[1] == 7);

  mlir::decisionforest::ClearPersistedForest();

  return true;
}

TestDescriptor testList[] = {
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_LeftHeavy),
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy),
  TEST_LIST_ENTRY(Test_BufferInitializationWithTwoTrees)
};

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
	// catch (...)
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
    TestArgs_t args = { context };    
    bool pass = RunTest(testList[i], args);			
    overallPass = overallPass && pass;
  }
  std::cout << (overallPass ? "\nTest Suite Passed" : "\nTest Suite Failed") << std::endl << std::endl;
}

} // test
