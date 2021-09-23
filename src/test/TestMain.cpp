#include <vector>
#include <sstream>
#include "xgboostparser.h"
#include "TreeTilingUtils.h"
#include "TestUtilsCommon.h"
#include "ExecutionHelpers.h"

#include "mlir/Dialect/SCF/SCF.h"

using namespace mlir;

namespace test
{

// Codegen tests
bool Test_LoadTileFeatureIndicesOp_DoubleInt32_TileSize1(TestArgs_t& args);
bool Test_LoadTileThresholdOp_DoubleInt32_TileSize1(TestArgs_t& args);

void InitializeVectorWithRandValues(std::vector<double>& vec) {
  for(size_t i=0 ; i<vec.size() ; ++i)
    vec[i] = (double)rand()/RAND_MAX;
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

std::vector<TileType> AddRightAndLeftHeavyTrees(decisionforest::DecisionForest<>& forest) {
  auto expectedArray = AddRightHeavyTree(forest);
  auto expectedArray2 = AddLeftHeavyTree(forest);
  expectedArray.insert(std::end(expectedArray), std::begin(expectedArray2), std::end(expectedArray2));
  return expectedArray;
}

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

// IR Tests
using ThresholdType = double;
using ReturnType = double;
using FeatureIndexType = int32_t;
using NodeIndexType = int32_t;
typedef std::vector<TileType> (*ForestConstructor_t)(decisionforest::DecisionForest<>& forest);

class FixedTreeIRConstructor : public TreeBeard::ModelJSONParser<double, double, int32_t, int32_t> {
  std::vector<TileType> m_treeSerialization;
  ForestConstructor_t m_constructForest;
public:
  FixedTreeIRConstructor(MLIRContext& context, int32_t batchSize, ForestConstructor_t constructForest)
    : TreeBeard::ModelJSONParser<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>(context, batchSize), m_constructForest(constructForest)
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
  mlir::decisionforest::LowerEnsembleToMemrefs(args.context, module);
  mlir::decisionforest::ConvertNodeTypeToIndexType(args.context, module);
  module->dump();
  mlir::decisionforest::LowerToLLVM(args.context, module);
  module->dump();
  mlir::decisionforest::dumpLLVMIR(module);
  decisionforest::InferenceRunner inferenceRunner(module);
  
  inferenceRunner.PrintLengthsArray();
  inferenceRunner.PrintOffsetsArray();
  
  for(auto& row : inputData) {
    double result[] = {-1, -1};
    inferenceRunner.RunInference<double, 5, 1, double>(row.data(), result);
    double expectedResult = irConstructor.GetForest().Predict(row);
    // Test_ASSERT(FPEqual(result, expectedResult));
  }
  return true;
}

std::vector<std::vector<double>> GetBatchSize1Data() {
  std::vector<double> inputData1 = {0.1, 0.2, 0.5, 0.3, 0.25};
  std::vector<double> inputData2 = {0.1, 0.2, 0.6, 0.3, 0.25};
  std::vector<std::vector<double>> data = {inputData1, inputData2};
  return data;
}

bool Test_CodeGeneration_LeftHeavy_BatchSize1(TestArgs_t& args) {
  auto data = GetBatchSize1Data();
  return Test_ForestCodeGen_BatchSize1(args, AddLeftHeavyTree, data);
}

bool Test_CodeGeneration_RightHeavy_BatchSize1(TestArgs_t& args) {
  auto data = GetBatchSize1Data();
  return Test_ForestCodeGen_BatchSize1(args, AddRightHeavyTree, data);
}

bool Test_CodeGeneration_RightAndLeftHeavy_BatchSize1(TestArgs_t& args) {
  auto data = GetBatchSize1Data();
  return Test_ForestCodeGen_BatchSize1(args, AddRightAndLeftHeavyTrees, data);
}

// TestDescriptor testList[] = {
//   TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_LeftHeavy),
//   TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy),
//   TEST_LIST_ENTRY(Test_BufferInitializationWithTwoTrees),
//   TEST_LIST_ENTRY(Test_CodeGeneration_LeftHeavy_BatchSize1),
//   TEST_LIST_ENTRY(Test_CodeGeneration_RightHeavy_BatchSize1),
//   TEST_LIST_ENTRY(Test_CodeGeneration_RightAndLeftHeavy_BatchSize1),
//   TEST_LIST_ENTRY(Test_LoadTileFeatureIndicesOp_DoubleInt32_TileSize1),
//   TEST_LIST_ENTRY(Test_LoadTileThresholdOp_DoubleInt32_TileSize1)
// };

TestDescriptor testList[] = {
  TEST_LIST_ENTRY(Test_CodeGeneration_RightAndLeftHeavy_BatchSize1)
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
