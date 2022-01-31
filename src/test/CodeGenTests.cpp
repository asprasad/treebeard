#include <vector>
#include <sstream>
#include "Dialect.h"
#include "TestUtilsCommon.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "llvm/ADT/STLExtras.h"

#include "xgboostparser.h"
#include "ExecutionHelpers.h"
#include "TreeTilingDescriptor.h"
#include "TreeTilingUtils.h"
#include "ForestTestUtils.h"

using namespace mlir;

namespace TreeBeard
{
namespace test
{

bool Test_LoadTileThresholdOp_DoubleInt32_TileSize1(TestArgs_t& args) {
  using TestTileType = NumericalTileType_Natural<double, int32_t>;
  
  auto& context = args.context;
  mlir::OpBuilder builder(&context);
  auto location = builder.getUnknownLoc();
  auto module = mlir::ModuleOp::create(location, llvm::StringRef("Test_LoadTileThresholdOp_DoubleInt32_TileSize1"));
  
  int64_t length = 5;
  int64_t shape[] = { length };
  auto tileType = decisionforest::TiledNumericalNodeType::get(builder.getF64Type(), builder.getI32Type(), 1);
  auto inputMemrefType=MemRefType::get(shape, tileType);
  auto outputMemrefType = MemRefType::get(shape, builder.getF64Type());
  auto functionType = builder.getFunctionType({inputMemrefType, outputMemrefType}, builder.getI32Type());

  auto func = builder.create<FuncOp>(location, std::string("TestFunction"), functionType, builder.getStringAttr("public"));
  auto &entryBlock = *(func.addEntryBlock());
  builder.setInsertionPointToStart(&entryBlock);

  auto lengthConstant = builder.create<arith::ConstantIndexOp>(location, length); 
  auto zeroConst = builder.create<arith::ConstantIndexOp>(location, 0);
  auto oneIndexConst = builder.create<arith::ConstantIndexOp>(location, 1);
  auto batchLoop = builder.create<scf::ForOp>(location, zeroConst, lengthConstant, oneIndexConst/*, static_cast<Value>(memrefResult)*/);

  builder.setInsertionPointToStart(batchLoop.getBody());
  auto i = batchLoop.getInductionVar();

  auto inputMemref = func.getArgument(0);
  auto outputMemref = func.getArgument(1);
  auto threshold = builder.create<decisionforest::LoadTileThresholdsOp>(location, builder.getF64Type(), static_cast<Value>(inputMemref), static_cast<Value>(i));

  builder.create<memref::StoreOp>(location, TypeRange({ }), static_cast<Value>(threshold), outputMemref, i);

  builder.setInsertionPointAfter(batchLoop);
  auto retVal = builder.create<arith::ConstantIntOp>(location, 0, builder.getI32Type());
  builder.create<mlir::ReturnOp>(location, static_cast<Value>(retVal));

  module.push_back(func);
  // module.dump();
  decisionforest::LowerToLLVM(context, module);
  // module.dump();

  auto maybeEngine = mlir::decisionforest::InferenceRunner::CreateExecutionEngine(module);
  auto& engine = maybeEngine.get();
  
  // Memref<InputElementType, 2> resultMemref;
  std::vector<TestTileType> tiles = { {1.0, 1}, {2.0, 2}, {3.0, 3}, {4.0, 4}, {5.0, 5} };
  assert (static_cast<size_t>(length) == tiles.size());
  std::vector<double> thresholds(length, -1.0);
  TestTileType *ptr = tiles.data(), *alignedPtr = tiles.data();
  int64_t offset = 0, stride = 1;
  double *resultPtr = thresholds.data(), *resultAlignedPtr = thresholds.data();
  int64_t resultLen = length;
  int32_t returnVal = -1;
  void *funcArgs[] = { &ptr, &alignedPtr, &offset, &length, &stride, // Input memref fields
                       &resultPtr, &resultAlignedPtr, &offset, &resultLen, &stride, // Result memref fields 
                       &returnVal };
  auto invocationResult = engine->invokePacked("TestFunction", funcArgs);
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }
  std::vector<double> expectedThresholds = {1.0, 2.0, 3.0, 4.0, 5.0};
  Test_ASSERT(thresholds == expectedThresholds);
  Test_ASSERT(returnVal == 0);
  return true;
}

bool Test_LoadTileFeatureIndicesOp_DoubleInt32_TileSize1(TestArgs_t& args) {
  using TestTileType = NumericalTileType_Natural<double, int32_t>;
  
  auto& context = args.context;
  mlir::OpBuilder builder(&context);
  auto location = builder.getUnknownLoc();
  auto module = mlir::ModuleOp::create(location, llvm::StringRef("Test_LoadTileFeatureIndicesOp_DoubleInt32_TileSize1"));
  
  int64_t length = 5;
  int64_t shape[] = { length };
  auto tileType = decisionforest::TiledNumericalNodeType::get(builder.getF64Type(), builder.getI32Type(), 1);
  auto inputMemrefType=MemRefType::get(shape, tileType);
  auto outputMemrefType = MemRefType::get(shape, builder.getI32Type());
  auto functionType = builder.getFunctionType({inputMemrefType, outputMemrefType}, builder.getI32Type());

  auto func = builder.create<FuncOp>(location, std::string("TestFunction"), functionType, builder.getStringAttr("public"));
  auto &entryBlock = *(func.addEntryBlock());
  builder.setInsertionPointToStart(&entryBlock);

  auto lengthConstant = builder.create<arith::ConstantIndexOp>(location, length); 
  auto zeroConst = builder.create<arith::ConstantIndexOp>(location, 0);
  auto oneIndexConst = builder.create<arith::ConstantIndexOp>(location, 1);
  auto batchLoop = builder.create<scf::ForOp>(location, zeroConst, lengthConstant, oneIndexConst/*, static_cast<Value>(memrefResult)*/);

  builder.setInsertionPointToStart(batchLoop.getBody());
  auto i = batchLoop.getInductionVar();

  auto inputMemref = func.getArgument(0);
  auto outputMemref = func.getArgument(1);
  auto featureIndex = builder.create<decisionforest::LoadTileFeatureIndicesOp>(location, builder.getI32Type(), static_cast<Value>(inputMemref), static_cast<Value>(i));

  builder.create<memref::StoreOp>(location, TypeRange({ }), static_cast<Value>(featureIndex), outputMemref, i);

  builder.setInsertionPointAfter(batchLoop);
  auto retVal = builder.create<mlir::arith::ConstantIntOp>(location, 0, builder.getI32Type());
  builder.create<mlir::ReturnOp>(location, static_cast<Value>(retVal));

  module.push_back(func);
  // module.dump();
  decisionforest::LowerToLLVM(context, module);
  // module.dump();

  auto maybeEngine = mlir::decisionforest::InferenceRunner::CreateExecutionEngine(module);
  auto& engine = maybeEngine.get();
  
  // Memref<InputElementType, 2> resultMemref;
  std::vector<TestTileType> tiles = { {0.0, 1}, {0.0, 2}, {0.0, 3}, {0.0, 4}, {0.0, 5} };
  assert (static_cast<size_t>(length) == tiles.size());
  std::vector<int32_t> indices(length, -1);
  TestTileType *ptr = tiles.data(), *alignedPtr = tiles.data();
  int64_t offset = 0, stride = 1;
  int32_t *resultPtr = indices.data(), *resultAlignedPtr = indices.data();
  int64_t resultLen = length;
  int32_t returnVal = -1;
  void *funcArgs[] = { &ptr, &alignedPtr, &offset, &length, &stride, // Input memref fields
                       &resultPtr, &resultAlignedPtr, &offset, &resultLen, &stride, // Result memref fields 
                       &returnVal };
  auto invocationResult = engine->invokePacked("TestFunction", funcArgs);
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }
  std::vector<int32_t> expectedIndices = {1, 2, 3, 4, 5};
  Test_ASSERT(indices == expectedIndices);
  Test_ASSERT(returnVal == 0);
  return true;
}

bool Test_LoadTileThresholdOp_Subview_DoubleInt32_TileSize1(TestArgs_t& args) {
  using TestTileType = NumericalTileType_Natural<double, int32_t>;
  
  auto& context = args.context;
  mlir::OpBuilder builder(&context);
  auto location = builder.getUnknownLoc();
  auto module = mlir::ModuleOp::create(location, llvm::StringRef("Test_LoadTileThresholdOp_Subview_DoubleInt32_TileSize1"));
  
  int64_t length = 5;
  int64_t shape[] = { length };
  int64_t offset = 1;
  int64_t outputShape[] = { length - offset };
  auto tileType = decisionforest::TiledNumericalNodeType::get(builder.getF64Type(), builder.getI32Type(), 1);
  auto inputMemrefType=MemRefType::get(shape, tileType);
  auto outputMemrefType = MemRefType::get(outputShape, builder.getF64Type());
  auto functionType = builder.getFunctionType({inputMemrefType, outputMemrefType}, builder.getI32Type());

  auto func = builder.create<FuncOp>(location, std::string("TestFunction"), functionType, builder.getStringAttr("public"));
  auto &entryBlock = *(func.addEntryBlock());
  builder.setInsertionPointToStart(&entryBlock);

  auto inputMemref = func.getArgument(0);
  auto outputMemref = func.getArgument(1);

  auto memrefSubview = builder.create<memref::SubViewOp>(location, inputMemref, ArrayRef<OpFoldResult>(builder.getIndexAttr(offset)), ArrayRef<OpFoldResult>(builder.getIndexAttr(length-offset)),
                                                         ArrayRef<OpFoldResult>(builder.getIndexAttr(1)));

  auto lengthConstant = builder.create<arith::ConstantIndexOp>(location, length-offset); 
  auto zeroConst = builder.create<arith::ConstantIndexOp>(location, 0);
  auto oneIndexConst = builder.create<arith::ConstantIndexOp>(location, 1);
  auto batchLoop = builder.create<scf::ForOp>(location, zeroConst, lengthConstant, oneIndexConst/*, static_cast<Value>(memrefResult)*/);

  builder.setInsertionPointToStart(batchLoop.getBody());
  auto i = batchLoop.getInductionVar();

  auto threshold = builder.create<decisionforest::LoadTileThresholdsOp>(location, builder.getF64Type(), static_cast<Value>(memrefSubview), static_cast<Value>(i));

  builder.create<memref::StoreOp>(location, TypeRange({ }), static_cast<Value>(threshold), outputMemref, i);

  builder.setInsertionPointAfter(batchLoop);
  auto retVal = builder.create<mlir::arith::ConstantIntOp>(location, 0, builder.getI32Type());
  builder.create<mlir::ReturnOp>(location, static_cast<Value>(retVal));

  module.push_back(func);
  // module.dump();
  decisionforest::LowerToLLVM(context, module);
  // module.dump();

  auto maybeEngine = mlir::decisionforest::InferenceRunner::CreateExecutionEngine(module);
  auto& engine = maybeEngine.get();
  
  // Memref<InputElementType, 2> resultMemref;
  std::vector<TestTileType> tiles = { {1.0, 1}, {2.0, 2}, {3.0, 3}, {4.0, 4}, {5.0, 5} };
  assert (static_cast<size_t>(length) == tiles.size());
  std::vector<double> thresholds(length - offset, -1.0);
  TestTileType *ptr = tiles.data(), *alignedPtr = tiles.data();
  int64_t memrefOffset = 0, stride = 1;
  double *resultPtr = thresholds.data(), *resultAlignedPtr = thresholds.data();
  int64_t resultLen = length;
  int32_t returnVal = -1;
  void *funcArgs[] = { &ptr, &alignedPtr, &memrefOffset, &length, &stride, // Input memref fields
                       &resultPtr, &resultAlignedPtr, &memrefOffset, &resultLen, &stride, // Result memref fields 
                       &returnVal };
  auto invocationResult = engine->invokePacked("TestFunction", funcArgs);
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }
  std::vector<double> expectedThresholds = {2.0, 3.0, 4.0, 5.0};
  Test_ASSERT(thresholds == expectedThresholds);
  Test_ASSERT(returnVal == 0);
  return true;
}

bool Test_LoadTileFeatureIndicesOp_Subview_DoubleInt32_TileSize1(TestArgs_t& args) {
  using TestTileType = NumericalTileType_Natural<double, int32_t>;
  
  auto& context = args.context;
  mlir::OpBuilder builder(&context);
  auto location = builder.getUnknownLoc();
  auto module = mlir::ModuleOp::create(location, llvm::StringRef("Test_LoadTileFeatureIndicesOp_Subview_DoubleInt32_TileSize1"));
  
  int64_t length = 5;
  int64_t offset = 1;
  int64_t shape[] = { length };
  int64_t outputShape[] = { length - offset };
  auto tileType = decisionforest::TiledNumericalNodeType::get(builder.getF64Type(), builder.getI32Type(), 1);
  auto inputMemrefType=MemRefType::get(shape, tileType);
  auto outputMemrefType = MemRefType::get(outputShape, builder.getI32Type());
  auto functionType = builder.getFunctionType({inputMemrefType, outputMemrefType}, builder.getI32Type());

  auto func = builder.create<FuncOp>(location, std::string("TestFunction"), functionType, builder.getStringAttr("public"));
  auto &entryBlock = *(func.addEntryBlock());
  builder.setInsertionPointToStart(&entryBlock);

  auto inputMemref = func.getArgument(0);
  auto outputMemref = func.getArgument(1);

  auto memrefSubview = builder.create<memref::SubViewOp>(location, inputMemref, ArrayRef<OpFoldResult>(builder.getIndexAttr(offset)), ArrayRef<OpFoldResult>(builder.getIndexAttr(length-offset)),
                                                         ArrayRef<OpFoldResult>(builder.getIndexAttr(1)));

  auto lengthConstant = builder.create<arith::ConstantIndexOp>(location, length-offset); 
  auto zeroConst = builder.create<arith::ConstantIndexOp>(location, 0);
  auto oneIndexConst = builder.create<arith::ConstantIndexOp>(location, 1);
  auto batchLoop = builder.create<scf::ForOp>(location, zeroConst, lengthConstant, oneIndexConst/*, static_cast<Value>(memrefResult)*/);

  builder.setInsertionPointToStart(batchLoop.getBody());
  auto i = batchLoop.getInductionVar();

  auto featureIndex = builder.create<decisionforest::LoadTileFeatureIndicesOp>(location, builder.getI32Type(), static_cast<Value>(memrefSubview), static_cast<Value>(i));

  builder.create<memref::StoreOp>(location, TypeRange({ }), static_cast<Value>(featureIndex), outputMemref, i);

  builder.setInsertionPointAfter(batchLoop);
  auto retVal = builder.create<mlir::arith::ConstantIntOp>(location, 0, builder.getI32Type());
  builder.create<mlir::ReturnOp>(location, static_cast<Value>(retVal));

  module.push_back(func);
  // module.dump();
  decisionforest::LowerToLLVM(context, module);
  // module.dump();

  auto maybeEngine = mlir::decisionforest::InferenceRunner::CreateExecutionEngine(module);
  auto& engine = maybeEngine.get();
  
  // Memref<InputElementType, 2> resultMemref;
  std::vector<TestTileType> tiles = { {0.0, 1}, {0.0, 2}, {0.0, 3}, {0.0, 4}, {0.0, 5} };
  assert (static_cast<size_t>(length) == tiles.size());
  std::vector<int32_t> indices(length - offset, -1);
  TestTileType *ptr = tiles.data(), *alignedPtr = tiles.data();
  int64_t memrefOffset = 0, stride = 1;
  int32_t *resultPtr = indices.data(), *resultAlignedPtr = indices.data();
  int64_t resultLen = length;
  int32_t returnVal = -1;
  void *funcArgs[] = { &ptr, &alignedPtr, &memrefOffset, &length, &stride, // Input memref fields
                       &resultPtr, &resultAlignedPtr, &memrefOffset, &resultLen, &stride, // Result memref fields 
                       &returnVal };
  auto invocationResult = engine->invokePacked("TestFunction", funcArgs);
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }
  std::vector<int32_t> expectedIndices = {2, 3, 4, 5};
  Test_ASSERT(indices == expectedIndices);
  Test_ASSERT(returnVal == 0);
  return true;
}


// ---------------------------------------------------
// Tests for Tiled trees
// ---------------------------------------------------

template<typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType, typename InputElementType>
class FixedTiledTreeIRConstructor : public TreeBeard::ModelJSONParser<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType, InputElementType> {
  std::vector<DoubleInt32Tile> m_treeSerialization;
  ForestConstructor_t m_constructForest;
  std::vector<decisionforest::TreeTilingDescriptor>& m_tilingDescriptors;
  int32_t m_tileShapeBitWidth;

  decisionforest::TreeEnsembleType GetEnsembleType() override {
    assert (this->m_forest->NumTrees() == m_tilingDescriptors.size());
    
    std::vector<Type> treeTypes;
    for (size_t i=0 ; i<this->m_forest->NumTrees() ; ++i) {
      auto treeType = mlir::decisionforest::TreeType::get(GetMLIRType(ReturnType(), this->m_builder), m_tilingDescriptors[i].MaxTileSize(), 
                                                          GetMLIRType(ThresholdType(), this->m_builder), 
                                                          GetMLIRType(FeatureIndexType(), this->m_builder), 
                                                          this->m_builder.getIntegerType(this->m_tileShapeBitWidth));
      treeTypes.push_back(treeType);
    }

    auto forestType = mlir::decisionforest::TreeEnsembleType::get(GetMLIRType(ReturnType(), this->m_builder),
                                                                  this->m_forest->NumTrees(), this->GetInputRowType(), 
                                                                  mlir::decisionforest::ReductionType::kAdd, treeTypes);
    return forestType;

  }
public:
  FixedTiledTreeIRConstructor(mlir::MLIRContext& context, int32_t batchSize, ForestConstructor_t constructForest,
                              std::vector<decisionforest::TreeTilingDescriptor>& tilingDescriptors, int32_t tileShapeBitWidth)
    : TreeBeard::ModelJSONParser<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType, InputElementType>(GetGlobalJSONNameForTests(), context, batchSize),
      m_constructForest(constructForest), m_tilingDescriptors(tilingDescriptors), m_tileShapeBitWidth(tileShapeBitWidth)
  {
    // Only for tiled scenarios
    assert (m_tilingDescriptors.at(0).MaxTileSize() > 1);
  }
  void Parse() override {
    m_treeSerialization = m_constructForest(*this->m_forest);
    AddFeaturesToForest(*this->m_forest, m_treeSerialization, "float");
    assert (m_tilingDescriptors.size() == this->m_forest->NumTrees());
    for (size_t i=0 ; i<this->m_forest->NumTrees() ; ++i) {
      this->m_forest->GetTree(i).SetTilingDescriptor(m_tilingDescriptors[i]);
    }
    this->m_forest->SetPredictionTransformation(decisionforest::PredictionTransformation::kIdentity);
  }

  // Add a function that takes a memref of appropriate type and copies the threshold values
  // from the model into the argument memref.
  void AddThresholdGetter() {
    int64_t length = decisionforest::ForestJSONReader::GetInstance().GetTotalNumberOfTiles();
    int64_t shape[] = { length };
    int32_t tileSize = m_tilingDescriptors.at(0).MaxTileSize();
    assert (tileSize > 1);
    
    mlir::OpBuilder builder(this->m_builder.getContext());
    auto location = builder.getUnknownLoc();
    auto tileShapeType = builder.getIntegerType(m_tileShapeBitWidth);
    auto tileType = decisionforest::TiledNumericalNodeType::get(GetMLIRType(ThresholdType(), builder), 
                                                                GetMLIRType(FeatureIndexType(), builder), tileShapeType, tileSize);
    
    auto inputMemrefType=MemRefType::get(shape, tileType);
    auto outputMemrefType = MemRefType::get(shape, GetMLIRType(ThresholdType(), builder));
    auto featureIndexMemrefType = MemRefType::get(shape, GetMLIRType(FeatureIndexType(), builder));
    auto tileShapeIDMemrefType = MemRefType::get(shape, tileShapeType);
    auto functionType = builder.getFunctionType({inputMemrefType, outputMemrefType, featureIndexMemrefType, tileShapeIDMemrefType}, builder.getI32Type());

    auto func = builder.create<FuncOp>(location, std::string("Get_ModelValues"), functionType, builder.getStringAttr("public"));
    auto &entryBlock = *(func.addEntryBlock());
    builder.setInsertionPointToStart(&entryBlock);

    auto lengthConstant = builder.create<arith::ConstantIndexOp>(location, length); 
    auto zeroConst = builder.create<arith::ConstantIndexOp>(location, 0);
    auto oneIndexConst = builder.create<arith::ConstantIndexOp>(location, 1);
    auto batchLoop = builder.create<scf::ForOp>(location, zeroConst, lengthConstant, oneIndexConst/*, static_cast<Value>(memrefResult)*/);

    builder.setInsertionPointToStart(batchLoop.getBody());
    auto i = batchLoop.getInductionVar();

    auto inputMemref = func.getArgument(0);
    auto outputMemref = func.getArgument(1);
    auto featureIndexMemref = func.getArgument(2);
    auto tileShapeMemref = func.getArgument(3);
    auto thresholdVectorType = mlir::VectorType::get({ tileSize }, GetMLIRType(ThresholdType(), this->m_builder));
    auto threshold = builder.create<decisionforest::LoadTileThresholdsOp>(location, thresholdVectorType,
                                                                          static_cast<Value>(inputMemref), static_cast<Value>(i));
    auto featureIndexVectorType = mlir::VectorType::get({ tileSize }, GetMLIRType(FeatureIndexType(), this->m_builder));
    auto index = builder.create<decisionforest::LoadTileFeatureIndicesOp>(location, featureIndexVectorType,
                                                                          static_cast<Value>(inputMemref), static_cast<Value>(i));

    auto tileShape = builder.create<decisionforest::LoadTileShapeOp>(location, tileShapeType,
                                                                     static_cast<Value>(inputMemref), static_cast<Value>(i));

    auto tileSizeConst = builder.create<arith::ConstantIndexOp>(location, tileSize);
    auto outputMemrefOffset = builder.create<mlir::arith::MulIOp>(location, builder.getIndexType(), i, tileSizeConst);
    for (int32_t j=0 ; j<tileSize ; ++j) {
      auto tileIndex = builder.create<arith::ConstantIndexOp>(location, j);
      auto outputMemrefIndex = builder.create<arith::AddIOp>(location, builder.getIndexType(), outputMemrefOffset, tileIndex);
      auto jConst = builder.create<arith::ConstantIntOp>(location, j, builder.getI32Type());
      auto element = builder.create<vector::ExtractElementOp>(location, threshold, jConst);
      builder.create<memref::StoreOp>(location, static_cast<Value>(element), static_cast<Value>(outputMemref), static_cast<Value>(outputMemrefIndex));
      auto indexElement = builder.create<vector::ExtractElementOp>(location, index, jConst);
      builder.create<memref::StoreOp>(location, static_cast<Value>(indexElement), static_cast<Value>(featureIndexMemref), static_cast<Value>(outputMemrefIndex));
    }
    builder.create<memref::StoreOp>(location, static_cast<Value>(tileShape), static_cast<Value>(tileShapeMemref), i);
    builder.setInsertionPointAfter(batchLoop);
    auto retVal = builder.create<mlir::arith::ConstantIntOp>(location, 0, builder.getI32Type());
    builder.create<mlir::ReturnOp>(location, static_cast<Value>(retVal));

    this->m_module.push_back(func);
  }
  decisionforest::DecisionForest<>& GetForest() { return *this->m_forest; }
};

// ===---------------------------------------------------=== //
// Tiled Tree Inference Tests
// ===---------------------------------------------------=== //

// Defined in TestMain.cpp
std::vector<std::vector<double>> GetBatchSize1Data();

template<typename ThresholdType=double, typename ReturnType=double, typename FeatureIndexType=int32_t,
         typename NodeIndexType=int32_t, typename InputElementType=double, typename TileShapeType=int32_t>
bool Test_TiledCodeGeneration_SingleTreeModels_BatchSize1(TestArgs_t& args, ForestConstructor_t forestConstructor, 
                                                          int32_t tileSize, const std::vector<std::vector<int32_t>>& tileIDsVec, int32_t childIndexBitWidth) {
  std::vector<decisionforest::TreeTilingDescriptor> tilingDescriptors;
  for (auto& tileIDs : tileIDsVec) {
    decisionforest::TreeTilingDescriptor tilingDescriptor(tileSize, 5, tileIDs, decisionforest::TilingType::kRegular);
    tilingDescriptors.push_back(tilingDescriptor);
  }
  
  auto tileShapeBitWidth = sizeof(TileShapeType)*8;
  FixedTiledTreeIRConstructor<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType, InputElementType>
                              irGenerator(args.context, 1, forestConstructor, tilingDescriptors, tileShapeBitWidth);
  irGenerator.Parse();
  irGenerator.SetChildIndexBitWidth(childIndexBitWidth);
  auto module = irGenerator.GetEvaluationFunction();
  decisionforest::LowerFromHighLevelToMidLevelIR(args.context, module);
  decisionforest::LowerEnsembleToMemrefs(args.context, module);
  decisionforest::ConvertNodeTypeToIndexType(args.context, module);
  // module->dump();
  decisionforest::LowerToLLVM(args.context, module);
  // module->dump();
  // decisionforest::dumpLLVMIR(module);
  decisionforest::InferenceRunner inferenceRunner(module, tileSize, sizeof(ThresholdType)*8, sizeof(FeatureIndexType)*8);
  
  auto inputData = GetBatchSize1Data();
  for(auto& row : inputData) {
    ThresholdType result = -1;
    std::vector<InputElementType> inputRow(row.begin(), row.end());
    inferenceRunner.RunInference<InputElementType, ReturnType>(inputRow.data(), &result, inputRow.size(), 1);
    ThresholdType expectedResult = irGenerator.GetForest().Predict(row);
    Test_ASSERT(FPEqual(result, expectedResult));
  }
  return true;
}

bool Test_TiledCodeGeneration_BalancedTree_BatchSize1(TestArgs_t& args) {
  auto forestConstructor = AddBalancedTree<DoubleInt32Tile>;
  int32_t childIndexBitWidth = 1;
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 0, 3, 4 };
  std::vector<int32_t> tileIDs_TileSize2 = { 0, 0, 1, 2, 5, 3, 4 };
  {
    using FPType = double;
    using IntType = int32_t;
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 2, {tileIDs_TileSize2}, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 3, { tileIDs }, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 4, { tileIDs }, childIndexBitWidth)));
  }  
  {
    using FPType = double;
    using IntType = int16_t;
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 2, { tileIDs_TileSize2 }, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 3, { tileIDs }, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 4, { tileIDs }, childIndexBitWidth)));
  }  
  {
    using FPType = double;
    using IntType = int8_t;
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 2, { tileIDs_TileSize2 }, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 3, { tileIDs }, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 4, { tileIDs }, childIndexBitWidth)));
  }
  {
    using FPType = float;
    using IntType = int32_t;
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 2, { tileIDs_TileSize2 }, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 3, { tileIDs }, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 4, { tileIDs }, childIndexBitWidth)));
  }  
  {
    using FPType = float;
    using IntType = int16_t;
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 2, { tileIDs_TileSize2 }, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 3, { tileIDs }, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 4, { tileIDs }, childIndexBitWidth)));
  }  
  {
    using FPType = float;
    using IntType = int8_t;
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 2, { tileIDs_TileSize2 }, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 3, { tileIDs }, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 4, { tileIDs }, childIndexBitWidth)));
  }
  return true;
}

template<typename TileShapeType>
bool Test_TiledCodeGeneration_ForestConstructor_BatchSize1(TestArgs_t& args, ForestConstructor_t forestConstructor, const std::vector<std::vector<int32_t>>& tileIDs, int32_t childIndexBitWidth=1) {
  {
    using FPType = double;
    using IntType = int32_t;
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType, TileShapeType>(args, forestConstructor, 2, tileIDs, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType, TileShapeType>(args, forestConstructor, 3, tileIDs, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType, TileShapeType>(args, forestConstructor, 4, tileIDs, childIndexBitWidth)));
  }  
  {
    using FPType = double;
    using IntType = int16_t;
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType, TileShapeType>(args, forestConstructor, 2, tileIDs, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType, TileShapeType>(args, forestConstructor, 3, tileIDs, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType, TileShapeType>(args, forestConstructor, 4, tileIDs, childIndexBitWidth)));
  }  
  {
    using FPType = double;
    using IntType = int8_t;
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType, TileShapeType>(args, forestConstructor, 2, tileIDs, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType, TileShapeType>(args, forestConstructor, 3, tileIDs, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType, TileShapeType>(args, forestConstructor, 4, tileIDs, childIndexBitWidth)));
  }
  {
    using FPType = float;
    using IntType = int32_t;
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType, TileShapeType>(args, forestConstructor, 2, tileIDs, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType, TileShapeType>(args, forestConstructor, 3, tileIDs, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType, TileShapeType>(args, forestConstructor, 4, tileIDs, childIndexBitWidth)));
  }  
  {
    using FPType = float;
    using IntType = int16_t;
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType, TileShapeType>(args, forestConstructor, 2, tileIDs, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType, TileShapeType>(args, forestConstructor, 3, tileIDs, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType, TileShapeType>(args, forestConstructor, 4, tileIDs, childIndexBitWidth)));
  }  
  {
    using FPType = float;
    using IntType = int8_t;
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType, TileShapeType>(args, forestConstructor, 2, tileIDs, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType, TileShapeType>(args, forestConstructor, 3, tileIDs, childIndexBitWidth)));
    Test_ASSERT((Test_TiledCodeGeneration_SingleTreeModels_BatchSize1<FPType, FPType, IntType, IntType, FPType, TileShapeType>(args, forestConstructor, 4, tileIDs, childIndexBitWidth)));
  }
  return true;
}

bool Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1(TestArgs_t& args) {
  std::vector<std::vector<int32_t>> tileIDs = { { 0, 0, 1, 2, 3 }, { 0, 0, 1, 2, 3 } };
  auto forestConstructor = AddRightAndLeftHeavyTrees<DoubleInt32Tile>;
  Test_ASSERT((Test_TiledCodeGeneration_ForestConstructor_BatchSize1<int32_t>(args, forestConstructor, tileIDs)));
  return true;
}

bool Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int8TileSize(TestArgs_t& args) {
  std::vector<std::vector<int32_t>> tileIDs = { { 0, 0, 1, 2, 3 }, { 0, 0, 1, 2, 3 } };
  auto forestConstructor = AddRightAndLeftHeavyTrees<DoubleInt32Tile>;
  Test_ASSERT((Test_TiledCodeGeneration_ForestConstructor_BatchSize1<int8_t>(args, forestConstructor, tileIDs)));
  return true;
}

bool Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int16TileSize(TestArgs_t& args) {
  std::vector<std::vector<int32_t>> tileIDs = { { 0, 0, 1, 2, 3 }, { 0, 0, 1, 2, 3 } };
  auto forestConstructor = AddRightAndLeftHeavyTrees<DoubleInt32Tile>;
  Test_ASSERT((Test_TiledCodeGeneration_ForestConstructor_BatchSize1<int16_t>(args, forestConstructor, tileIDs)));
  return true;
}

bool Test_TiledCodeGeneration_LeftHeavy_BatchSize1(TestArgs_t& args) {
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  return Test_TiledCodeGeneration_ForestConstructor_BatchSize1<int32_t>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs } );
}

bool Test_TiledCodeGeneration_RightHeavy_BatchSize1(TestArgs_t& args) {
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  return Test_TiledCodeGeneration_ForestConstructor_BatchSize1<int32_t>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs } );
}

bool Test_TiledCodeGeneration_LeftHeavy_BatchSize1_Int8TileShape(TestArgs_t& args) {
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  return Test_TiledCodeGeneration_ForestConstructor_BatchSize1<int8_t>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs } );
}

bool Test_TiledCodeGeneration_RightHeavy_BatchSize1_Int8TileShape(TestArgs_t& args) {
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  return Test_TiledCodeGeneration_ForestConstructor_BatchSize1<int8_t>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs } );
}

bool Test_TiledCodeGeneration_LeftHeavy_BatchSize1_Int16TileShape(TestArgs_t& args) {
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  return Test_TiledCodeGeneration_ForestConstructor_BatchSize1<int16_t>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs } );
}

bool Test_TiledCodeGeneration_RightHeavy_BatchSize1_Int16TileShape(TestArgs_t& args) {
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  return Test_TiledCodeGeneration_ForestConstructor_BatchSize1<int16_t>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs } );
}

// -------------------------------------------------- //
// Tiled Model Sparse Inference Tests
// -------------------------------------------------- //
bool Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  std::vector<std::vector<int32_t>> tileIDs = { { 0, 0, 1, 2, 3 }, { 0, 0, 1, 2, 3 } };
  auto forestConstructor = AddRightAndLeftHeavyTrees<DoubleInt32Tile>;
  Test_ASSERT((Test_TiledCodeGeneration_ForestConstructor_BatchSize1<int32_t>(args, forestConstructor, tileIDs, 32)));
  return true;
}

bool Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int8TileSize(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  std::vector<std::vector<int32_t>> tileIDs = { { 0, 0, 1, 2, 3 }, { 0, 0, 1, 2, 3 } };
  auto forestConstructor = AddRightAndLeftHeavyTrees<DoubleInt32Tile>;
  Test_ASSERT((Test_TiledCodeGeneration_ForestConstructor_BatchSize1<int8_t>(args, forestConstructor, tileIDs, 32)));
  return true;
}

bool Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int16TileSize(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  std::vector<std::vector<int32_t>> tileIDs = { { 0, 0, 1, 2, 3 }, { 0, 0, 1, 2, 3 } };
  auto forestConstructor = AddRightAndLeftHeavyTrees<DoubleInt32Tile>;
  Test_ASSERT((Test_TiledCodeGeneration_ForestConstructor_BatchSize1<int16_t>(args, forestConstructor, tileIDs, 32)));
  return true;
}

bool Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  return Test_TiledCodeGeneration_ForestConstructor_BatchSize1<int32_t>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs }, 32 );
}

bool Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  return Test_TiledCodeGeneration_ForestConstructor_BatchSize1<int32_t>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs }, 32 );
}

bool Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1_Int8TileShape(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  return Test_TiledCodeGeneration_ForestConstructor_BatchSize1<int8_t>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs }, 32 );
}

bool Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1_Int8TileShape(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  return Test_TiledCodeGeneration_ForestConstructor_BatchSize1<int8_t>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs }, 32 );
}

bool Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1_Int16TileShape(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  return Test_TiledCodeGeneration_ForestConstructor_BatchSize1<int16_t>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs }, 32 );
}

bool Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1_Int16TileShape(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  return Test_TiledCodeGeneration_ForestConstructor_BatchSize1<int16_t>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs }, 32 );
}

// -------------------------------------------------- //
// Tiled Model Initialization Tests
// -------------------------------------------------- //

template<typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType,
         typename InputElementType, int32_t tileSize, typename TileShapeType=int32_t>
bool Test_ModelInitialization(TestArgs_t& args, ForestConstructor_t forestConstructor, 
                              const std::vector<std::vector<int32_t>>& tileIDsVec) {
  std::vector<decisionforest::TreeTilingDescriptor> tilingDescriptors;
  for (auto& tileIDs : tileIDsVec) {
    decisionforest::TreeTilingDescriptor tilingDescriptor(tileSize, -1, tileIDs, decisionforest::TilingType::kRegular);
    tilingDescriptors.push_back(tilingDescriptor);
  }

  auto tileShapeBitWidth = sizeof(TileShapeType)*8;
  FixedTiledTreeIRConstructor<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType, InputElementType> 
                              irGenerator(args.context, 1, forestConstructor, tilingDescriptors, tileShapeBitWidth);
  irGenerator.Parse();
  auto module = irGenerator.GetEvaluationFunction();

  decisionforest::LowerFromHighLevelToMidLevelIR(args.context, module);
  decisionforest::LowerEnsembleToMemrefs(args.context, module);
  decisionforest::ConvertNodeTypeToIndexType(args.context, module);
  irGenerator.AddThresholdGetter();  
  // module->dump();
  decisionforest::LowerToLLVM(args.context, module);
  // module->dump();
  // decisionforest::dumpLLVMIR(module);
  int32_t thresholdSize = sizeof(ThresholdType)*8;
  int32_t featureIndexSize = sizeof(FeatureIndexType)*8;
  InferenceRunnerForTest inferenceRunner(module, tileSize, thresholdSize, featureIndexSize);
  
  std::vector<ThresholdType> thresholds;
  std::vector<FeatureIndexType> featureIndices;
  std::vector<TileShapeType> tileShapeIDs;
  mlir::decisionforest::ForestJSONReader::GetInstance().GetModelValues(tileSize, thresholdSize, featureIndexSize, thresholds, featureIndices, tileShapeIDs);

  decisionforest::Memref<ThresholdType, 1> modelMemref;
  {
    // Get the model memref from the MLIR module
    std::vector<void*> args = { &modelMemref };
    inferenceRunner.ExecuteFunction("Get_model", args);
  }

  std::vector<ThresholdType> thresholdsInMLIRModule(thresholds.size(), -1);
  decisionforest::Memref<ThresholdType, 1> thresholdMemref{thresholdsInMLIRModule.data(), thresholdsInMLIRModule.data(), 0, (int64_t)thresholds.size(), 1};
  
  std::vector<FeatureIndexType> featureIndicesInMLIRModule(featureIndices.size(), -1);
  decisionforest::Memref<FeatureIndexType, 1> featureIndicesMemref{featureIndicesInMLIRModule.data(), featureIndicesInMLIRModule.data(), 
                                                                   0, (int64_t)featureIndicesInMLIRModule.size(), 1};
  
  std::vector<TileShapeType> tileShapeIDsInMLIRModule(tileShapeIDs.size(), -1);
  decisionforest::Memref<TileShapeType, 1> tileShapeIDsMemref{tileShapeIDsInMLIRModule.data(), tileShapeIDsInMLIRModule.data(), 
                                                        0, (int64_t)tileShapeIDsInMLIRModule.size(), 1};

  {
    int32_t retVal = -1;
    // Get the threshold values from the model memref
    std::vector<void*> args = { &modelMemref.bufferPtr, &modelMemref.alignedPtr, &modelMemref.offset, &modelMemref.lengths[0], &modelMemref.strides[0],
                                &thresholdMemref.bufferPtr, &thresholdMemref.alignedPtr, &thresholdMemref.offset, &thresholdMemref.lengths[0], &thresholdMemref.strides[0],
                                &featureIndicesMemref.bufferPtr, &featureIndicesMemref.alignedPtr, &featureIndicesMemref.offset, &featureIndicesMemref.lengths[0], &featureIndicesMemref.strides[0],
                                &tileShapeIDsMemref.bufferPtr, &tileShapeIDsMemref.alignedPtr, &tileShapeIDsMemref.offset, &tileShapeIDsMemref.lengths[0], &tileShapeIDsMemref.strides[0],
                                &retVal };
    inferenceRunner.ExecuteFunction("Get_ModelValues", args);
    Test_ASSERT(retVal == 0);
  }
  Test_ASSERT(thresholds == thresholdsInMLIRModule);
  Test_ASSERT(featureIndicesInMLIRModule == featureIndices);
  Test_ASSERT(tileShapeIDsInMLIRModule == tileShapeIDs);
  return true;
}

bool Test_ModelInit_LeftHeavy(TestArgs_t& args) {
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  {
    using FPType = double;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
  }
  return true;
}

bool Test_ModelInit_LeftHeavy_Int8TileShape(TestArgs_t& args) {
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  using TileShapeType = int8_t;
  {
    using FPType = double;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
  }
  return true;
}

bool Test_ModelInit_LeftHeavy_Int16TileShape(TestArgs_t& args) {
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  using TileShapeType = int16_t;
  {
    using FPType = double;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4, TileShapeType>(args, AddLeftHeavyTree<DoubleInt32Tile>, { tileIDs })));
  }
  return true;
}

bool Test_ModelInit_RightHeavy(TestArgs_t& args) {
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  {
    using FPType = double;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
  }
  return true;
}

bool Test_ModelInit_RightHeavy_Int8TileShape(TestArgs_t& args) {
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  using TileShapeType = int8_t;
  {
    using FPType = double;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
  }
  return true;
}

bool Test_ModelInit_RightHeavy_Int16TileShape(TestArgs_t& args) {
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  using TileShapeType = int16_t;
  {
    using FPType = double;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4, TileShapeType>(args, AddRightHeavyTree<DoubleInt32Tile>, { tileIDs })));
  }
  return true;
}

bool Test_ModelInit_Balanced(TestArgs_t& args) {
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 0, 3, 4 };
  std::vector<int32_t> tileIDs_TileSize2 = { 0, 0, 1, 2, 5, 3, 4 };
  {
    using FPType = double;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs_TileSize2 })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs_TileSize2 })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs_TileSize2 })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs_TileSize2 })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs_TileSize2 })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs_TileSize2 })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
  }
  return true;
}

bool Test_ModelInit_Balanced_Int8TileShape(TestArgs_t& args) {
  using TileShapeType = int8_t;
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 0, 3, 4 };
  std::vector<int32_t> tileIDs_TileSize2 = { 0, 0, 1, 2, 5, 3, 4 };
  {
    using FPType = double;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs_TileSize2 })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs_TileSize2 })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs_TileSize2 })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs_TileSize2 })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs_TileSize2 })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs_TileSize2 })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
  }
  return true;
}

bool Test_ModelInit_Balanced_Int16TileShape(TestArgs_t& args) {
  using TileShapeType = int16_t;
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 0, 3, 4 };
  std::vector<int32_t> tileIDs_TileSize2 = { 0, 0, 1, 2, 5, 3, 4 };
  {
    using FPType = double;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs_TileSize2 })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs_TileSize2 })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs_TileSize2 })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs_TileSize2 })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs_TileSize2 })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs_TileSize2 })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4, TileShapeType>(args, AddBalancedTree<DoubleInt32Tile>, { tileIDs })));
  }
  return true;
}

bool Test_ModelInit_RightAndLeftHeavy(TestArgs_t& args) {
  std::vector<std::vector<int32_t>> tileIDs = { { 0, 0, 1, 2, 3 }, { 0, 0, 1, 2, 3 } }; // The root and one of its children are in one tile and all leaves are in separate tiles
  {
    using FPType = double;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
  }
  return true;
}

bool Test_ModelInit_RightAndLeftHeavy_Int8TileShape(TestArgs_t& args) {
  using TileShapeType = int8_t;
  std::vector<std::vector<int32_t>> tileIDs = { { 0, 0, 1, 2, 3 }, { 0, 0, 1, 2, 3 } }; // The root and one of its children are in one tile and all leaves are in separate tiles
  {
    using FPType = double;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
  }
  return true;
}

bool Test_ModelInit_RightAndLeftHeavy_Int16TileShape(TestArgs_t& args) {
  using TileShapeType = int16_t;
  std::vector<std::vector<int32_t>> tileIDs = { { 0, 0, 1, 2, 3 }, { 0, 0, 1, 2, 3 } }; // The root and one of its children are in one tile and all leaves are in separate tiles
  {
    using FPType = double;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
  }
  {
    using FPType = float;
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 2, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 3, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int32_t, int32_t, FPType, 4, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 2, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 3, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int16_t, int16_t, FPType, 4, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 2, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 3, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
    Test_ASSERT((Test_ModelInitialization<FPType, FPType, int8_t, int8_t, FPType, 4, TileShapeType>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, tileIDs)));
  }
  return true;
}

// --------------------------------------------------------------------------
// Uniform Tiling Tests
// --------------------------------------------------------------------------
template<typename ThresholdType=double, typename ReturnType=double, 
         typename FeatureIndexType=int32_t, typename NodeIndexType=int32_t, typename InputElementType=double>
bool Test_UniformTiling_BatchSize1(TestArgs_t& args, ForestConstructor_t forestConstructor, int32_t tileSize, int32_t tileShapeBitWidth) {
  FixedTreeIRConstructor<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType, InputElementType> irGenerator(args.context, 1, forestConstructor);
  irGenerator.Parse();
  auto module = irGenerator.GetEvaluationFunction();
  decisionforest::LowerFromHighLevelToMidLevelIR(args.context, module);
  decisionforest::DoUniformTiling(args.context, module, tileSize, tileShapeBitWidth);
  // module->dump();
  decisionforest::LowerEnsembleToMemrefs(args.context, module);
  decisionforest::ConvertNodeTypeToIndexType(args.context, module);
  // module->dump();
  decisionforest::LowerToLLVM(args.context, module);
  // module->dump();
  // decisionforest::dumpLLVMIR(module);
  decisionforest::InferenceRunner inferenceRunner(module, tileSize, sizeof(ThresholdType)*8, sizeof(FeatureIndexType)*8);
  
  auto inputData = GetBatchSize1Data();
  for(auto& row : inputData) {
    ThresholdType result = -1;
    std::vector<InputElementType> inputRow(row.begin(), row.end());
    inferenceRunner.RunInference<InputElementType, ReturnType>(inputRow.data(), &result, inputRow.size(), 1);
    ThresholdType expectedResult = irGenerator.GetForest().Predict(row);
    Test_ASSERT(FPEqual(result, expectedResult));
  }
  return true;
}

bool Test_UniformTiling_BatchSize1_AllTypes(TestArgs_t& args, ForestConstructor_t forestConstructor, int32_t tileShapeBitWidth) {
  {
    using FPType = double;
    using IntType = int32_t;
    Test_ASSERT((Test_UniformTiling_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 2, tileShapeBitWidth)));
    Test_ASSERT((Test_UniformTiling_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 3, tileShapeBitWidth)));
    Test_ASSERT((Test_UniformTiling_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 4, tileShapeBitWidth)));
  }  
  {
    using FPType = double;
    using IntType = int16_t;
    Test_ASSERT((Test_UniformTiling_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 2, tileShapeBitWidth)));
    Test_ASSERT((Test_UniformTiling_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 3, tileShapeBitWidth)));
    Test_ASSERT((Test_UniformTiling_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 4, tileShapeBitWidth)));
  }  
  {
    using FPType = double;
    using IntType = int8_t;
    Test_ASSERT((Test_UniformTiling_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 2, tileShapeBitWidth)));
    Test_ASSERT((Test_UniformTiling_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 3, tileShapeBitWidth)));
    Test_ASSERT((Test_UniformTiling_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 4, tileShapeBitWidth)));
  }
  {
    using FPType = float;
    using IntType = int32_t;
    Test_ASSERT((Test_UniformTiling_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 2, tileShapeBitWidth)));
    Test_ASSERT((Test_UniformTiling_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 3, tileShapeBitWidth)));
    Test_ASSERT((Test_UniformTiling_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 4, tileShapeBitWidth)));
  }  
  {
    using FPType = float;
    using IntType = int16_t;
    Test_ASSERT((Test_UniformTiling_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 2, tileShapeBitWidth)));
    Test_ASSERT((Test_UniformTiling_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 3, tileShapeBitWidth)));
    Test_ASSERT((Test_UniformTiling_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 4, tileShapeBitWidth)));
  }  
  {
    using FPType = float;
    using IntType = int8_t;
    Test_ASSERT((Test_UniformTiling_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 2, tileShapeBitWidth)));
    Test_ASSERT((Test_UniformTiling_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 3, tileShapeBitWidth)));
    Test_ASSERT((Test_UniformTiling_BatchSize1<FPType, FPType, IntType, IntType, FPType>(args, forestConstructor, 4, tileShapeBitWidth)));
  }
  return true;
}

bool Test_UniformTiling_LeftHeavy_BatchSize1(TestArgs_t &args) {
  return Test_UniformTiling_BatchSize1_AllTypes(args, AddLeftHeavyTree<DoubleInt32Tile>, 32);
}

bool Test_UniformTiling_RightHeavy_BatchSize1(TestArgs_t &args) {
  return Test_UniformTiling_BatchSize1_AllTypes(args, AddRightHeavyTree<DoubleInt32Tile>, 32);
}

bool Test_UniformTiling_Balanced_BatchSize1(TestArgs_t &args) {
  return Test_UniformTiling_BatchSize1_AllTypes(args, AddBalancedTree<DoubleInt32Tile>, 32);
}

bool Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1(TestArgs_t &args) {
  return Test_UniformTiling_BatchSize1_AllTypes(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, 32);
}

bool Test_UniformTiling_LeftHeavy_BatchSize1_Int8TileShape(TestArgs_t &args) {
  return Test_UniformTiling_BatchSize1_AllTypes(args, AddLeftHeavyTree<DoubleInt32Tile>, 8);
}

bool Test_UniformTiling_RightHeavy_BatchSize1_Int8TileShape(TestArgs_t &args) {
  return Test_UniformTiling_BatchSize1_AllTypes(args, AddRightHeavyTree<DoubleInt32Tile>, 8);
}

bool Test_UniformTiling_Balanced_BatchSize1_Int8TileShape(TestArgs_t &args) {
  return Test_UniformTiling_BatchSize1_AllTypes(args, AddBalancedTree<DoubleInt32Tile>, 8);
}

bool Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1_Int8TileShape(TestArgs_t &args) {
  return Test_UniformTiling_BatchSize1_AllTypes(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, 8);
}

bool Test_UniformTiling_LeftHeavy_BatchSize1_Int16TileShape(TestArgs_t &args) {
  return Test_UniformTiling_BatchSize1_AllTypes(args, AddLeftHeavyTree<DoubleInt32Tile>, 16);
}

bool Test_UniformTiling_RightHeavy_BatchSize1_Int16TileShape(TestArgs_t &args) {
  return Test_UniformTiling_BatchSize1_AllTypes(args, AddRightHeavyTree<DoubleInt32Tile>, 16);
}

bool Test_UniformTiling_Balanced_BatchSize1_Int16TileShape(TestArgs_t &args) {
  return Test_UniformTiling_BatchSize1_AllTypes(args, AddBalancedTree<DoubleInt32Tile>, 16);
}

bool Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1_Int16TileShape(TestArgs_t &args) {
  return Test_UniformTiling_BatchSize1_AllTypes(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, 16);
}

} // test
} // TreeBeard