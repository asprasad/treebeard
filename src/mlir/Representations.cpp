#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "Representations.h"
#include "OpLoweringUtils.h"
#include "LIRLoweringHelpers.h"
#include "Logger.h"

using namespace mlir::decisionforest::helpers;

// Maps an ensemble constant operation to a model memref and an offsets memref
std::map<mlir::Operation*, EnsembleConstantLoweringInfo> ensembleConstantToMemrefsMap;
// Maps a GetTree operation to a memref that represents the tree once the ensemble constant has been replaced
std::map<mlir::Operation*, mlir::Value> getTreeOperationMap;

// Maps an ensemble constant operation to a model memref and an offsets memref
std::map<mlir::Operation*, SparseEnsembleConstantLoweringInfo> sparseEnsembleConstantToMemrefsMap;
// Maps a GetTree operation to a memref that represents the tree once the ensemble constant has been replaced
std::map<mlir::Operation*, GetTreeLoweringInfo> sparseGetTreeOperationMap;

namespace mlir
{
namespace decisionforest
{
// ===---------------------------------------------------=== //
// Array based representation
// ===---------------------------------------------------=== //

mlir::LogicalResult ArrayBasedRepresentation::GenerateModelGlobals(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter,
                                                    std::shared_ptr<decisionforest::IModelSerializer> serializer) {

    mlir::decisionforest::EnsembleConstantOp ensembleConstOp = llvm::dyn_cast<mlir::decisionforest::EnsembleConstantOp>(op);
    assert(ensembleConstOp);
    assert(operands.size() == 0);
    if (!ensembleConstOp)
        return mlir::failure();
    
    auto location = op->getLoc();
    auto owningModule = op->getParentOfType<mlir::ModuleOp>();
    assert (owningModule);
    
    // TODO the names of the model and offset global should be generated so they're unique for each ensemble constant
    // TODO the getter function names need to be persisted with the actual tree values in the JSON so the runtime can call them. 
    std::string modelMemrefName = "model";
    std::string offsetMemrefName = "offsets";
    std::string lengthMemrefName = "lengths";
    std::string classInfoMemrefName = "treeClassInfo";


    auto memrefTypes = AddGlobalMemrefs(
      owningModule,
      ensembleConstOp,
      rewriter,
      location,
      modelMemrefName,
      offsetMemrefName,
      lengthMemrefName,
      classInfoMemrefName,
      serializer);


    AddModelMemrefInitFunction(owningModule, modelMemrefName, memrefTypes.model.cast<MemRefType>(), rewriter, location);
    auto getModelGlobal = rewriter.create<memref::GetGlobalOp>(location, memrefTypes.model, modelMemrefName);
    auto getOffsetGlobal = rewriter.create<memref::GetGlobalOp>(location, memrefTypes.offset, offsetMemrefName);
    auto getLengthGlobal = rewriter.create<memref::GetGlobalOp>(location, memrefTypes.offset, lengthMemrefName);
    auto classInfoGlobal = rewriter.create<memref::GetGlobalOp>(location, memrefTypes.classInfo, classInfoMemrefName);

    Type lookUpTableMemrefType;
    Value getLUT;

    EnsembleConstantLoweringInfo info 
    {
      static_cast<Value>(getModelGlobal),
      static_cast<Value>(getOffsetGlobal),
      static_cast<Value>(getLengthGlobal),
      getLUT,
      static_cast<Value>(classInfoGlobal),
      memrefTypes.model,
      memrefTypes.offset,
      memrefTypes.offset,
      lookUpTableMemrefType,
      memrefTypes.classInfo,
    };
    ensembleConstantToMemrefsMap[op] = info;
    return mlir::success();
}

GlobalMemrefTypes ArrayBasedRepresentation::AddGlobalMemrefs(
  mlir::ModuleOp module,
  mlir::decisionforest::EnsembleConstantOp& ensembleConstOp,
  ConversionPatternRewriter &rewriter,
  Location location,
  const std::string& modelMemrefName,
  const std::string& offsetMemrefName,
  const std::string& lengthMemrefName,
  const std::string& treeInfo,
  std::shared_ptr<decisionforest::IModelSerializer> serializer) {
  mlir::decisionforest::DecisionForestAttribute forestAttribute = ensembleConstOp.forest();
  mlir::decisionforest::DecisionForest<>& forest = forestAttribute.GetDecisionForest();

  SaveAndRestoreInsertionPoint saveAndRestoreInsertPoint(rewriter);
  rewriter.setInsertionPoint(&module.front());

  auto forestType = ensembleConstOp.getResult().getType().cast<decisionforest::TreeEnsembleType>();
  assert (forestType.doAllTreesHaveSameTileSize()); // There is still an assumption here that all trees have the same tile size
  auto treeType = forestType.getTreeType(0).cast<decisionforest::TreeType>();

  auto thresholdType = treeType.getThresholdType();
  auto featureIndexType = treeType.getFeatureIndexType(); 
  auto tileSize = treeType.getTileSize();
  auto tileShapeType = treeType.getTileShapeType();
  // assert (tileSize == 1);
  Type memrefElementType = decisionforest::TiledNumericalNodeType::get(thresholdType, featureIndexType, tileShapeType, tileSize);

  serializer->Persist(forest, forestType);
  
  auto modelMemrefSize = decisionforest::GetTotalNumberOfTiles();
  auto modelMemrefType = MemRefType::get({modelMemrefSize}, memrefElementType);
  rewriter.create<memref::GlobalOp>(location, modelMemrefName,
                                    /*sym_visibility=*/rewriter.getStringAttr("private"),
                                    /*type=*/modelMemrefType,
                                    /*initial_value=*/rewriter.getUnitAttr(),
                                    /*constant=*/false, IntegerAttr());
  AddGlobalMemrefGetter(module, modelMemrefName, modelMemrefType, rewriter, location);
  
  auto offsetSize = (int32_t)forest.NumTrees();
  auto offsetMemrefType = MemRefType::get({offsetSize}, rewriter.getIndexType());
  rewriter.create<memref::GlobalOp>(location, offsetMemrefName, rewriter.getStringAttr("private"),
                                    offsetMemrefType, rewriter.getUnitAttr(), false, IntegerAttr());
  AddGlobalMemrefGetter(module, offsetMemrefName, offsetMemrefType, rewriter, location);
  
  rewriter.create<memref::GlobalOp>(location, lengthMemrefName, rewriter.getStringAttr("private"),
                                    offsetMemrefType, rewriter.getUnitAttr(), false, IntegerAttr());
  AddGlobalMemrefGetter(module, lengthMemrefName, offsetMemrefType, rewriter, location);

  auto classInfoSize = forest.IsMultiClassClassifier() ? offsetSize : 0;
  auto classInfoMemrefType = MemRefType::get({classInfoSize}, treeType.getResultType());
  rewriter.create<memref::GlobalOp>(
    location,
    treeInfo,
    rewriter.getStringAttr("public"),
    classInfoMemrefType,
    rewriter.getUnitAttr(),
    false,
    IntegerAttr());
    AddGlobalMemrefGetter(module, treeInfo, classInfoMemrefType, rewriter, location);

    return GlobalMemrefTypes { modelMemrefType, offsetMemrefType, classInfoMemrefType };
}

void ArrayBasedRepresentation::AddModelMemrefInitFunction(mlir::ModuleOp module, std::string globalName, MemRefType memrefType, 
                                                          ConversionPatternRewriter &rewriter, Location location) {
  assert (memrefType.getShape().size() == 1);
  SaveAndRestoreInsertionPoint saveAndRestoreEntryPoint(rewriter);
  auto modelMemrefElementType = memrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
  int32_t tileSize = modelMemrefElementType.getTileSize();
  auto thresholdArgType = MemRefType::get({ memrefType.getShape()[0] * tileSize }, modelMemrefElementType.getThresholdElementType());
  auto indexArgType = MemRefType::get({ memrefType.getShape()[0] * tileSize }, modelMemrefElementType.getIndexElementType());
  auto tileShapeIDArgType = MemRefType::get(memrefType.getShape(), modelMemrefElementType.getTileShapeType());
  auto getMemrefFuncType = rewriter.getFunctionType(TypeRange{thresholdArgType, indexArgType, tileShapeIDArgType}, rewriter.getI32Type());
  std::string funcName = "Init_" + globalName;
  NamedAttribute visibilityAttribute{module.getSymVisibilityAttrName(), rewriter.getStringAttr("public")};
  auto initModelMemrefFunc = mlir::func::FuncOp::create(location, funcName, getMemrefFuncType, ArrayRef<NamedAttribute>(visibilityAttribute));
  auto &entryBlock = *initModelMemrefFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(&entryBlock);

  // for tileIndex = 0 : len
  auto getGlobalMemref = rewriter.create<memref::GetGlobalOp>(location, memrefType, globalName);
  auto zeroIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
  auto oneIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 1);
  auto lenIndexConst = rewriter.create<arith::ConstantIndexOp>(location, memrefType.getShape()[0]);
  auto forLoop = rewriter.create<scf::ForOp>(location, zeroIndexConst, lenIndexConst, oneIndexConst);
  auto tileIndex = forLoop.getInductionVar();
  rewriter.setInsertionPointToStart(forLoop.getBody());

  // index = tileSize * tileIndex
  auto tileSizeConst = rewriter.create<arith::ConstantIndexOp>(location, tileSize);
  auto tileSizeTimesi = rewriter.create<arith::MulIOp>(location, tileIndex, tileSizeConst);
  
  if (tileSize > 1) {
    auto thresholdVec = CreateZeroVectorFPConst(rewriter, location, modelMemrefElementType.getThresholdElementType(), tileSize);
    auto indexVec = CreateZeroVectorIntConst(rewriter, location, modelMemrefElementType.getIndexElementType(), tileSize);

    // Load from index to index + (tileSize - 1) into a vector
    for (int32_t j = 0 ; j<tileSize ; ++j) {
      auto offset = rewriter.create<arith::ConstantIndexOp>(location, j);
      auto index =  rewriter.create<arith::AddIOp>(location, tileSizeTimesi, offset);
      auto thresholdVal = rewriter.create<memref::LoadOp>(location, entryBlock.getArgument(0), static_cast<Value>(index));
      auto jConst = rewriter.create<arith::ConstantIntOp>(location, j, rewriter.getI32Type());
      thresholdVec = rewriter.create<vector::InsertElementOp>(location, thresholdVal, thresholdVec, jConst);
      auto indexVal = rewriter.create<memref::LoadOp>(location, entryBlock.getArgument(1), static_cast<Value>(index));
      indexVec = rewriter.create<vector::InsertElementOp>(location, indexVal, indexVec, jConst);
    }
    auto tileShapeID = rewriter.create<memref::LoadOp>(location, entryBlock.getArgument(2), tileIndex);
    rewriter.create<decisionforest::InitTileOp>(location, getGlobalMemref, tileIndex, thresholdVec, indexVec, tileShapeID);
  }
  else {
    // Load from index to index + (tileSize - 1) into a vector
    auto thresholdVal = rewriter.create<memref::LoadOp>(location, entryBlock.getArgument(0), static_cast<Value>(tileIndex));
    auto indexVal = rewriter.create<memref::LoadOp>(location, entryBlock.getArgument(1), static_cast<Value>(tileIndex));
    // TODO check how tileShapeID vector is created when tileSize = 1
    auto tileShapeID = rewriter.create<arith::ConstantIntOp>(location, 0, rewriter.getI32Type());
    rewriter.create<decisionforest::InitTileOp>(location, getGlobalMemref, tileIndex, thresholdVal, indexVal, tileShapeID);
  }
  rewriter.setInsertionPointAfter(forLoop);
  
  auto modelSize = rewriter.create<decisionforest::GetModelMemrefSizeOp>(location, rewriter.getI32Type(), getGlobalMemref, lenIndexConst);
  rewriter.create<mlir::func::ReturnOp>(location, static_cast<Value>(modelSize));
  module.push_back(initModelMemrefFunc);
}

mlir::Value ArrayBasedRepresentation::GetTreeMemref(mlir::Value treeValue) {
  auto getTreeOp = treeValue.getDefiningOp();
  AssertOpIsOfType<mlir::decisionforest::GetTreeFromEnsembleOp>(getTreeOp);
  auto getTreeOperationMapIter = getTreeOperationMap.find(getTreeOp);
  assert(getTreeOperationMapIter != getTreeOperationMap.end());
  auto treeMemref = getTreeOperationMapIter->second;
  return treeMemref;
}

mlir::Value ArrayBasedRepresentation::GenerateMoveToChild(mlir::Location location, ConversionPatternRewriter &rewriter, mlir::Value nodeIndex,
                                                          mlir::Value childNumber, int32_t tileSize, std::vector<mlir::Value>& extraLoads) {
  auto oneConstant = rewriter.create<arith::ConstantIndexOp>(location, 1);
  auto tileSizeConstant = rewriter.create<arith::ConstantIndexOp>(location, tileSize+1);
  auto tileSizeTimesIndex = rewriter.create<arith::MulIOp>(location, rewriter.getIndexType(), static_cast<Value>(nodeIndex), static_cast<Value>(tileSizeConstant));
  auto tileSizeTimesIndexPlus1 = rewriter.create<arith::AddIOp>(location, rewriter.getIndexType(), static_cast<Value>(tileSizeTimesIndex), static_cast<Value>(oneConstant));
  
  auto newIndex = rewriter.create<arith::AddIOp>(location, rewriter.getIndexType(), tileSizeTimesIndexPlus1, childNumber);
  return newIndex;
}

void ArrayBasedRepresentation::GenerateTreeMemref(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex) {
    // Create a subview of the model memref corresponding to this ensemble with the index equal to offsetMemref[treeIndex]
    auto location = op->getLoc();
    Operation* ensembleConstOp = ensemble.getDefiningOp();
    AssertOpIsOfType<mlir::decisionforest::EnsembleConstantOp>(ensembleConstOp);
    
    auto mapIter = ensembleConstantToMemrefsMap.find(ensembleConstOp);
    assert (mapIter != ensembleConstantToMemrefsMap.end());
    auto& ensembleInfo = mapIter->second;

    auto modelMemrefIndex = rewriter.create<memref::LoadOp>(location, ensembleInfo.offsetGlobal, treeIndex);
    auto treeLength = rewriter.create<memref::LoadOp>(location, ensembleInfo.lengthGlobal, treeIndex);; // TODO Need to put this into the map too
    auto treeMemref = rewriter.create<memref::SubViewOp>(location, ensembleInfo.modelGlobal, ArrayRef<OpFoldResult>({static_cast<Value>(modelMemrefIndex)}),
                                                         ArrayRef<OpFoldResult>({static_cast<Value>(treeLength)}), ArrayRef<OpFoldResult>({rewriter.getIndexAttr(1)}));
    
    // if (decisionforest::InsertDebugHelpers) {
    //   rewriter.create<decisionforest::PrintTreeToDOTFileOp>(location, treeMemref, treeIndex);
    // }
    getTreeOperationMap[op] = static_cast<Value>(treeMemref);
}

mlir::Value ArrayBasedRepresentation::GenerateGetTreeClassId(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex) {
  Operation* ensembleConstOp = ensemble.getDefiningOp();
  AssertOpIsOfType<mlir::decisionforest::EnsembleConstantOp>(ensembleConstOp);
  
  auto mapIter = ensembleConstantToMemrefsMap.find(ensembleConstOp);
  assert (mapIter != ensembleConstantToMemrefsMap.end());
  auto& ensembleInfo = mapIter->second;

  auto treeClassMemref = ensembleInfo.classInfoGlobal;
  auto treeClassMemrefType = treeClassMemref.getType().cast<mlir::MemRefType>();

  auto classId = rewriter.create<memref::LoadOp>(op->getLoc(), treeClassMemrefType.getElementType(), treeClassMemref, treeIndex);
  return classId;
}

mlir::Value ArrayBasedRepresentation::GenerateGetLeafValueOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, 
                                                             mlir::Value nodeIndex) {
  auto location = op->getLoc();

  auto treeMemref = this->GetTreeMemref(treeValue);
  auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
  assert (treeMemrefType);

  auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
  auto thresholdType = treeTileType.getThresholdFieldType();

  // Load threshold
  // TODO Ideally, this should be a different op for when we deal with tile sizes != 1. We will then need to load 
  // a single threshold value and cast it the trees return type
  auto loadThresholdOp = rewriter.create<decisionforest::LoadTileThresholdsOp>(location, thresholdType, treeMemref, static_cast<Value>(nodeIndex));
  Value leafValue = loadThresholdOp;
  
  if (treeTileType.getTileSize() != 1) {
    if (decisionforest::InsertDebugHelpers) {
      InsertPrintVectorOp(rewriter, location, 0, treeTileType.getThresholdElementType().getIntOrFloatBitWidth(), treeTileType.getTileSize(), loadThresholdOp);
    }
    auto zeroConst = rewriter.create<arith::ConstantIntOp>(location, int64_t(0), rewriter.getI32Type());
    auto extractElement = rewriter.create<vector::ExtractElementOp>(location, static_cast<Value>(loadThresholdOp), zeroConst);
    leafValue = extractElement;
  }
  return leafValue;
}

mlir::Value ArrayBasedRepresentation::GenerateIsLeafOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, mlir::Value nodeIndex) {
  auto location = op->getLoc();
  auto treeMemref = this->GetTreeMemref(treeValue);
  auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
  assert (treeMemrefType);

  auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
  auto featureIndexType = treeTileType.getIndexFieldType();
  auto loadFeatureIndexOp = rewriter.create<decisionforest::LoadTileFeatureIndicesOp>(location, featureIndexType, treeMemref, static_cast<Value>(nodeIndex));    
  
  Value featureIndexValue;
  if (treeTileType.getTileSize() == 1) {
    featureIndexValue = loadFeatureIndexOp;
  }
  else {
    auto indexVectorType = featureIndexType.cast<mlir::VectorType>();
    assert (indexVectorType);
    auto zeroConst = rewriter.create<arith::ConstantIntOp>(location, int64_t(0), rewriter.getI32Type());
    auto extractFirstElement = rewriter.create<vector::ExtractElementOp>(location, static_cast<Value>(loadFeatureIndexOp), zeroConst);
    featureIndexValue = extractFirstElement;
  }
  auto minusOneConstant = rewriter.create<arith::ConstantIntOp>(location, int64_t(-1), treeTileType.getIndexElementType());
  auto comparison = rewriter.create<arith::CmpIOp>(location, mlir::arith::CmpIPredicate::eq, featureIndexValue, static_cast<Value>(minusOneConstant));
  
  if (decisionforest::InsertDebugHelpers) {
    Value outcome = rewriter.create<mlir::arith::ExtUIOp>(location, rewriter.getI32Type(), static_cast<Value>(comparison));
    rewriter.create<decisionforest::PrintIsLeafOp>(location, nodeIndex, featureIndexValue, outcome);
  }
  return static_cast<Value>(comparison);
}

mlir::Value ArrayBasedRepresentation::GenerateIsLeafTileOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, mlir::Value nodeIndex) {
  return this->GenerateIsLeafOp(rewriter, op, treeValue, nodeIndex);
}

// ===---------------------------------------------------=== //
// Sparse representation
// ===---------------------------------------------------=== //

mlir::LogicalResult SparseRepresentation::GenerateModelGlobals(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter,
                                                               std::shared_ptr<decisionforest::IModelSerializer> m_serializer) {
    mlir::decisionforest::EnsembleConstantOp ensembleConstOp = llvm::dyn_cast<mlir::decisionforest::EnsembleConstantOp>(op);
    assert(ensembleConstOp);
    assert(operands.size() == 0);
    if (!ensembleConstOp)
        return mlir::failure();
    
    auto location = op->getLoc();
    auto owningModule = op->getParentOfType<mlir::ModuleOp>();
    assert (owningModule);
    
    // TODO the names of the model and offset global should be generated so they're unique for each ensemble constant
    // TODO the getter function names need to be persisted with the actual tree values in the JSON so the runtime can call them. 
    std::string modelMemrefName = "model";
    std::string offsetMemrefName = "offsets";
    std::string lengthMemrefName = "lengths";
    std::string leavesMemrefName = "leaves";
    std::string leavesLengthMemrefName = "leavesLengths";
    std::string leavesOffsetMemrefName = "leavesOffsets";
    std::string classInfoMemrefName = "treeClassInfo";

    auto memrefTypes = AddGlobalMemrefs(owningModule, ensembleConstOp, rewriter, location, modelMemrefName, offsetMemrefName, lengthMemrefName, 
                                        leavesMemrefName, leavesLengthMemrefName, leavesOffsetMemrefName, classInfoMemrefName, m_serializer);
    AddModelMemrefInitFunction(owningModule, modelMemrefName, std::get<0>(memrefTypes).cast<MemRefType>(), rewriter, location);
    
    // Add getters for all the globals we've created
    auto getModelGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<0>(memrefTypes), modelMemrefName);
    auto getOffsetGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<1>(memrefTypes), offsetMemrefName);
    auto getLengthGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<1>(memrefTypes), lengthMemrefName);
    auto getLeavesGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<2>(memrefTypes), leavesMemrefName);
    auto getLeavesOffsetGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<1>(memrefTypes), leavesOffsetMemrefName);
    auto getLeavesLengthGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<1>(memrefTypes), leavesLengthMemrefName);
    auto classInfoGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<3>(memrefTypes), classInfoMemrefName);
    
    Type lookUpTableMemrefType;
    Value getLUT;

    SparseEnsembleConstantLoweringInfo info {static_cast<Value>(getModelGlobal), static_cast<Value>(getOffsetGlobal), 
                                       static_cast<Value>(getLengthGlobal), getLUT,
                                       getLeavesGlobal, getLeavesOffsetGlobal, getLeavesLengthGlobal, classInfoGlobal,
                                       std::get<0>(memrefTypes), std::get<1>(memrefTypes), std::get<1>(memrefTypes), 
                                       lookUpTableMemrefType, std::get<2>(memrefTypes), std::get<3>(memrefTypes)};
    sparseEnsembleConstantToMemrefsMap[op] = info;
    return mlir::success();
}

std::tuple<Type, Type, Type, Type> SparseRepresentation::AddGlobalMemrefs(mlir::ModuleOp module, mlir::decisionforest::EnsembleConstantOp& ensembleConstOp,
                                        ConversionPatternRewriter &rewriter, Location location,
                                        const std::string& modelMemrefName, const std::string& offsetMemrefName, const std::string& lengthMemrefName,
                                        const std::string& leavesMemrefName, const std::string& leavesLengthMemrefName,
                                        const std::string& leavesOffsetMemrefName, const std::string& treeInfo,
                                        std::shared_ptr<IModelSerializer> serializer) {
  mlir::decisionforest::DecisionForestAttribute forestAttribute = ensembleConstOp.forest();
  mlir::decisionforest::DecisionForest<>& forest = forestAttribute.GetDecisionForest();

  SaveAndRestoreInsertionPoint saveAndRestoreInsertPoint(rewriter);
  rewriter.setInsertionPoint(&module.front());

  auto forestType = ensembleConstOp.getResult().getType().cast<decisionforest::TreeEnsembleType>();
  assert (forestType.doAllTreesHaveSameTileSize()); // There is still an assumption here that all trees have the same tile size
  auto treeType = forestType.getTreeType(0).cast<decisionforest::TreeType>();

  auto thresholdType = treeType.getThresholdType();
  auto featureIndexType = treeType.getFeatureIndexType(); 
  auto tileSize = treeType.getTileSize();
  auto tileShapeType = treeType.getTileShapeType();
  auto childIndexType = treeType.getChildIndexType();
  // assert (tileSize == 1);
  Type memrefElementType = decisionforest::TiledNumericalNodeType::get(thresholdType, featureIndexType, tileShapeType, 
                                                                        tileSize, childIndexType);

  serializer->Persist(forest, forestType);
  
  auto modelMemrefSize = decisionforest::GetTotalNumberOfTiles();
  auto modelMemrefType = MemRefType::get({modelMemrefSize}, memrefElementType);
  rewriter.create<memref::GlobalOp>(location, modelMemrefName,
                                    /*sym_visibility=*/rewriter.getStringAttr("private"),
                                    /*type=*/modelMemrefType,
                                    /*initial_value=*/rewriter.getUnitAttr(),
                                    /*constant=*/false, IntegerAttr());
  AddGlobalMemrefGetter(module, modelMemrefName, modelMemrefType, rewriter, location);
  
  // Create offset memref
  auto offsetSize = (int32_t)forest.NumTrees();
  auto offsetMemrefType = MemRefType::get({offsetSize}, rewriter.getIndexType());
  rewriter.create<memref::GlobalOp>(location, offsetMemrefName, rewriter.getStringAttr("private"),
                                    offsetMemrefType, rewriter.getUnitAttr(), false, IntegerAttr());
  AddGlobalMemrefGetter(module, offsetMemrefName, offsetMemrefType, rewriter, location);
  
  // Create length memref
  rewriter.create<memref::GlobalOp>(location, lengthMemrefName, rewriter.getStringAttr("private"),
                                    offsetMemrefType, rewriter.getUnitAttr(), false, IntegerAttr());
  AddGlobalMemrefGetter(module, lengthMemrefName, offsetMemrefType, rewriter, location);

  auto leavesMemrefSize = decisionforest::GetTotalNumberOfLeaves();
  auto leavesMemrefType = MemRefType::get({leavesMemrefSize}, thresholdType);
  rewriter.create<memref::GlobalOp>(location, leavesMemrefName, rewriter.getStringAttr("private"),
                                    leavesMemrefType, rewriter.getUnitAttr(), false, IntegerAttr());
  AddGlobalMemrefGetter(module, leavesMemrefName, leavesMemrefType, rewriter, location);
  
  if (TreeBeard::Logging::loggingOptions.logGenCodeStats)
      TreeBeard::Logging::Log("Leaves memref size : " + std::to_string(leavesMemrefSize * (thresholdType.getIntOrFloatBitWidth()/8)));

  // Create leaf offset memref
  rewriter.create<memref::GlobalOp>(location, leavesOffsetMemrefName, rewriter.getStringAttr("private"),
                                    offsetMemrefType, rewriter.getUnitAttr(), false, IntegerAttr());
  AddGlobalMemrefGetter(module, leavesOffsetMemrefName, offsetMemrefType, rewriter, location);

  // Create leaf length memref
  rewriter.create<memref::GlobalOp>(location, leavesLengthMemrefName, rewriter.getStringAttr("private"),
                                    offsetMemrefType, rewriter.getUnitAttr(), false, IntegerAttr());
  AddGlobalMemrefGetter(module, leavesLengthMemrefName, offsetMemrefType, rewriter, location);

  auto classInfoSize = forest.IsMultiClassClassifier() ? offsetSize : 0;
  auto classInfoMemrefType = MemRefType::get({classInfoSize}, treeType.getResultType());
  rewriter.create<memref::GlobalOp>(
    location,
    treeInfo,
    rewriter.getStringAttr("public"),
    classInfoMemrefType,
    rewriter.getUnitAttr(),
    false,
    IntegerAttr());
  AddGlobalMemrefGetter(module, treeInfo, classInfoMemrefType, rewriter, location);

  return std::make_tuple(modelMemrefType, offsetMemrefType, leavesMemrefType, classInfoMemrefType);
}

void SparseRepresentation::AddModelMemrefInitFunction(mlir::ModuleOp module, std::string globalName, MemRefType memrefType, 
                                ConversionPatternRewriter &rewriter, Location location) {
  assert (memrefType.getShape().size() == 1);
  SaveAndRestoreInsertionPoint saveAndRestoreEntryPoint(rewriter);
  auto modelMemrefElementType = memrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
  int32_t tileSize = modelMemrefElementType.getTileSize();
  auto thresholdArgType = MemRefType::get({ memrefType.getShape()[0] * tileSize }, modelMemrefElementType.getThresholdElementType());
  auto indexArgType = MemRefType::get({ memrefType.getShape()[0] * tileSize }, modelMemrefElementType.getIndexElementType());
  auto tileShapeIDArgType = MemRefType::get(memrefType.getShape(), modelMemrefElementType.getTileShapeType());
  auto childrenIndexArgType = MemRefType::get(memrefType.getShape(), modelMemrefElementType.getChildIndexType());
  mlir::FunctionType initMemrefFuncType;
  initMemrefFuncType = rewriter.getFunctionType(TypeRange{thresholdArgType, 
                                                          indexArgType, tileShapeIDArgType, 
                                                          childrenIndexArgType},
                                                rewriter.getI32Type());
  std::string funcName = "Init_" + globalName;
  NamedAttribute visibilityAttribute{module.getSymVisibilityAttrName(), rewriter.getStringAttr("public")};
  auto initModelMemrefFunc = mlir::func::FuncOp::create(location, funcName, initMemrefFuncType, ArrayRef<NamedAttribute>(visibilityAttribute));
  auto &entryBlock = *initModelMemrefFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(&entryBlock);

  // for tileIndex = 0 : len
  auto getGlobalMemref = rewriter.create<memref::GetGlobalOp>(location, memrefType, globalName);
  auto zeroIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
  auto oneIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 1);
  auto lenIndexConst = rewriter.create<arith::ConstantIndexOp>(location, memrefType.getShape()[0]);
  auto forLoop = rewriter.create<scf::ForOp>(location, zeroIndexConst, lenIndexConst, oneIndexConst);
  auto tileIndex = forLoop.getInductionVar();
  rewriter.setInsertionPointToStart(forLoop.getBody());

  // index = tileSize * tileIndex
  auto tileSizeConst = rewriter.create<arith::ConstantIndexOp>(location, tileSize);
  auto tileSizeTimesi = rewriter.create<arith::MulIOp>(location, tileIndex, tileSizeConst);
  
  if (tileSize > 1) {
    auto thresholdVec = CreateZeroVectorFPConst(rewriter, location, modelMemrefElementType.getThresholdElementType(), tileSize);
    auto indexVec = CreateZeroVectorIntConst(rewriter, location, modelMemrefElementType.getIndexElementType(), tileSize);

    // Load from index to index + (tileSize - 1) into a vector
    for (int32_t j = 0 ; j<tileSize ; ++j) {
      auto offset = rewriter.create<arith::ConstantIndexOp>(location, j);
      auto index =  rewriter.create<arith::AddIOp>(location, tileSizeTimesi, offset);
      auto thresholdVal = rewriter.create<memref::LoadOp>(location, entryBlock.getArgument(0), static_cast<Value>(index));
      auto jConst = rewriter.create<arith::ConstantIntOp>(location, j, rewriter.getI32Type());
      thresholdVec = rewriter.create<vector::InsertElementOp>(location, thresholdVal, thresholdVec, jConst);
      auto indexVal = rewriter.create<memref::LoadOp>(location, entryBlock.getArgument(1), static_cast<Value>(index));
      indexVec = rewriter.create<vector::InsertElementOp>(location, indexVal, indexVec, jConst);
    }
    auto tileShapeID = rewriter.create<memref::LoadOp>(location, entryBlock.getArgument(2), tileIndex);
    auto childIndex = rewriter.create<memref::LoadOp>(location, entryBlock.getArgument(3), tileIndex);
    rewriter.create<decisionforest::InitSparseTileOp>(location, getGlobalMemref, tileIndex, thresholdVec, indexVec, tileShapeID, childIndex);
  }
  else {
    // Load from index to index + (tileSize - 1) into a vector
    auto thresholdVal = rewriter.create<memref::LoadOp>(location, entryBlock.getArgument(0), static_cast<Value>(tileIndex));
    auto indexVal = rewriter.create<memref::LoadOp>(location, entryBlock.getArgument(1), static_cast<Value>(tileIndex));
    auto childIndex = rewriter.create<memref::LoadOp>(location, entryBlock.getArgument(3), static_cast<Value>(tileIndex));
    // TODO check how tileShapeID vector is created when tileSize = 1
    auto tileShapeID = rewriter.create<arith::ConstantIntOp>(location, 0, rewriter.getI32Type());
    rewriter.create<decisionforest::InitSparseTileOp>(location, getGlobalMemref, tileIndex, thresholdVal, indexVal, tileShapeID, childIndex);
  }
  rewriter.setInsertionPointAfter(forLoop);
  
  auto modelSize = rewriter.create<decisionforest::GetModelMemrefSizeOp>(location, rewriter.getI32Type(), getGlobalMemref, lenIndexConst);
  rewriter.create<mlir::func::ReturnOp>(location, static_cast<Value>(modelSize));
  module.push_back(initModelMemrefFunc);
}

mlir::Value SparseRepresentation::GetTreeMemref(mlir::Value treeValue) {
  auto getTreeOp = treeValue.getDefiningOp();
  AssertOpIsOfType<mlir::decisionforest::GetTreeFromEnsembleOp>(getTreeOp);
  auto getTreeOperationMapIter = sparseGetTreeOperationMap.find(getTreeOp);
  assert(getTreeOperationMapIter != sparseGetTreeOperationMap.end());
  auto treeMemref = getTreeOperationMapIter->second.treeMemref;
  return treeMemref;
}

mlir::Value SparseRepresentation::GetLeafMemref(mlir::Value treeValue) {
  auto getTreeOp = treeValue.getDefiningOp();
  AssertOpIsOfType<mlir::decisionforest::GetTreeFromEnsembleOp>(getTreeOp);
  auto getTreeOperationMapIter = sparseGetTreeOperationMap.find(getTreeOp);
  assert(getTreeOperationMapIter != sparseGetTreeOperationMap.end());
  auto leafMemref = getTreeOperationMapIter->second.leavesMemref;
  return leafMemref;
}

std::vector<mlir::Value> SparseRepresentation::GenerateExtraLoads(mlir::Location location, ConversionPatternRewriter &rewriter, mlir::Value treeMemref, 
                                              mlir::Value nodeIndex, mlir::Type tileType) {
  auto treeTileType = tileType.cast<decisionforest::TiledNumericalNodeType>();
  auto loadChildIndexOp = rewriter.create<decisionforest::LoadChildIndexOp>(location, treeTileType.getChildIndexType(), treeMemref, nodeIndex);
  auto childIndex = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(loadChildIndexOp));
  return std::vector<mlir::Value>{childIndex};
}

mlir::Value SparseRepresentation::GenerateMoveToChild(mlir::Location location, ConversionPatternRewriter &rewriter, mlir::Value nodeIndex, 
                                                      mlir::Value childNumber, int32_t tileSize, std::vector<mlir::Value>& extraLoads) {
  assert (extraLoads.size() > 0);
  auto childIndex = extraLoads.front();
  auto newIndex = rewriter.create<arith::AddIOp>(location, rewriter.getIndexType(), childIndex, childNumber);

  if (decisionforest::InsertDebugHelpers) {
    // (child base index, lutLookup result, new index)
    auto zeroVector = CreateZeroVectorIndexConst(rewriter, location, 3);
    auto zeroConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
    auto elem0Set = rewriter.create<vector::InsertElementOp>(location, childIndex, zeroVector, zeroConst);
    auto oneConst = rewriter.create<arith::ConstantIndexOp>(location, 1);
    auto elem1Set = rewriter.create<vector::InsertElementOp>(location, childNumber, elem0Set, oneConst);
    auto twoConst = rewriter.create<arith::ConstantIndexOp>(location, 2);
    auto elem2Set = rewriter.create<vector::InsertElementOp>(location, newIndex, elem1Set, twoConst);
    InsertPrintVectorOp(rewriter, location, 1, 64, 3, elem2Set);
  }
  return newIndex;
}

void SparseRepresentation::GenerateTreeMemref(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex) {
  auto location = op->getLoc();
  Operation* ensembleConstOp = ensemble.getDefiningOp();
  AssertOpIsOfType<mlir::decisionforest::EnsembleConstantOp>(ensembleConstOp);
  
  auto mapIter = sparseEnsembleConstantToMemrefsMap.find(ensembleConstOp);
  assert (mapIter != sparseEnsembleConstantToMemrefsMap.end());
  auto& ensembleInfo = mapIter->second;

  auto modelMemrefIndex = rewriter.create<memref::LoadOp>(location, ensembleInfo.offsetGlobal, treeIndex);
  auto treeLength = rewriter.create<memref::LoadOp>(location, ensembleInfo.lengthGlobal, treeIndex);; // TODO Need to put this into the map too
  auto treeMemref = rewriter.create<memref::SubViewOp>(location, ensembleInfo.modelGlobal, ArrayRef<OpFoldResult>({static_cast<Value>(modelMemrefIndex)}),
                                                        ArrayRef<OpFoldResult>({static_cast<Value>(treeLength)}), ArrayRef<OpFoldResult>({rewriter.getIndexAttr(1)}));

  int32_t tileSize = ensembleInfo.modelGlobal.getType().cast<MemRefType>().getElementType().cast<decisionforest::TiledNumericalNodeType>().getTileSize();
  Value leavesMemref;
  if (tileSize > 1) {
    auto leavesMemrefIndex = rewriter.create<memref::LoadOp>(location, ensembleInfo.leavesOffsetGlobal, treeIndex);
    auto leavesLength = rewriter.create<memref::LoadOp>(location, ensembleInfo.leavesLengthGlobal, treeIndex);; // TODO Need to put this into the map too
    leavesMemref = rewriter.create<memref::SubViewOp>(location, ensembleInfo.leavesGlobal, ArrayRef<OpFoldResult>({static_cast<Value>(leavesMemrefIndex)}),
                                                        ArrayRef<OpFoldResult>({static_cast<Value>(leavesLength)}), ArrayRef<OpFoldResult>({rewriter.getIndexAttr(1)}));
  }   
  // if (decisionforest::InsertDebugHelpers) {
  //   rewriter.create<decisionforest::PrintTreeToDOTFileOp>(location, treeMemref, treeIndex);
  // }
  sparseGetTreeOperationMap[op] = { static_cast<Value>(treeMemref), static_cast<Value>(leavesMemref) };
}

mlir::Value SparseRepresentation::GenerateGetTreeClassId(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex) {
  Operation* ensembleConstOp = ensemble.getDefiningOp();
  AssertOpIsOfType<mlir::decisionforest::EnsembleConstantOp>(ensembleConstOp);
  
  auto mapIter = sparseEnsembleConstantToMemrefsMap.find(ensembleConstOp);
  assert (mapIter != sparseEnsembleConstantToMemrefsMap.end());
  auto& ensembleInfo = mapIter->second;

  auto treeClassMemref = ensembleInfo.classInfoGlobal;
  auto treeClassMemrefType = treeClassMemref.getType().cast<mlir::MemRefType>();

  auto classId = rewriter.create<memref::LoadOp>(op->getLoc(), treeClassMemrefType.getElementType(), treeClassMemref, treeIndex);
  return classId;
}

mlir::Value SparseRepresentation::GenerateGetLeafValueOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, 
                                                         mlir::Value nodeIndex) {
  auto location = op->getLoc();
  auto treeMemref = this->GetTreeMemref(treeValue);
  auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
  assert (treeMemrefType);

  auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
  auto thresholdType = treeTileType.getThresholdFieldType();

  if (decisionforest::InsertDebugHelpers) {
    rewriter.create<decisionforest::PrintTreeNodeOp>(location, nodeIndex);
  }

  if (treeTileType.getTileSize() == 1) {
    auto loadThresholdOp = rewriter.create<decisionforest::LoadTileThresholdsOp>(location, thresholdType, treeMemref, static_cast<Value>(nodeIndex));
    Value leafValue = loadThresholdOp;
    return static_cast<Value>(leafValue);
  }
  else {
    auto treeMemrefLen = rewriter.create<memref::DimOp>(location, treeMemref, 0);
    auto leafIndex = rewriter.create<arith::SubIOp>(location, nodeIndex, treeMemrefLen);
    auto leavesMemref = this->GetLeafMemref(treeValue);
    auto leafValue = rewriter.create<memref::LoadOp>(location, leavesMemref, static_cast<Value>(leafIndex));
    
    // auto resultConst = rewriter.create<arith::ConstantFloatOp>(location, APFloat(double(0.5)), rewriter.getF64Type());
    // TODO cast the loaded value to the correct result type of the tree. 
    return static_cast<Value>(leafValue);
  }
}

mlir::Value SparseRepresentation::GenerateIsLeafOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, mlir::Value nodeIndex) {
  auto location = op->getLoc();
  auto treeMemref = this->GetTreeMemref(treeValue);
  auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
  assert (treeMemrefType);

  auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
  if (treeTileType.getTileSize() == 1) {
    auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
    auto featureIndexType = treeTileType.getIndexFieldType();
    auto loadFeatureIndexOp = rewriter.create<decisionforest::LoadTileFeatureIndicesOp>(location, featureIndexType, treeMemref, static_cast<Value>(nodeIndex));    
    Value featureIndexValue = loadFeatureIndexOp;
    auto minusOneConstant = rewriter.create<arith::ConstantIntOp>(location, int64_t(-1), treeTileType.getIndexElementType());
    auto comparison = rewriter.create<arith::CmpIOp>(location, mlir::arith::CmpIPredicate::eq, featureIndexValue, static_cast<Value>(minusOneConstant));
    
    if (decisionforest::InsertDebugHelpers) {
      Value outcome = rewriter.create<mlir::arith::ExtUIOp>(location, rewriter.getI32Type(), static_cast<Value>(comparison));
      Value featureIndexI32 = featureIndexValue;
      if (!featureIndexType.isInteger(32))
        featureIndexI32 = rewriter.create<mlir::arith::ExtSIOp>(location, rewriter.getI32Type(), featureIndexValue);
      rewriter.create<decisionforest::PrintIsLeafOp>(location, nodeIndex, featureIndexI32, outcome);
    }
    return static_cast<Value>(comparison);
  }
  else {
    // Check if node index is out of bounds
    auto treeMemrefLen = rewriter.create<memref::DimOp>(location, treeMemref, 0);
    auto nodeIndexOutOfBounds = rewriter.create<arith::CmpIOp>(location, arith::CmpIPredicate::sge, nodeIndex, treeMemrefLen);
    return static_cast<Value>(nodeIndexOutOfBounds);
  }
}

mlir::Value SparseRepresentation::GenerateIsLeafTileOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, mlir::Value nodeIndex) {
    auto location = op->getLoc();
    auto treeMemref = GetTreeMemref(treeValue);
    auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
    assert (treeMemrefType);
    
    auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
    auto childIndexType = treeTileType.getChildIndexType();
    auto loadChildIndexOp = rewriter.create<decisionforest::LoadChildIndexOp>(location, childIndexType, treeMemref, static_cast<Value>(nodeIndex));    
    
    Value childIndexValue = static_cast<Value>(loadChildIndexOp);
    assert (treeTileType.getTileSize() != 1);
    
    auto minusOneConstant = rewriter.create<arith::ConstantIntOp>(location, int64_t(-1), childIndexType);
    auto comparison = rewriter.create<arith::CmpIOp>(location, mlir::arith::CmpIPredicate::eq, childIndexValue, static_cast<Value>(minusOneConstant));
    
    if (decisionforest::InsertDebugHelpers) {
      Value outcome = rewriter.create<mlir::arith::ExtUIOp>(location, rewriter.getI32Type(), static_cast<Value>(comparison));
      Value childIndexI32 = childIndexValue;
      if (!childIndexType.isInteger(32))
        childIndexI32 = rewriter.create<mlir::arith::ExtSIOp>(location, rewriter.getI32Type(), childIndexValue);
      rewriter.create<decisionforest::PrintIsLeafOp>(location, nodeIndex, childIndexI32, outcome);
    }
    return comparison;
}

std::shared_ptr<IRepresentation> RepresentationFactory::GetRepresentation(const std::string& name) {
  if (name == "array")
    return std::make_shared<ArrayBasedRepresentation>();
  else if (name == "sparse")
    return std::make_shared<SparseRepresentation>();
  
  assert(false && "Unknown serialization format");
  return nullptr;
}

std::shared_ptr<IRepresentation> ConstructRepresentation() {
  if (decisionforest::UseSparseTreeRepresentation)
    return RepresentationFactory::GetRepresentation("sparse");
  else
    return RepresentationFactory::GetRepresentation("array");
}

} // decisionforest
} // mlir
