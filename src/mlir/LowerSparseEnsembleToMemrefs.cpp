#include <iostream>
#include "Dialect.h"
// #include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "MemrefTypes.h"
#include "Dialect.h"
#include "TreeTilingUtils.h"
#include "TiledTree.h"
#include "Logger.h"
#include "CodeGenStateMachine.h"
#include "TraverseTreeTileOpLowering.h"
#include "OpLoweringUtils.h"

using namespace mlir;

namespace {

struct EnsembleConstantLoweringInfo {
  Value modelGlobal;
  Value offsetGlobal;
  Value lengthGlobal;
  Value lutGlobal;
  Value leavesGlobal;
  Value leavesOffsetGlobal;
  Value leavesLengthGlobal;
  Value classInfoGlobal;

  Type modelGlobalType;
  Type offsetGlobaltype;
  Type lengthGlobalType;
  Type lutGlobalType;
  Type leavesGlobalType;
  Type classInfoType;
};

// Maps an ensemble constant operation to a model memref and an offsets memref
std::map<Operation*, EnsembleConstantLoweringInfo> ensembleConstantToMemrefsMap;

struct GetTreeLoweringInfo {
  Value treeMemref;
  Value leavesMemref;
};

// Maps a GetTree operation to a memref that represents the tree once the ensemble constant has been replaced
std::map<Operation*, GetTreeLoweringInfo> getTreeOperationMap;

class SaveAndRestoreInsertionPoint {
  mlir::OpBuilder::InsertPoint m_insertPoint;
  mlir::ConversionPatternRewriter& m_builder;
public:
  SaveAndRestoreInsertionPoint(mlir::ConversionPatternRewriter& builder)
    : m_builder(builder)
  {
    m_insertPoint = m_builder.saveInsertionPoint();
  }

  ~SaveAndRestoreInsertionPoint() {
   m_builder.restoreInsertionPoint(m_insertPoint);
  }
};

void InsertPrintVectorOp(ConversionPatternRewriter &rewriter, Location location, int32_t kind, int32_t bitWidth, 
                         int32_t tileSize, Value vectorValue) {
  auto tileSizeConst = rewriter.create<arith::ConstantIntOp>(location, tileSize, rewriter.getI32Type());
  auto kindConst = rewriter.create<arith::ConstantIntOp>(location, kind, rewriter.getI32Type());
  auto bitWidthConst = rewriter.create<arith::ConstantIntOp>(location, bitWidth, rewriter.getI32Type());
  std::vector<Value> vectorValues;
  for (int32_t i=0; i<tileSize ; ++i) {
    auto iConst = rewriter.create<arith::ConstantIntOp>(location, int64_t(i), rewriter.getI32Type());
    auto ithValue = rewriter.create<vector::ExtractElementOp>(location, vectorValue, iConst);
    vectorValues.push_back(ithValue);
  }
  rewriter.create<decisionforest::PrintVectorOp>(location, kindConst, bitWidthConst, tileSizeConst, ValueRange(vectorValues));
}

Value CreateZeroVectorFPConst(ConversionPatternRewriter &rewriter, Location location, Type fpType, int32_t tileSize) {
  Value zeroConst;
  auto vectorType = VectorType::get(tileSize, fpType);
  if (fpType.isa<mlir::Float64Type>())
    zeroConst = rewriter.create<arith::ConstantFloatOp>(location, llvm::APFloat(0.0), fpType.cast<FloatType>());
  else if(fpType.isa<mlir::Float32Type>())
    zeroConst = rewriter.create<arith::ConstantFloatOp>(location, llvm::APFloat((float)0.0), fpType.cast<FloatType>());
  else
    assert(false && "Unsupported floating point type");
  auto vectorValue = rewriter.create<vector::BroadcastOp>(location, vectorType, zeroConst);
  return vectorValue;
}

Value CreateZeroVectorIntConst(ConversionPatternRewriter &rewriter, Location location, Type intType, int32_t tileSize) {
  Value zeroConst = rewriter.create<arith::ConstantIntOp>(location, 0, intType);
  auto vectorType = VectorType::get(tileSize, intType);
  auto vectorValue = rewriter.create<vector::BroadcastOp>(location, vectorType, zeroConst);
  return vectorValue;
}

Value CreateZeroVectorIndexConst(ConversionPatternRewriter &rewriter, Location location, int32_t tileSize) {
  Value zeroConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
  auto vectorType = VectorType::get(tileSize, rewriter.getIndexType());
  auto vectorValue = rewriter.create<vector::BroadcastOp>(location, vectorType, zeroConst);
  return vectorValue;
}

struct EnsembleConstantOpLowering: public ConversionPattern {
  EnsembleConstantOpLowering(MLIRContext *ctx) : ConversionPattern(mlir::decisionforest::EnsembleConstantOp::getOperationName(), 1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
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
                                        leavesMemrefName, leavesLengthMemrefName, leavesOffsetMemrefName, classInfoMemrefName);
    AddModelMemrefInitFunction(owningModule, modelMemrefName, std::get<0>(memrefTypes).cast<MemRefType>(), rewriter, location);
    
    // Add getters for all the globals we've created
    auto getModelGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<0>(memrefTypes), modelMemrefName);
    auto getOffsetGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<1>(memrefTypes), offsetMemrefName);
    auto getLengthGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<1>(memrefTypes), lengthMemrefName);
    auto getLeavesGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<2>(memrefTypes), leavesMemrefName);
    auto getLeavesOffsetGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<1>(memrefTypes), leavesOffsetMemrefName);
    auto getLeavesLengthGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<1>(memrefTypes), leavesLengthMemrefName);
    auto classInfoGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<3>(memrefTypes), classInfoMemrefName);
    
    auto forestType = ensembleConstOp.getResult().getType().cast<decisionforest::TreeEnsembleType>();
    auto firstTreeType = forestType.getTreeType(0).cast<mlir::decisionforest::TreeType>();
    auto firstTreeTileSize = firstTreeType.getTileSize();

    Type lookUpTableMemrefType;
    Value getLUT;
    if (firstTreeTileSize > 1) {
      std::string lookupTableMemrefName = "lookupTable";
      lookUpTableMemrefType = AddChildIndexLookUpTable(owningModule, ensembleConstOp, rewriter, location, lookupTableMemrefName);
      getLUT = rewriter.create<memref::GetGlobalOp>(location, lookUpTableMemrefType, lookupTableMemrefName);
    }

    EnsembleConstantLoweringInfo info {static_cast<Value>(getModelGlobal), static_cast<Value>(getOffsetGlobal), 
                                       static_cast<Value>(getLengthGlobal), getLUT,
                                       getLeavesGlobal, getLeavesOffsetGlobal, getLeavesLengthGlobal, classInfoGlobal,
                                       std::get<0>(memrefTypes), std::get<1>(memrefTypes), std::get<1>(memrefTypes), 
                                       lookUpTableMemrefType, std::get<2>(memrefTypes), std::get<3>(memrefTypes)};
    ensembleConstantToMemrefsMap[op] = info;
    
    // rewriter.replaceOp(op, static_cast<Value>(getModelGlobal));
    rewriter.eraseOp(op);

    return mlir::success();
  }

  Type AddChildIndexLookUpTable(mlir::ModuleOp module, mlir::decisionforest::EnsembleConstantOp& ensembleConstOp,
                                 ConversionPatternRewriter &rewriter, Location location, std::string& lookupTableMemrefName) const {
    SaveAndRestoreInsertionPoint saveAndRestoreInsertPoint(rewriter);
    rewriter.setInsertionPoint(&module.front());

    auto forestType = ensembleConstOp.getResult().getType().cast<decisionforest::TreeEnsembleType>();
    // We will assume that all trees have the same tile size
    auto numTrees = static_cast<int32_t>(forestType.getNumberOfTrees());
    assert(numTrees > 0);
    auto firstTreeType = forestType.getTreeType(0).cast<mlir::decisionforest::TreeType>();
    auto firstTreeTileSize = firstTreeType.getTileSize();
    for (int32_t i=1 ; i<numTrees ; ++i) {
      auto treeType = forestType.getTreeType(i).cast<mlir::decisionforest::TreeType>();
      auto tileSize = treeType.getTileSize();
      assert (firstTreeTileSize == tileSize && "All tree's should have the same tile size");
    }
    auto tileSize = firstTreeTileSize;
    if (tileSize == 1)
      return Type(); // We don't need a lookup table if the tile size is 1
    
    auto numberOfTileOutcomes = static_cast<int>(std::pow(2, tileSize));
    auto numberOfTileShapes = mlir::decisionforest::TileShapeToTileIDMap::NumberOfTileShapes(tileSize);
    // TODO We may need to implement something smarter here. We don't really need I8's for each outcome. We could store all outcomes
    // in a single int64 for tile size 4 for example (each entry needs 3 bits and there are 16 entries -- one for each outcome). 
    auto lutMemrefType = MemRefType::get({numberOfTileShapes, numberOfTileOutcomes}, rewriter.getI8Type());

    rewriter.create<memref::GlobalOp>(location, lookupTableMemrefName,
                                      /*sym_visibility=*/rewriter.getStringAttr("private"),
                                      /*type=*/lutMemrefType,
                                      /*initial_value=*/rewriter.getUnitAttr(),
                                      /*constant=*/false, IntegerAttr());
    AddGlobalMemrefGetter(module, lookupTableMemrefName, lutMemrefType, rewriter, location);

    return lutMemrefType;
  }
  
  void AddGlobalMemrefGetter(mlir::ModuleOp module, std::string globalName, Type memrefType, ConversionPatternRewriter &rewriter, Location location) const {
    SaveAndRestoreInsertionPoint saveAndRestoreEntryPoint(rewriter);
    auto getMemrefFuncType = rewriter.getFunctionType(TypeRange({}), memrefType);
    std::string funcName = "Get_" + globalName;
    NamedAttribute visibilityAttribute{module.getSymVisibilityAttrName(), rewriter.getStringAttr("public")};
    auto getGlobalMemrefFunc = mlir::func::FuncOp::create(location, funcName, getMemrefFuncType, ArrayRef<NamedAttribute>(visibilityAttribute));
    auto &entryBlock = *getGlobalMemrefFunc.addEntryBlock();
    rewriter.setInsertionPointToStart(&entryBlock);

    auto getGlobalOffsets = rewriter.create<memref::GetGlobalOp>(location, memrefType, globalName);
    rewriter.create<mlir::func::ReturnOp>(location, static_cast<Value>(getGlobalOffsets));

    module.push_back(getGlobalMemrefFunc);
  }
  
  void AddModelMemrefInitFunction(mlir::ModuleOp module, std::string globalName, MemRefType memrefType, 
                                  ConversionPatternRewriter &rewriter, Location location) const {
    assert (memrefType.getShape().size() == 1);
    SaveAndRestoreInsertionPoint saveAndRestoreEntryPoint(rewriter);
    auto modelMemrefElementType = memrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
    int32_t tileSize = modelMemrefElementType.getTileSize();
    auto thresholdArgType = MemRefType::get({ memrefType.getShape()[0] * tileSize }, modelMemrefElementType.getThresholdElementType());
    auto indexArgType = MemRefType::get({ memrefType.getShape()[0] * tileSize }, modelMemrefElementType.getIndexElementType());
    auto tileShapeIDArgType = MemRefType::get(memrefType.getShape(), modelMemrefElementType.getTileShapeType());
    auto childrenIndexArgType = MemRefType::get(memrefType.getShape(), modelMemrefElementType.getChildIndexType());
    auto leafBitMaskArgType = MemRefType::get(memrefType.getShape(), modelMemrefElementType.getLeafBitMaskType());
    mlir::FunctionType initMemrefFuncType;
    if (decisionforest::RemoveExtraHopInSparseRepresentation)
      initMemrefFuncType = rewriter.getFunctionType(TypeRange{thresholdArgType, 
                                                             indexArgType, tileShapeIDArgType,
                                                             childrenIndexArgType,
                                                             childrenIndexArgType /*Type of leaf index*/,
                                                             leafBitMaskArgType},
                                                    rewriter.getI32Type());
    else
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
      if (decisionforest::RemoveExtraHopInSparseRepresentation) {
        auto leafIndex = rewriter.create<memref::LoadOp>(location, entryBlock.getArgument(4), tileIndex);
        assert (childIndex.getType() == leafIndex.getType());
        auto leafBitMask = rewriter.create<memref::LoadOp>(location, entryBlock.getArgument(5), tileIndex);
        // Create a vector value <childIndex, leafIndex>
        auto childAndLeafIndexVec = CreateZeroVectorIntConst(rewriter, location, childIndex.getType(), 2);
        // auto zeroConst = rewriter.create<arith::ConstantIntOp>(location, 0, rewriter.getI32Type());
        // auto oneConst = rewriter.create<arith::ConstantIntOp>(location, 1, rewriter.getI32Type());
        childAndLeafIndexVec = rewriter.create<vector::InsertElementOp>(location, childIndex, childAndLeafIndexVec, zeroIndexConst);
        childAndLeafIndexVec = rewriter.create<vector::InsertElementOp>(location, leafIndex, childAndLeafIndexVec, oneIndexConst);
        rewriter.create<decisionforest::InitSparseTileWithLeafIndexOp>(location, getGlobalMemref, tileIndex, thresholdVec, 
                                                          indexVec, tileShapeID, childAndLeafIndexVec, leafBitMask);
      }
      else {
        rewriter.create<decisionforest::InitSparseTileOp>(location, getGlobalMemref, tileIndex, thresholdVec, indexVec, tileShapeID, childIndex);
      }
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

  std::tuple<Type, Type, Type, Type> AddGlobalMemrefs(mlir::ModuleOp module, mlir::decisionforest::EnsembleConstantOp& ensembleConstOp,
                                          ConversionPatternRewriter &rewriter, Location location,
                                          const std::string& modelMemrefName, const std::string& offsetMemrefName, const std::string& lengthMemrefName,
                                          const std::string& leavesMemrefName, const std::string& leavesLengthMemrefName,
                                          const std::string& leavesOffsetMemrefName, const std::string& treeInfo) const {
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
                                                                         tileSize, childIndexType, decisionforest::getDefaultLeafBitMakType(thresholdType.getContext()));

    PersistDecisionForest(forest, forestType);
    
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
};

struct GetTreeOpLowering: public ConversionPattern {
  GetTreeOpLowering(MLIRContext *ctx) : ConversionPattern(mlir::decisionforest::GetTreeFromEnsembleOp::getOperationName(), 1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    // Create a subview of the model memref corresponding to this ensemble with the index equal to offsetMemref[treeIndex]
    mlir::decisionforest::GetTreeFromEnsembleOp getTreeOp = llvm::dyn_cast<mlir::decisionforest::GetTreeFromEnsembleOp>(op);
    assert(getTreeOp);
    assert(operands.size() == 2);
    if (!getTreeOp)
        return mlir::failure();

    auto location = op->getLoc();
    Operation* ensembleConstOp = operands[0].getDefiningOp();
    AssertOpIsOfType<mlir::decisionforest::EnsembleConstantOp>(ensembleConstOp);
    
    auto mapIter = ensembleConstantToMemrefsMap.find(ensembleConstOp);
    assert (mapIter != ensembleConstantToMemrefsMap.end());
    auto& ensembleInfo = mapIter->second;

    Value treeIndex = operands[1];

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
    getTreeOperationMap[op] = { static_cast<Value>(treeMemref), static_cast<Value>(leavesMemref) };

    rewriter.eraseOp(op);

    return mlir::success();
  }
};

Value GetTreeMemrefFromTreeOperand(Value treeValue) {
  auto getTreeOp = treeValue.getDefiningOp();
  AssertOpIsOfType<mlir::decisionforest::GetTreeFromEnsembleOp>(getTreeOp);
  auto getTreeOperationMapIter = getTreeOperationMap.find(getTreeOp);
  assert(getTreeOperationMapIter != getTreeOperationMap.end());
  auto treeMemref = getTreeOperationMapIter->second.treeMemref;
  return treeMemref;
}

Value GetLeavesMemrefFromTreeOperand(Value treeValue) {
  auto getTreeOp = treeValue.getDefiningOp();
  AssertOpIsOfType<mlir::decisionforest::GetTreeFromEnsembleOp>(getTreeOp);
  auto getTreeOperationMapIter = getTreeOperationMap.find(getTreeOp);
  assert(getTreeOperationMapIter != getTreeOperationMap.end());
  return getTreeOperationMapIter->second.leavesMemref;
}

Value GetLUTFromTreeOperand(Value treeValue) {
  auto getTreeOp = treeValue.getDefiningOp();
  auto getTreeFromEnsembleOp = AssertOpIsOfType<mlir::decisionforest::GetTreeFromEnsembleOp>(getTreeOp);
  auto ensembleVal = getTreeFromEnsembleOp.getOperand(0);
  
  auto ensembleConstOp = AssertOpIsOfType<mlir::decisionforest::EnsembleConstantOp>(ensembleVal.getDefiningOp());
  auto mapIter = ensembleConstantToMemrefsMap.find(ensembleConstOp);
  assert (mapIter != ensembleConstantToMemrefsMap.end());
  auto& ensembleInfo = mapIter->second;
  return ensembleInfo.lutGlobal;
}

struct GetRootOpLowering: public ConversionPattern {
  GetRootOpLowering(MLIRContext *ctx) : ConversionPattern(mlir::decisionforest::GetRootOp::getOperationName(), 1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    auto getRootOp = AssertOpIsOfType<mlir::decisionforest::GetRootOp>(op);
    auto nodeIndexConst = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);

    auto treeMemref = GetTreeMemrefFromTreeOperand(operands[0]);
    auto nodeType = getRootOp.getResult().getType();
    auto node = rewriter.create<decisionforest::IndexToNodeOp>(op->getLoc(), nodeType, treeMemref, static_cast<Value>(nodeIndexConst));
    rewriter.replaceOp(op, static_cast<Value>(node));
    return mlir::success();
  }
};

struct IsLeafTileOpLowering: public ConversionPattern {
  IsLeafTileOpLowering(MLIRContext *ctx) : ConversionPattern(mlir::decisionforest::IsLeafTileOp::getOperationName(), 1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    
    // Create a subview of the model memref corresponding to this ensemble with the index equal to offsetMemref[treeIndex]
    mlir::decisionforest::IsLeafTileOp isLeafOp = AssertOpIsOfType<mlir::decisionforest::IsLeafTileOp>(op);
    assert(operands.size() == 2);
    if (!isLeafOp)
        return mlir::failure();
    
    auto location = op->getLoc();

    auto treeMemref = GetTreeMemrefFromTreeOperand(operands[0]);
    auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
    assert (treeMemrefType);

    auto nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(location, rewriter.getIndexType(), treeMemref, operands[1]); // Convert the node to an index
    // auto nodeIndexType = nodeIndex.getType().cast<IndexType>();
    // assert(nodeIndexType);

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
    rewriter.replaceOp(op, static_cast<Value>(comparison));

    return mlir::success();
  }
};

struct IsLeafOpLowering: public ConversionPattern {
  IsLeafOpLowering(MLIRContext *ctx) : ConversionPattern(mlir::decisionforest::IsLeafOp::getOperationName(), 1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    
    // Create a subview of the model memref corresponding to this ensemble with the index equal to offsetMemref[treeIndex]
    mlir::decisionforest::IsLeafOp isLeafOp = AssertOpIsOfType<mlir::decisionforest::IsLeafOp>(op);
    assert(operands.size() == 2);
    if (!isLeafOp)
        return mlir::failure();
    
    auto location = op->getLoc();

    auto treeMemref = GetTreeMemrefFromTreeOperand(operands[0]);
    auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
    assert (treeMemrefType);

    auto nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(location, rewriter.getIndexType(), treeMemref, operands[1]); // Convert the node to an index

    // auto nodeIndexType = nodeIndex.getType().cast<IndexType>();
    // assert(nodeIndexType);
    if (decisionforest::OptimizedSparseRepresentation == false) {
      // Check if node index is out of bounds
      auto treeMemrefLen = rewriter.create<memref::DimOp>(location, treeMemref, 0);
      auto nodeIndexOutOfBounds = rewriter.create<arith::CmpIOp>(location, arith::CmpIPredicate::sge, nodeIndex, treeMemrefLen);
      auto ifElse = rewriter.create<scf::IfOp>(location, TypeRange{ rewriter.getI1Type() }, nodeIndexOutOfBounds, true);
      {
        // return true if the index is out of bounds
        auto ifBuilder = ifElse.getThenBodyBuilder();
        auto trueConst = ifBuilder.create<arith::ConstantIntOp>(location, 1, rewriter.getI1Type());
        if (decisionforest::InsertDebugHelpers) {
          Value outcome = ifBuilder.create<mlir::arith::ExtUIOp>(location, ifBuilder.getI32Type(), static_cast<Value>(nodeIndexOutOfBounds));
          Value featureIndexValue = ifBuilder.create<arith::ConstantIntOp>(location, int64_t(-1), ifBuilder.getI32Type());
          ifBuilder.create<decisionforest::PrintIsLeafOp>(location, nodeIndex, featureIndexValue, outcome);
        }
        ifBuilder.create<scf::YieldOp>(location, static_cast<Value>(trueConst));
      }
      {    
        auto elseBuilder = ifElse.getElseBodyBuilder();
        auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
        auto featureIndexType = treeTileType.getIndexFieldType();
        auto loadFeatureIndexOp = elseBuilder.create<decisionforest::LoadTileFeatureIndicesOp>(location, featureIndexType, treeMemref, static_cast<Value>(nodeIndex));    
        
        Value featureIndexValue;
        if (treeTileType.getTileSize() == 1) {
          featureIndexValue = loadFeatureIndexOp;
        }
        else {
          auto indexVectorType = featureIndexType.cast<mlir::VectorType>();
          assert (indexVectorType);
          auto zeroConst = elseBuilder.create<arith::ConstantIntOp>(location, int64_t(0), elseBuilder.getI32Type());
          auto extractFirstElement = elseBuilder.create<vector::ExtractElementOp>(location, static_cast<Value>(loadFeatureIndexOp), zeroConst);
          featureIndexValue = extractFirstElement;
        }
        auto minusOneConstant = elseBuilder.create<arith::ConstantIntOp>(location, int64_t(-1), treeTileType.getIndexElementType());
        auto comparison = elseBuilder.create<arith::CmpIOp>(location, mlir::arith::CmpIPredicate::eq, featureIndexValue, static_cast<Value>(minusOneConstant));
        
        // Loop condition = out of bounds || feature index == -1
        // auto loopCondition = elseBuilder.create<arith::OrIOp>(location, nodeIndexOutOfBounds, comparison);

        if (decisionforest::InsertDebugHelpers) {
          Value outcome = elseBuilder.create<mlir::arith::ExtUIOp>(location, elseBuilder.getI32Type(), static_cast<Value>(comparison));
          elseBuilder.create<decisionforest::PrintIsLeafOp>(location, nodeIndex, featureIndexValue, outcome);
        }
        elseBuilder.create<scf::YieldOp>(location, static_cast<Value>(comparison));
      }
      rewriter.replaceOp(op, static_cast<Value>(ifElse.getResult(0)));
    }
    else {
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
        rewriter.replaceOp(op, static_cast<Value>(comparison));
      }
      else {
        // Check if node index is out of bounds
        auto treeMemrefLen = rewriter.create<memref::DimOp>(location, treeMemref, 0);
        auto nodeIndexOutOfBounds = rewriter.create<arith::CmpIOp>(location, arith::CmpIPredicate::sge, nodeIndex, treeMemrefLen);
        rewriter.replaceOp(op, static_cast<Value>(nodeIndexOutOfBounds));
      }
    }
    return mlir::success();
  }
};

struct GetTreeClassIdOpLowering: public ConversionPattern {
  GetTreeClassIdOpLowering(MLIRContext *ctx) : ConversionPattern(mlir::decisionforest::GetTreeClassIdOp::getOperationName(), 1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    // auto getClassIdOp = AssertOpIsOfType<mlir::decisionforest::GetTreeClassIdOp>(op);
    assert(operands.size() == 2);

    Operation* ensembleConstOp = operands[0].getDefiningOp();
    AssertOpIsOfType<mlir::decisionforest::EnsembleConstantOp>(ensembleConstOp);
    
    auto mapIter = ensembleConstantToMemrefsMap.find(ensembleConstOp);
    assert (mapIter != ensembleConstantToMemrefsMap.end());
    auto& ensembleInfo = mapIter->second;

    auto treeClassMemref = ensembleInfo.classInfoGlobal;
    auto treeClassMemrefType = treeClassMemref.getType().cast<mlir::MemRefType>();

    Value treeIndex = operands[1];
    auto classId = rewriter.create<memref::LoadOp>(op->getLoc(), treeClassMemrefType.getElementType(), treeClassMemref, treeIndex);
    
    rewriter.replaceOp(op, static_cast<Value>(classId));
    return mlir::success();
  }
};

struct InterleavedTraverseTreeTileOpLowering : public ConversionPattern {
  InterleavedTraverseTreeTileOpLowering(MLIRContext *ctx) : ConversionPattern(mlir::decisionforest::InterleavedTraverseTreeTileOp::getOperationName(), 1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,ConversionPatternRewriter &rewriter) const final {
    decisionforest::InterleavedTraverseTreeTileOpLoweringHelper traverseLowringHelper(GetTreeMemrefFromTreeOperand, GetLUTFromTreeOperand, decisionforest::Representation::kSparse);
    return traverseLowringHelper.matchAndRewrite(AssertOpIsOfType<mlir::decisionforest::InterleavedTraverseTreeTileOp>(op), operands, rewriter);
  }
};

struct TraverseTreeTileOpLowering : public ConversionPattern {
  TraverseTreeTileOpLowering(MLIRContext *ctx) : ConversionPattern(mlir::decisionforest::TraverseTreeTileOp::getOperationName(), 1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    auto traverseTileOp = AssertOpIsOfType<mlir::decisionforest::TraverseTreeTileOp>(op);
    assert(operands.size() == 3);
    if (!traverseTileOp)
        return mlir::failure();
    
    auto treeMemref = GetTreeMemrefFromTreeOperand(operands[0]);
    auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
    assert (treeMemrefType);

    auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
    decisionforest::InterleavedCodeGenStateMachine codeGenStateMachine;
    if (treeTileType.getTileSize() == 1)
      codeGenStateMachine.AddStateMachine(
        std::make_unique<decisionforest::ScalarTraverseTileCodeGenerator>(
          treeMemref,
          operands[2],
          operands[1],
          traverseTileOp.getResult().getType(),
          decisionforest::kSparse));
      // LowerOpTileSize1(op, operands, rewriter);
    else
      codeGenStateMachine.AddStateMachine(
        std::make_unique<decisionforest::VectorTraverseTileCodeGenerator>(
          operands[0],
          treeMemref,
          operands[2],
          operands[1],
          traverseTileOp.getResult().getType(),
          decisionforest::kSparse,
          GetLUTFromTreeOperand));
      // LowerOpForVectorTile(op, operands, rewriter);
    
    auto location = op->getLoc();
    while (codeGenStateMachine.EmitNext(rewriter, location));
    
    rewriter.replaceOp(op, static_cast<Value>(codeGenStateMachine.GetResult()[0]));

    return mlir::success();
  }

  void LowerOpTileSize1(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const {
    auto traverseTileOp = AssertOpIsOfType<mlir::decisionforest::TraverseTreeTileOp>(op);
    auto location = op->getLoc();
    
    auto treeMemref = GetTreeMemrefFromTreeOperand(operands[0]);
    auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
    assert (treeMemrefType);

    auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
    auto featureIndexType = treeTileType.getIndexElementType();
    auto thresholdType = treeTileType.getThresholdElementType();
    auto childIndexType = treeTileType.getChildIndexType();
    // Assert tile size is 1
    assert (treeTileType.getTileSize() == 1);

    auto node = operands[1];
    auto nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(location, rewriter.getIndexType(), treeMemref, node);
    if (decisionforest::InsertDebugHelpers) {
      rewriter.create<decisionforest::PrintTreeNodeOp>(location, nodeIndex);
    }
    // Load threshold
    auto loadThresholdOp = rewriter.create<decisionforest::LoadTileThresholdsOp>(location, thresholdType, treeMemref, static_cast<Value>(nodeIndex));
    // Load feature index
    auto loadFeatureIndexOp = rewriter.create<decisionforest::LoadTileFeatureIndicesOp>(location, featureIndexType, treeMemref, static_cast<Value>(nodeIndex));
    // Load the child index
    auto loadChildIndex = rewriter.create<decisionforest::LoadChildIndexOp>(location, childIndexType, treeMemref, static_cast<Value>(nodeIndex));
    auto childIndex = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(loadChildIndex));
    // Load feature value
    auto rowMemref = operands[2];
    auto rowMemrefType = rowMemref.getType().cast<MemRefType>();
    auto rowIndex = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(loadFeatureIndexOp));
    auto zeroIndex = rewriter.create<arith::ConstantIndexOp>(location, 0);
    auto feature = rewriter.create<memref::LoadOp>(location, rowMemrefType.getElementType(), rowMemref,
                                                   ValueRange({static_cast<Value>(zeroIndex), static_cast<Value>(rowIndex)}));

    if(decisionforest::InsertDebugHelpers) {
      rewriter.create<decisionforest::PrintComparisonOp>(location, feature, loadThresholdOp, loadFeatureIndexOp);
    }

    // result = Compare
    // TODO we need a cast here to make sure the threshold and the row element are the same type. The op expects both operands to be the same type.
    auto comparison = rewriter.create<arith::CmpFOp>(location,  mlir::arith::CmpFPredicate::UGE, static_cast<Value>(feature), static_cast<Value>(loadThresholdOp));
    auto comparisonUnsigned = rewriter.create<arith::ExtUIOp>(location, rewriter.getI32Type(), static_cast<Value>(comparison));

    // index = childIndex + result
    auto comparisonResultIndex = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(comparisonUnsigned));
    auto newIndex = rewriter.create<arith::AddIOp>(location, rewriter.getIndexType(), static_cast<Value>(childIndex), static_cast<Value>(comparisonResultIndex));
    
    // node = indexToNode(index)
    auto newNode = rewriter.create<decisionforest::IndexToNodeOp>(location, traverseTileOp.getResult().getType(), treeMemref, static_cast<Value>(newIndex));

    rewriter.replaceOp(op, static_cast<Value>(newNode));
  }

  Value ReduceComparisonResultVectorToInt(Value comparisonResult, int32_t tileSize, ConversionPatternRewriter &rewriter, Location location) const {
    auto i32VectorType = VectorType::get(tileSize, rewriter.getI32Type());
    auto comparisonExtended = rewriter.create<arith::ExtUIOp>(location, i32VectorType, comparisonResult);

    auto zeroI32Const = rewriter.create<arith::ConstantIntOp>(location, int64_t(0), rewriter.getI32Type());
    auto shiftVector = static_cast<Value>(rewriter.create<vector::BroadcastOp>(location, i32VectorType, zeroI32Const));
    for (int32_t shift=0, pos=tileSize-1 ; shift<tileSize; ++shift, --pos) {
      auto shiftValConst = rewriter.create<arith::ConstantIntOp>(location, int64_t(shift), rewriter.getI32Type());
      shiftVector = rewriter.create<vector::InsertOp>(location, static_cast<Value>(shiftValConst), 
                                                      static_cast<Value>(shiftVector), ArrayRef<int64_t>({ pos }));
    }

    auto leftShift = rewriter.create<arith::ShLIOp>(location, i32VectorType, comparisonExtended, shiftVector);
    auto kind = vector::CombiningKind::ADD;
    auto sum = rewriter.create<vector::ReductionOp>(location, kind, static_cast<Value>(leftShift));
    auto index = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(sum));
    return index;
  }

  Value ReduceComparisonResultVectorToInt_Bitcast(Value comparisonResult, int32_t tileSize, ConversionPatternRewriter &rewriter, Location location) const {
    auto bitcastVectorType = VectorType::get(1, rewriter.getIntegerType(tileSize));
    auto bitcastOp = rewriter.create<vector::BitCastOp>(location, bitcastVectorType, comparisonResult);
    auto zeroConst = rewriter.create<arith::ConstantIntOp>(location, int64_t(0), rewriter.getI32Type());
    auto integerResult = rewriter.create<vector::ExtractElementOp>(location, static_cast<Value>(bitcastOp), static_cast<Value>(zeroConst));
    auto zeroExtend = rewriter.create<arith::ExtUIOp>(location, rewriter.getI64Type(), integerResult); 
    auto index = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(zeroExtend));
    return index;
  }

  void LowerOpForVectorTile(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const {
    auto traverseTileOp = AssertOpIsOfType<mlir::decisionforest::TraverseTreeTileOp>(op);
    auto location = op->getLoc();
    
    auto treeMemref = GetTreeMemrefFromTreeOperand(operands[0]);
    auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
    assert (treeMemrefType);

    auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
    auto featureIndexType = treeTileType.getIndexFieldType();
    auto featureIndexVectorType = featureIndexType.cast<VectorType>();
    assert(featureIndexVectorType);
    auto thresholdType = treeTileType.getThresholdFieldType();
    auto thresholdVectorType = thresholdType.cast<VectorType>();
    assert(thresholdVectorType);
    auto tileShapeType = treeTileType.getTileShapeType();
    auto childIndexType = treeTileType.getChildIndexType();

    assert (treeTileType.getTileSize() > 1);
    auto tileSize = treeTileType.getTileSize();
    auto node = operands[1];
    auto nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(location, rewriter.getIndexType(), treeMemref, node);
    if (decisionforest::InsertDebugHelpers) {
      rewriter.create<decisionforest::PrintTreeNodeOp>(location, nodeIndex);
    }
    // Load threshold
    auto loadThresholdOp = rewriter.create<decisionforest::LoadTileThresholdsOp>(location, thresholdType, treeMemref, static_cast<Value>(nodeIndex));
    if (decisionforest::InsertDebugHelpers) {
      Value vectorVal = loadThresholdOp;
      if (!thresholdVectorType.getElementType().isF64()) {
        auto doubleVectorType = mlir::VectorType::get({ tileSize }, rewriter.getF64Type());
        vectorVal = rewriter.create<arith::ExtFOp>(location, doubleVectorType, loadThresholdOp);
      }
      InsertPrintVectorOp(rewriter, location, 0 /*fp kind*/, 64 /*thresholdVectorType.getElementType().getIntOrFloatBitWidth()*/, 
                          tileSize, static_cast<Value>(vectorVal));
    }
    // Load feature index
    auto loadFeatureIndexOp = rewriter.create<decisionforest::LoadTileFeatureIndicesOp>(location, featureIndexType, treeMemref, static_cast<Value>(nodeIndex));
    if (decisionforest::InsertDebugHelpers) {
      InsertPrintVectorOp(rewriter, location, 1 /*int kind*/, featureIndexVectorType.getElementType().getIntOrFloatBitWidth(), 
                          tileSize, static_cast<Value>(loadFeatureIndexOp));
    }

    Value tileShapeIndex, childIndex, leafBitMask;
    // Load the tile shape
    auto loadTileShapeOp = rewriter.create<decisionforest::LoadTileShapeOp>(location, tileShapeType, treeMemref, static_cast<Value>(nodeIndex));
    tileShapeIndex = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(loadTileShapeOp));

    if (decisionforest::RemoveExtraHopInSparseRepresentation) {
      // Load both the child and 
      auto childIndicesVectorType = VectorType::get({ 2 }, childIndexType);
      auto loadChildandLeafIndexOp = rewriter.create<decisionforest::LoadChildAndLeafIndexOp>(location, childIndicesVectorType, treeMemref, static_cast<Value>(nodeIndex));
      auto indexVectorType = VectorType::get({ 2 }, rewriter.getIndexType());
      childIndex = rewriter.create<arith::IndexCastOp>(location, indexVectorType, static_cast<Value>(loadChildandLeafIndexOp));
      
      auto loadLeafBitMask = rewriter.create<decisionforest::LoadLeafBitMaskOp>(location, treeTileType.getLeafBitMaskType(), treeMemref, static_cast<Value>(nodeIndex));
      leafBitMask = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(loadLeafBitMask));
    }
    else {
      // Load the child index
      auto loadChildIndexOp = rewriter.create<decisionforest::LoadChildIndexOp>(location, childIndexType, treeMemref, static_cast<Value>(nodeIndex));
      childIndex = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(loadChildIndexOp));
    }

    // Load feature value
    auto rowMemref = operands[2];
    auto rowMemrefType = rowMemref.getType().cast<MemRefType>();
    auto vectorIndexType = VectorType::get({ tileSize }, rewriter.getIndexType());
    auto rowIndex = rewriter.create<arith::IndexCastOp>(location, vectorIndexType, static_cast<Value>(loadFeatureIndexOp));
    auto zeroIndex = rewriter.create<arith::ConstantIndexOp>(location, 0);
    // auto zeroIndexVector = rewriter.create<vector::BroadcastOp>(location, vectorIndexType, zeroIndex);

    auto featuresVectorType = VectorType::get({ tileSize }, rowMemrefType.getElementType());
    auto oneI1Const = rewriter.create<arith::ConstantIntOp>(location, 1, rewriter.getI1Type());
    auto i1VectorType = VectorType::get(tileSize, rewriter.getI1Type());
    auto mask = rewriter.create<vector::BroadcastOp>(location, i1VectorType, oneI1Const);

    Value zeroPassThruConst;
    if (rowMemrefType.getElementType().isa<mlir::Float64Type>())
      zeroPassThruConst = rewriter.create<arith::ConstantFloatOp>(location, llvm::APFloat(0.0), rowMemrefType.getElementType().cast<FloatType>());
    else if(rowMemrefType.getElementType().isa<mlir::Float32Type>())
      zeroPassThruConst = rewriter.create<arith::ConstantFloatOp>(location, llvm::APFloat((float)0.0), rowMemrefType.getElementType().cast<FloatType>());
    else
      assert(false && "Unsupported floating point type");
    auto zeroPassThruVector = rewriter.create<vector::BroadcastOp>(location, featuresVectorType, zeroPassThruConst);
    
    auto features = rewriter.create<vector::GatherOp>(location, featuresVectorType, rowMemref,
                                                      ValueRange({static_cast<Value>(zeroIndex), static_cast<Value>(zeroIndex)}),
                                                      rowIndex, mask, zeroPassThruVector);

    if (decisionforest::InsertDebugHelpers) {
      Value vectorVal = features;
      if (!featuresVectorType.getElementType().isF64()) {
        auto doubleVectorType = mlir::VectorType::get({ tileSize }, rewriter.getF64Type());
        vectorVal = rewriter.create<arith::ExtFOp>(location, doubleVectorType, features);
      }
      InsertPrintVectorOp(rewriter, location, 0 /*fp kind*/, 64 /*featuresVectorType.getElementType().getIntOrFloatBitWidth()*/, 
                          tileSize, static_cast<Value>(vectorVal));
    }

    // TODO This needs a different print routine!
    // if(decisionforest::InsertDebugHelpers) {
    //   rewriter.create<decisionforest::PrintComparisonOp>(location, feature, loadThresholdOp, loadFeatureIndexOp);
    // }

    // result = Compare
    // TODO we need a cast here to make sure the threshold and the row element are the same type. The op expects both operands to be the same type.
    auto comparison = rewriter.create<arith::CmpFOp>(location,  mlir::arith::CmpFPredicate::ULT, static_cast<Value>(features), static_cast<Value>(loadThresholdOp));
    Value comparisonIndex;
    if (decisionforest::UseBitcastForComparisonOutcome)
      comparisonIndex = ReduceComparisonResultVectorToInt_Bitcast(comparison, tileSize, rewriter, location);
    else
      comparisonIndex = ReduceComparisonResultVectorToInt(comparison, tileSize, rewriter, location);


    // Load the child index from the LUT
    auto lutValue = GetLUTFromTreeOperand(operands[0]);
    auto childIndexInt = rewriter.create<memref::LoadOp>(location, lutValue, ValueRange{tileShapeIndex, comparisonIndex});
    auto childNumber = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(childIndexInt));

    if (decisionforest::RemoveExtraHopInSparseRepresentation) {
      // Shift so that the bit of leafBitMask corresponding to the 
      auto shift = rewriter.create<arith::ShRUIOp>(location, leafBitMask.getType(), leafBitMask, childNumber);

      // Add child index to the child and leaf index vector
      auto childNumberVectorValue = rewriter.create<vector::BroadcastOp>(location, childIndex.getType(), childNumber);
      auto potentialNewIndices = rewriter.create<arith::AddIOp>(location, childIndex, childNumberVectorValue);

      // Find the value of the bit
      auto oneIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 1);
      auto bitValue = rewriter.create<arith::AndIOp>(location, rewriter.getIndexType(), shift, oneIndexConst);

      // Extract the right index
      auto newIndex = rewriter.create<vector::ExtractElementOp>(location, potentialNewIndices, bitValue);

      if (decisionforest::InsertDebugHelpers) {
        // (is leaf bit, lutLookup result, new index)
        auto zeroVector = CreateZeroVectorIndexConst(rewriter, location, 3);
        auto zeroConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
        auto elem0Set = rewriter.create<vector::InsertElementOp>(location, bitValue, zeroVector, zeroConst);
        auto oneConst = rewriter.create<arith::ConstantIndexOp>(location, 1);
        auto elem1Set = rewriter.create<vector::InsertElementOp>(location, childNumber, elem0Set, oneConst);
        auto twoConst = rewriter.create<arith::ConstantIndexOp>(location, 2);
        auto elem2Set = rewriter.create<vector::InsertElementOp>(location, newIndex, elem1Set, twoConst);
        InsertPrintVectorOp(rewriter, location, 1, 64, 3, elem2Set);
      }    

      // node = indexToNode(index)
      auto newNode = rewriter.create<decisionforest::IndexToNodeOp>(location, traverseTileOp.getResult().getType(), treeMemref, static_cast<Value>(newIndex));
      rewriter.replaceOp(op, static_cast<Value>(newNode));
    }
    else {
      auto newIndex = rewriter.create<arith::AddIOp>(location, rewriter.getIndexType(), static_cast<Value>(childIndex), static_cast<Value>(childNumber));

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
      // node = indexToNode(index)
      auto newNode = rewriter.create<decisionforest::IndexToNodeOp>(location, traverseTileOp.getResult().getType(), treeMemref, static_cast<Value>(newIndex));
      rewriter.replaceOp(op, static_cast<Value>(newNode));
    }
  }
};

struct GetLeafValueOpLowering : public ConversionPattern {
  GetLeafValueOpLowering(MLIRContext *ctx) : ConversionPattern(mlir::decisionforest::GetLeafValueOp::getOperationName(), 1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    auto getLeafVal = AssertOpIsOfType<mlir::decisionforest::GetLeafValueOp>(op);
    assert(operands.size() == 2);
    if (!getLeafVal)
        return mlir::failure();
    auto location = op->getLoc();

    auto treeMemref = GetTreeMemrefFromTreeOperand(operands[0]);
    auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
    assert (treeMemrefType);

    auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
    auto thresholdType = treeTileType.getThresholdFieldType();
    auto node = operands[1];
    auto nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(location, rewriter.getIndexType(), treeMemref, node);
    
    if (decisionforest::InsertDebugHelpers) {
      rewriter.create<decisionforest::PrintTreeNodeOp>(location, nodeIndex);
    }

    if (treeTileType.getTileSize() == 1) {
      auto loadThresholdOp = rewriter.create<decisionforest::LoadTileThresholdsOp>(location, thresholdType, treeMemref, static_cast<Value>(nodeIndex));
      Value leafValue = loadThresholdOp;
      rewriter.replaceOp(op, static_cast<Value>(leafValue));
    }
    else {
      if (decisionforest::OptimizedSparseRepresentation == false) {
        auto treeMemrefLen = rewriter.create<memref::DimOp>(location, treeMemref, 0);
        auto nodeIndexOutOfBounds = rewriter.create<arith::CmpIOp>(location, arith::CmpIPredicate::slt, nodeIndex, treeMemrefLen);

        auto ifElse = rewriter.create<scf::IfOp>(location, TypeRange{ treeTileType.getThresholdElementType() }, nodeIndexOutOfBounds, true);
        {
          auto thenBuilder = ifElse.getThenBodyBuilder();
          // Load threshold
          // TODO Ideally, this should be a different op for when we deal with tile sizes != 1. We will then need to load 
          // a single threshold value and cast it the trees return type
          auto loadThresholdOp = thenBuilder.create<decisionforest::LoadTileThresholdsOp>(location, thresholdType, treeMemref, static_cast<Value>(nodeIndex));
          Value leafValue = loadThresholdOp;
          
          if (treeTileType.getTileSize() != 1) {
            // if (decisionforest::InsertDebugHelpers) {
            //   InsertPrintVectorOp(rewriter, location, 0, treeTileType.getThresholdElementType().getIntOrFloatBitWidth(), treeTileType.getTileSize(), loadThresholdOp);
            // }
            auto zeroConst = thenBuilder.create<arith::ConstantIntOp>(location, int64_t(0), rewriter.getI32Type());
            auto extractElement = thenBuilder.create<vector::ExtractElementOp>(location, static_cast<Value>(loadThresholdOp), zeroConst);
            leafValue = extractElement;
          }
          thenBuilder.create<scf::YieldOp>(location, leafValue);
        }
        {
          auto elseBuilder = ifElse.getElseBodyBuilder();
          auto leafIndex = elseBuilder.create<arith::SubIOp>(location, nodeIndex, treeMemrefLen);
          auto leavesMemref = GetLeavesMemrefFromTreeOperand(operands[0]);
          auto leafValue = elseBuilder.create<memref::LoadOp>(location, leavesMemref, static_cast<Value>(leafIndex));
          elseBuilder.create<scf::YieldOp>(location, static_cast<Value>(leafValue));
        }
        // auto resultConst = rewriter.create<arith::ConstantFloatOp>(location, APFloat(double(0.5)), rewriter.getF64Type());
        // TODO cast the loaded value to the correct result type of the tree. 
        rewriter.replaceOp(op, static_cast<Value>(ifElse.getResult(0)));
      }
      else {
        auto treeMemrefLen = rewriter.create<memref::DimOp>(location, treeMemref, 0);
        auto leafIndex = rewriter.create<arith::SubIOp>(location, nodeIndex, treeMemrefLen);
        auto leavesMemref = GetLeavesMemrefFromTreeOperand(operands[0]);
        auto leafValue = rewriter.create<memref::LoadOp>(location, leavesMemref, static_cast<Value>(leafIndex));
        
        // auto resultConst = rewriter.create<arith::ConstantFloatOp>(location, APFloat(double(0.5)), rewriter.getF64Type());
        // TODO cast the loaded value to the correct result type of the tree. 
        rewriter.replaceOp(op, static_cast<Value>(leafValue));
      }
    }
    return mlir::success();
  }
};

struct GetLeafTileValueOpLowering : public ConversionPattern {
  GetLeafTileValueOpLowering(MLIRContext *ctx) : ConversionPattern(mlir::decisionforest::GetLeafTileValueOp::getOperationName(), 1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    auto getLeafVal = AssertOpIsOfType<mlir::decisionforest::GetLeafTileValueOp>(op);
    assert(operands.size() == 2);
    if (!getLeafVal)
        return mlir::failure();
    auto location = op->getLoc();

    auto treeMemref = GetTreeMemrefFromTreeOperand(operands[0]);
    auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
    assert (treeMemrefType);

    auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
    auto thresholdType = treeTileType.getThresholdFieldType();
    auto node = operands[1];
    auto nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(location, rewriter.getIndexType(), treeMemref, node);
    if (decisionforest::InsertDebugHelpers) {
      rewriter.create<decisionforest::PrintTreeNodeOp>(location, nodeIndex);
    }

    // Load threshold
    // TODO Ideally, this should be a different op for when we deal with tile sizes != 1. We will then need to load 
    // a single threshold value and cast it the trees return type
    auto loadThresholdOp = rewriter.create<decisionforest::LoadTileThresholdsOp>(location, thresholdType, treeMemref, static_cast<Value>(nodeIndex));
    Value leafValue = loadThresholdOp;
    
    if (treeTileType.getTileSize() != 1) {
      auto thresholdVectorType = thresholdType.cast<VectorType>();
      if (decisionforest::InsertDebugHelpers) {
        Value vectorVal = loadThresholdOp;
        if (!thresholdVectorType.getElementType().isF64()) {
          auto doubleVectorType = mlir::VectorType::get({ treeTileType.getTileSize() }, rewriter.getF64Type());
          vectorVal = rewriter.create<arith::ExtFOp>(location, doubleVectorType, loadThresholdOp);
        }
        InsertPrintVectorOp(rewriter, location, 0, 64, treeTileType.getTileSize(), vectorVal);
      }
      auto zeroConst = rewriter.create<arith::ConstantIntOp>(location, int64_t(0), rewriter.getI32Type());
      auto extractElement = rewriter.create<vector::ExtractElementOp>(location, static_cast<Value>(loadThresholdOp), zeroConst);
      leafValue = extractElement;
    }
    
    // TODO cast the loaded value to the correct result type of the tree. 
    rewriter.replaceOp(op, static_cast<Value>(leafValue));
    return mlir::success();
  }
};

} // anonymous

namespace mlir
{
namespace decisionforest
{

void ClearSparseGlobalMaps() {
  ensembleConstantToMemrefsMap.clear();
  getTreeOperationMap.clear();
}

void PopulateLowerToSparseRepresentationPatterns(RewritePatternSet& patterns) {
    patterns.add<EnsembleConstantOpLowering,
                GetTreeOpLowering,
                GetRootOpLowering,
                IsLeafOpLowering,
                IsLeafTileOpLowering,
                GetTreeClassIdOpLowering,
                TraverseTreeTileOpLowering,
                InterleavedTraverseTreeTileOpLowering,
                GetLeafValueOpLowering,
                GetLeafTileValueOpLowering>(patterns.getContext());
}

} // decisionforest
} // mlir