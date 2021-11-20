#include <iostream>
#include "Dialect.h"
// #include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "MemrefTypes.h"
#include "Dialect.h"
#include "TreeTilingUtils.h"
#include "TiledTree.h"

/*
Plan and issues
* We can add a memref argument to the inference function, but modifying
  the function type while lowering doesn't seem easy. We don't know the 
  type of the model memref until all optimizations have run -- so we 
  can't just add an argument while we create the function in HIR. 
* We could just clone our function into a function with a different 
  signature though
* [Problem] If we add a single memref to represent the model, we're
  we're assuming that all trees are tiled identically! Is this an
  assumption we should bake into the code gen?
* [Selected] [Option] What if we add a global memref to the module and a function 
  that just returns this memref? We can the populate it in C code 
  and the inference function can use the global. We could just have
  multiple global memrefs if we want different tiling for different
  trees. 
  - [Problem] How would the IR pick between these memrefs though? We
    would need something like an array of memrefs so we can pick the
    right one based on the tree index. Also, each of these memrefs
    would have a different type.
    [A] This may not be a problem. We need to statically know the 
    type of the tree (which includes tree tiling) to be able to generate
    code. So we should know which memref to access if there is one 
    memref per unique tree type. 
*/

/*
  trees = memref<Tiles, ?>
  offsets = memref<int32>

  all rows in batch
    all trees in forest
      tree = trees + offset[treeIndex] // memref.subview
      n = 0
      while (!IsLeaf(n))
        thresholds = LoadTileThresholds(tree, n)
        indices  = LoadTileFeatureIndices(tree, n)
        features = gather(data[i], indices)
        outcome = features < thresholds // type bool if tileSize = 1
        // Assuming TileSize == 1
        n = 2*n + 1 + outcome

*/
using namespace mlir;

namespace {

using ThresholdType = double;
using FeatureIndexType = int32_t;

struct EnsembleConstantLoweringInfo {
  Value modelGlobal;
  Value offsetGlobal;
  Value lengthGlobal;
  Value lutGlobal;
  Type modelGlobalType;
  Type offsetGlobaltype;
  Type lengthGlobalType;
  Type lutGlobalType;
};

// Maps an ensemble constant operation to a model memref and an offsets memref
std::map<Operation*, EnsembleConstantLoweringInfo> ensembleConstantToMemrefsMap;
// Maps a GetTree operation to a memref that represents the tree once the ensemble constant has been replaced
std::map<Operation*, Value> getTreeOperationMap;
// Maps a GetRoot operation to the integer constant (=0) that will represent the index into the tree memref
std::map<Operation*, Value> getRootOperationMap;

template<typename T>
T AssertOpIsOfType(Operation* operation) {
  T typedOp = llvm::dyn_cast<T>(operation);
  assert(typedOp);
  return typedOp;
}

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
  auto tileSizeConst = rewriter.create<ConstantIntOp>(location, tileSize, rewriter.getI32Type());
  auto kindConst = rewriter.create<ConstantIntOp>(location, kind, rewriter.getI32Type());
  auto bitWidthConst = rewriter.create<ConstantIntOp>(location, bitWidth, rewriter.getI32Type());
  std::vector<Value> vectorValues;
  for (int32_t i=0; i<tileSize ; ++i) {
    auto ithValue = rewriter.create<vector::ExtractElementOp>(location, vectorValue, int64_t(i));
    vectorValues.push_back(ithValue);
  }
  rewriter.create<decisionforest::PrintVectorOp>(location, kindConst, bitWidthConst, tileSizeConst, ValueRange(vectorValues));
}

Value CreateZeroVectorFPConst(ConversionPatternRewriter &rewriter, Location location, Type fpType, int32_t tileSize) {
  Value zeroConst;
  auto vectorType = VectorType::get(tileSize, fpType);
  if (fpType.isa<mlir::Float64Type>())
    zeroConst = rewriter.create<ConstantFloatOp>(location, llvm::APFloat(0.0), fpType.cast<FloatType>());
  else if(fpType.isa<mlir::Float32Type>())
    zeroConst = rewriter.create<ConstantFloatOp>(location, llvm::APFloat((float)0.0), fpType.cast<FloatType>());
  else
    assert(false && "Unsupported floating point type");
  auto vectorValue = rewriter.create<vector::BroadcastOp>(location, vectorType, zeroConst);
  return vectorValue;
}

Value CreateZeroVectorIntConst(ConversionPatternRewriter &rewriter, Location location, Type intType, int32_t tileSize) {
  Value zeroConst = rewriter.create<ConstantIntOp>(location, 0, intType);
  auto vectorType = VectorType::get(tileSize, intType);
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
    auto memrefTypes = AddGlobalMemrefs(owningModule, ensembleConstOp, rewriter, location, modelMemrefName, offsetMemrefName, lengthMemrefName);
    AddModelMemrefInitFunction(owningModule, modelMemrefName, std::get<0>(memrefTypes).cast<MemRefType>(), rewriter, location);
    auto getModelGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<0>(memrefTypes), modelMemrefName);
    auto getOffsetGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<1>(memrefTypes), offsetMemrefName);
    auto getLengthGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<1>(memrefTypes), lengthMemrefName);
    
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

    EnsembleConstantLoweringInfo info {static_cast<Value>(getModelGlobal), static_cast<Value>(getOffsetGlobal), static_cast<Value>(getLengthGlobal), getLUT,
                                       std::get<0>(memrefTypes), std::get<1>(memrefTypes), std::get<1>(memrefTypes), lookUpTableMemrefType};
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
                                      /*constant=*/false);
    AddGlobalMemrefGetter(module, lookupTableMemrefName, lutMemrefType, rewriter, location);

    return lutMemrefType;
  }
  
  void AddGlobalMemrefGetter(mlir::ModuleOp module, std::string globalName, Type memrefType, ConversionPatternRewriter &rewriter, Location location) const {
    SaveAndRestoreInsertionPoint saveAndRestoreEntryPoint(rewriter);
    auto getMemrefFuncType = rewriter.getFunctionType(TypeRange({}), memrefType);
    std::string funcName = "Get_" + globalName;
    NamedAttribute visibilityAttribute{module.sym_visibilityAttrName(), rewriter.getStringAttr("public")};
    auto getGlobalMemrefFunc = FuncOp::create(location, funcName, getMemrefFuncType, ArrayRef<NamedAttribute>(visibilityAttribute));
    auto &entryBlock = *getGlobalMemrefFunc.addEntryBlock();
    rewriter.setInsertionPointToStart(&entryBlock);

    auto getGlobalOffsets = rewriter.create<memref::GetGlobalOp>(location, memrefType, globalName);
    rewriter.create<mlir::ReturnOp>(location, static_cast<Value>(getGlobalOffsets));

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
    auto tileShapeIDArgType = MemRefType::get(memrefType.getShape(), rewriter.getI32Type());
    auto getMemrefFuncType = rewriter.getFunctionType(TypeRange{thresholdArgType, indexArgType, tileShapeIDArgType}, rewriter.getI32Type());
    std::string funcName = "Init_" + globalName;
    NamedAttribute visibilityAttribute{module.sym_visibilityAttrName(), rewriter.getStringAttr("public")};
    auto initModelMemrefFunc = FuncOp::create(location, funcName, getMemrefFuncType, ArrayRef<NamedAttribute>(visibilityAttribute));
    auto &entryBlock = *initModelMemrefFunc.addEntryBlock();
    rewriter.setInsertionPointToStart(&entryBlock);

    // for tileIndex = 0 : len
    auto getGlobalMemref = rewriter.create<memref::GetGlobalOp>(location, memrefType, globalName);
    auto zeroIndexConst = rewriter.create<ConstantIndexOp>(location, 0);
    auto oneIndexConst = rewriter.create<ConstantIndexOp>(location, 1);
    auto lenIndexConst = rewriter.create<ConstantIndexOp>(location, memrefType.getShape()[0]);
    auto forLoop = rewriter.create<scf::ForOp>(location, zeroIndexConst, lenIndexConst, oneIndexConst);
    auto tileIndex = forLoop.getInductionVar();
    rewriter.setInsertionPointToStart(forLoop.getBody());

    // index = tileSize * tileIndex
    auto tileSizeConst = rewriter.create<ConstantIndexOp>(location, tileSize);
    auto tileSizeTimesi = rewriter.create<MulIOp>(location, tileIndex, tileSizeConst);
    
    if (tileSize > 1) {
      auto thresholdVec = CreateZeroVectorFPConst(rewriter, location, modelMemrefElementType.getThresholdElementType(), tileSize);
      auto indexVec = CreateZeroVectorIntConst(rewriter, location, modelMemrefElementType.getIndexElementType(), tileSize);

      // Load from index to index + (tileSize - 1) into a vector
      for (int32_t j = 0 ; j<tileSize ; ++j) {
        auto offset = rewriter.create<ConstantIndexOp>(location, j);
        auto index =  rewriter.create<AddIOp>(location, tileSizeTimesi, offset);
        auto thresholdVal = rewriter.create<memref::LoadOp>(location, entryBlock.getArgument(0), static_cast<Value>(index));
        thresholdVec = rewriter.create<vector::InsertElementOp>(location, thresholdVal, thresholdVec, j);
        auto indexVal = rewriter.create<memref::LoadOp>(location, entryBlock.getArgument(1), static_cast<Value>(index));
        indexVec = rewriter.create<vector::InsertElementOp>(location, indexVal, indexVec, j);
      }
      auto tileShapeID = rewriter.create<memref::LoadOp>(location, entryBlock.getArgument(2), tileIndex);
      rewriter.create<decisionforest::InitTileOp>(location, getGlobalMemref, tileIndex, thresholdVec, indexVec, tileShapeID);
    }
    else {
      // Load from index to index + (tileSize - 1) into a vector
      auto thresholdVal = rewriter.create<memref::LoadOp>(location, entryBlock.getArgument(0), static_cast<Value>(tileIndex));
      auto indexVal = rewriter.create<memref::LoadOp>(location, entryBlock.getArgument(1), static_cast<Value>(tileIndex));
      // TODO check how tileShapeID vector is created when tileSize = 1
      auto tileShapeID = rewriter.create<ConstantIntOp>(location, 0, rewriter.getI32Type());
      rewriter.create<decisionforest::InitTileOp>(location, getGlobalMemref, tileIndex, thresholdVal, indexVal, tileShapeID);
    }
    rewriter.setInsertionPointAfter(forLoop);
    auto retVal = rewriter.create<ConstantIntOp>(location, 0, rewriter.getI32Type());
    rewriter.create<mlir::ReturnOp>(location, static_cast<Value>(retVal));
    module.push_back(initModelMemrefFunc);
  }

  std::tuple<Type, Type> AddGlobalMemrefs(mlir::ModuleOp module, mlir::decisionforest::EnsembleConstantOp& ensembleConstOp,
                                          ConversionPatternRewriter &rewriter, Location location,
                                          const std::string& modelMemrefName, const std::string& offsetMemrefName, const std::string& lengthMemrefName) const {
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
    // assert (tileSize == 1);
    Type memrefElementType = decisionforest::TiledNumericalNodeType::get(thresholdType, featureIndexType, tileSize);

    PersistDecisionForest(forest, forestType);
    
    auto modelMemrefSize = decisionforest::GetTotalNumberOfTiles();
    auto modelMemrefType = MemRefType::get({modelMemrefSize}, memrefElementType);
    rewriter.create<memref::GlobalOp>(location, modelMemrefName,
                                      /*sym_visibility=*/rewriter.getStringAttr("private"),
                                      /*type=*/modelMemrefType,
                                      /*initial_value=*/rewriter.getUnitAttr(),
                                      /*constant=*/false);
    AddGlobalMemrefGetter(module, modelMemrefName, modelMemrefType, rewriter, location);
    
    auto offsetSize = (int32_t)forest.NumTrees();
    auto offsetMemrefType = MemRefType::get({offsetSize}, rewriter.getIndexType());
    rewriter.create<memref::GlobalOp>(location, offsetMemrefName, rewriter.getStringAttr("private"),
                                      offsetMemrefType, rewriter.getUnitAttr(), false);
    AddGlobalMemrefGetter(module, offsetMemrefName, offsetMemrefType, rewriter, location);
    
    rewriter.create<memref::GlobalOp>(location, lengthMemrefName, rewriter.getStringAttr("private"),
                                      offsetMemrefType, rewriter.getUnitAttr(), false);
    AddGlobalMemrefGetter(module, lengthMemrefName, offsetMemrefType, rewriter, location);

    return std::make_tuple(modelMemrefType, offsetMemrefType);
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
    // TODO There is no way to infer the length of the tree memref here (since this could potentially be multiple trees.)
    // The length of the memref needs to be a runtime value with the length of the tree. This may need an additional global.
    auto treeLength = rewriter.create<memref::LoadOp>(location, ensembleInfo.lengthGlobal, treeIndex);; // TODO Need to put this into the map too
    auto treeMemref = rewriter.create<memref::SubViewOp>(location, ensembleInfo.modelGlobal, ArrayRef<OpFoldResult>({static_cast<Value>(modelMemrefIndex)}),
                                                         ArrayRef<OpFoldResult>({static_cast<Value>(treeLength)}), ArrayRef<OpFoldResult>({rewriter.getIndexAttr(1)}));

    if (decisionforest::InsertDebugHelpers) {
      rewriter.create<decisionforest::PrintTreeToDOTFileOp>(location, treeMemref, treeIndex);
    }
    getTreeOperationMap[op] = static_cast<Value>(treeMemref);

    rewriter.eraseOp(op);

    return mlir::success();
  }
};

Value GetTreeMemrefFromTreeOperand(Value treeValue) {
  auto getTreeOp = treeValue.getDefiningOp();
  AssertOpIsOfType<mlir::decisionforest::GetTreeFromEnsembleOp>(getTreeOp);
  auto getTreeOperationMapIter = getTreeOperationMap.find(getTreeOp);
  assert(getTreeOperationMapIter != getTreeOperationMap.end());
  auto treeMemref = getTreeOperationMapIter->second;
  return treeMemref;
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
    auto nodeIndexConst = rewriter.create<ConstantIndexOp>(op->getLoc(), 0);

    auto treeMemref = GetTreeMemrefFromTreeOperand(operands[0]);
    auto nodeType = getRootOp.getResult().getType();
    auto node = rewriter.create<decisionforest::IndexToNodeOp>(op->getLoc(), nodeType, treeMemref, static_cast<Value>(nodeIndexConst));
    rewriter.replaceOp(op, static_cast<Value>(node));
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
      auto extractFirstElement = rewriter.create<vector::ExtractElementOp>(location, static_cast<Value>(loadFeatureIndexOp), int64_t(0));
      featureIndexValue = extractFirstElement;
    }
    auto minusOneConstant = rewriter.create<ConstantIntOp>(location, int64_t(-1), treeTileType.getIndexElementType());
    auto comparison = rewriter.create<CmpIOp>(location, mlir::CmpIPredicate::eq, featureIndexValue, static_cast<Value>(minusOneConstant));
    
    if (decisionforest::InsertDebugHelpers) {
      Value outcome = rewriter.create<mlir::ZeroExtendIOp>(location, rewriter.getI32Type(), static_cast<Value>(comparison));
      rewriter.create<decisionforest::PrintIsLeafOp>(location, nodeIndex, featureIndexValue, outcome);
    }
    rewriter.replaceOp(op, static_cast<Value>(comparison));

    return mlir::success();
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
    if (treeTileType.getTileSize() == 1)
      LowerOpTileSize1(op, operands, rewriter);
    else
      LowerOpForVectorTile(op, operands, rewriter);

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
    // Load feature value
    auto rowMemref = operands[2];
    auto rowMemrefType = rowMemref.getType().cast<MemRefType>();
    auto rowIndex = rewriter.create<IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(loadFeatureIndexOp));
    auto zeroIndex = rewriter.create<ConstantIndexOp>(location, 0);
    auto feature = rewriter.create<memref::LoadOp>(location, rowMemrefType.getElementType(), rowMemref,
                                                   ValueRange({static_cast<Value>(zeroIndex), static_cast<Value>(rowIndex)}));

    if(decisionforest::InsertDebugHelpers) {
      rewriter.create<decisionforest::PrintComparisonOp>(location, feature, loadThresholdOp, loadFeatureIndexOp);
    }

    // result = Compare
    // TODO we need a cast here to make sure the threshold and the row element are the same type. The op expects both operands to be the same type.
    auto comparison = rewriter.create<CmpFOp>(location,  mlir::CmpFPredicate::UGT, static_cast<Value>(feature), static_cast<Value>(loadThresholdOp));
    auto comparisonUnsigned = rewriter.create<ZeroExtendIOp>(location, rewriter.getI32Type(), static_cast<Value>(comparison));

    // index = 2*index + 1 + result
    auto oneConstant = rewriter.create<ConstantIndexOp>(location, 1);
    auto twoConstant = rewriter.create<ConstantIndexOp>(location, 2);
    auto twoTimesIndex = rewriter.create<MulIOp>(location, rewriter.getIndexType(), static_cast<Value>(nodeIndex), static_cast<Value>(twoConstant));
    auto twoTimesIndexPlus1 = rewriter.create<AddIOp>(location, rewriter.getIndexType(), static_cast<Value>(twoTimesIndex), static_cast<Value>(oneConstant));
    auto comparisonResultIndex = rewriter.create<IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(comparisonUnsigned));
    auto newIndex = rewriter.create<AddIOp>(location, rewriter.getIndexType(), static_cast<Value>(twoTimesIndexPlus1), static_cast<Value>(comparisonResultIndex));
    
    // node = indexToNode(index)
    auto newNode = rewriter.create<decisionforest::IndexToNodeOp>(location, traverseTileOp.getResult().getType(), treeMemref, static_cast<Value>(newIndex));

    rewriter.replaceOp(op, static_cast<Value>(newNode));
  }

  Value ReduceComparisonResultVectorToInt(Value comparisonResult, int32_t tileSize, ConversionPatternRewriter &rewriter, Location location) const {
    auto i32VectorType = VectorType::get(tileSize, rewriter.getI32Type());
    auto comparisonExtended = rewriter.create<ZeroExtendIOp>(location, i32VectorType, comparisonResult);

    auto zeroI32Const = rewriter.create<ConstantIntOp>(location, int64_t(0), rewriter.getI32Type());
    auto shiftVector = static_cast<Value>(rewriter.create<vector::BroadcastOp>(location, i32VectorType, zeroI32Const));
    for (int32_t shift=0, pos=tileSize-1 ; shift<tileSize; ++shift, --pos) {
      auto shiftValConst = rewriter.create<ConstantIntOp>(location, int64_t(shift), rewriter.getI32Type());
      shiftVector = rewriter.create<vector::InsertOp>(location, static_cast<Value>(shiftValConst), 
                                                      static_cast<Value>(shiftVector), ArrayRef<int64_t>({ pos }));
    }

    auto leftShift = rewriter.create<ShiftLeftOp>(location, i32VectorType, comparisonExtended, shiftVector);
    auto kind = rewriter.getStringAttr("add");
    auto sum = rewriter.create<vector::ReductionOp>(location, rewriter.getI32Type(), kind, static_cast<Value>(leftShift), ValueRange{ });
    auto index = rewriter.create<IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(sum));
    return index;
  }

  Value ReduceComparisonResultVectorToInt_Bitcast(Value comparisonResult, int32_t tileSize, ConversionPatternRewriter &rewriter, Location location) const {
    auto bitcastVectorType = VectorType::get(1, rewriter.getIntegerType(tileSize));
    auto bitcastOp = rewriter.create<vector::BitCastOp>(location, bitcastVectorType, comparisonResult);
    auto integerResult = rewriter.create<vector::ExtractElementOp>(location, static_cast<Value>(bitcastOp), 0);
    auto zeroExtend = rewriter.create<ZeroExtendIOp>(location, integerResult, rewriter.getI64Type()); 
    auto index = rewriter.create<IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(zeroExtend));
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
      InsertPrintVectorOp(rewriter, location, 0 /*fp kind*/, thresholdVectorType.getElementType().getIntOrFloatBitWidth(), 
                          tileSize, static_cast<Value>(loadThresholdOp));
    }
    // Load feature index
    auto loadFeatureIndexOp = rewriter.create<decisionforest::LoadTileFeatureIndicesOp>(location, featureIndexType, treeMemref, static_cast<Value>(nodeIndex));
    if (decisionforest::InsertDebugHelpers) {
      InsertPrintVectorOp(rewriter, location, 1 /*int kind*/, featureIndexVectorType.getElementType().getIntOrFloatBitWidth(), 
                          tileSize, static_cast<Value>(loadFeatureIndexOp));
    }

    // Load the tile shape
    auto loadTileShapeOp = rewriter.create<decisionforest::LoadTileShapeOp>(location, rewriter.getI32Type(), treeMemref, static_cast<Value>(nodeIndex));
    auto tileShapeIndex = rewriter.create<IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(loadTileShapeOp));

    // index = (tileSize+1)*index + 1 + childIndex
    auto oneConstant = rewriter.create<ConstantIndexOp>(location, 1);
    auto tileSizeConstant = rewriter.create<ConstantIndexOp>(location, tileSize+1);
    auto tileSizeTimesIndex = rewriter.create<MulIOp>(location, rewriter.getIndexType(), static_cast<Value>(nodeIndex), static_cast<Value>(tileSizeConstant));
    auto tileSizeTimesIndexPlus1 = rewriter.create<AddIOp>(location, rewriter.getIndexType(), static_cast<Value>(tileSizeTimesIndex), static_cast<Value>(oneConstant));
    
    // Load feature value
    auto rowMemref = operands[2];
    auto rowMemrefType = rowMemref.getType().cast<MemRefType>();
    auto vectorIndexType = VectorType::get({ tileSize }, rewriter.getIndexType());
    auto rowIndex = rewriter.create<IndexCastOp>(location, vectorIndexType, static_cast<Value>(loadFeatureIndexOp));
    auto zeroIndex = rewriter.create<ConstantIndexOp>(location, 0);
    // auto zeroIndexVector = rewriter.create<vector::BroadcastOp>(location, vectorIndexType, zeroIndex);

    auto featuresVectorType = VectorType::get({ tileSize }, rowMemrefType.getElementType());
    auto oneI1Const = rewriter.create<ConstantIntOp>(location, 1, rewriter.getI1Type());
    auto i1VectorType = VectorType::get(tileSize, rewriter.getI1Type());
    auto mask = rewriter.create<vector::BroadcastOp>(location, i1VectorType, oneI1Const);

    Value zeroPassThruConst;
    if (rowMemrefType.getElementType().isa<mlir::Float64Type>())
      zeroPassThruConst = rewriter.create<ConstantFloatOp>(location, llvm::APFloat(0.0), rowMemrefType.getElementType().cast<FloatType>());
    else if(rowMemrefType.getElementType().isa<mlir::Float32Type>())
      zeroPassThruConst = rewriter.create<ConstantFloatOp>(location, llvm::APFloat((float)0.0), rowMemrefType.getElementType().cast<FloatType>());
    else
      assert(false && "Unsupported floating point type");
    auto zeroPassThruVector = rewriter.create<vector::BroadcastOp>(location, featuresVectorType, zeroPassThruConst);
    
    auto features = rewriter.create<vector::GatherOp>(location, featuresVectorType, rowMemref,
                                                      ValueRange({static_cast<Value>(zeroIndex), static_cast<Value>(zeroIndex)}),
                                                      rowIndex, mask, zeroPassThruVector);

    if (decisionforest::InsertDebugHelpers) {
      InsertPrintVectorOp(rewriter, location, 0 /*fp kind*/, featuresVectorType.getElementType().getIntOrFloatBitWidth(), 
                          tileSize, static_cast<Value>(features));
    }

    // TODO This needs a different print routine!
    // if(decisionforest::InsertDebugHelpers) {
    //   rewriter.create<decisionforest::PrintComparisonOp>(location, feature, loadThresholdOp, loadFeatureIndexOp);
    // }

    // result = Compare
    // TODO we need a cast here to make sure the threshold and the row element are the same type. The op expects both operands to be the same type.
    auto comparison = rewriter.create<CmpFOp>(location,  mlir::CmpFPredicate::ULE, static_cast<Value>(features), static_cast<Value>(loadThresholdOp));
    Value comparisonIndex;
    if (decisionforest::UseBitcastForComparisonOutcome)
      comparisonIndex = ReduceComparisonResultVectorToInt_Bitcast(comparison, tileSize, rewriter, location);
    else
      comparisonIndex = ReduceComparisonResultVectorToInt(comparison, tileSize, rewriter, location);


    // Load the child index from the LUT
    auto lutValue = GetLUTFromTreeOperand(operands[0]);
    auto childIndexInt = rewriter.create<memref::LoadOp>(location, lutValue, ValueRange{tileShapeIndex, comparisonIndex});
    auto childIndex = rewriter.create<IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(childIndexInt));

    auto newIndex = rewriter.create<AddIOp>(location, rewriter.getIndexType(), static_cast<Value>(tileSizeTimesIndexPlus1), static_cast<Value>(childIndex));
    
    // node = indexToNode(index)
    auto newNode = rewriter.create<decisionforest::IndexToNodeOp>(location, traverseTileOp.getResult().getType(), treeMemref, static_cast<Value>(newIndex));
    rewriter.replaceOp(op, static_cast<Value>(newNode));
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

    // Load threshold
    // TODO Ideally, this should be a different op for when we deal with tile sizes != 1. We will then need to load 
    // a single threshold value and cast it the trees return type
    auto loadThresholdOp = rewriter.create<decisionforest::LoadTileThresholdsOp>(location, thresholdType, treeMemref, static_cast<Value>(nodeIndex));
    Value leafValue = loadThresholdOp;
    
    if (treeTileType.getTileSize() != 1) {
      if (decisionforest::InsertDebugHelpers) {
        InsertPrintVectorOp(rewriter, location, 0, treeTileType.getThresholdElementType().getIntOrFloatBitWidth(), treeTileType.getTileSize(), loadThresholdOp);
      }
      auto extractElement = rewriter.create<vector::ExtractElementOp>(location, static_cast<Value>(loadThresholdOp), int64_t(0));
      leafValue = extractElement;
    }
    
    // TODO cast the loaded value to the correct result type of the tree. 
    rewriter.replaceOp(op, static_cast<Value>(leafValue));
    return mlir::success();
  }
};

struct MidLevelIRToMemrefLoweringPass: public PassWrapper<MidLevelIRToMemrefLoweringPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, StandardOpsDialect, scf::SCFDialect>();
  }
  void runOnFunction() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<AffineDialect, memref::MemRefDialect, StandardOpsDialect, 
                           scf::SCFDialect, decisionforest::DecisionForestDialect, vector::VectorDialect,
                           math::MathDialect>();

    target.addIllegalOp<decisionforest::EnsembleConstantOp,
                        decisionforest::GetTreeFromEnsembleOp,
                        decisionforest::GetRootOp,
                        decisionforest::IsLeafOp,
                        decisionforest::TraverseTreeTileOp,
                        decisionforest::GetLeafValueOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<EnsembleConstantOpLowering,
                 GetTreeOpLowering,
                 GetRootOpLowering,
                 IsLeafOpLowering,
                 TraverseTreeTileOpLowering,
                 GetLeafValueOpLowering>(&getContext());

    if (failed(applyPartialConversion(getFunction(), target, std::move(patterns))))
        signalPassFailure();
  }
};
}

namespace mlir
{
namespace decisionforest
{
void LowerEnsembleToMemrefs(mlir::MLIRContext& context, mlir::ModuleOp module) {
  // Lower from high-level IR to mid-level IR
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  optPM.addPass(std::make_unique<MidLevelIRToMemrefLoweringPass>());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to memrefs failed.\n";
  }
}

} // decisionforest
} // mlir