#include <iostream>
#include "Dialect.h"
// #include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "MemrefTypes.h"
#include "Dialect.h"
#include "TreeTilingUtils.h"

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
  Type modelGlobalType;
  Type offsetGlobaltype;
  std::vector<ThresholdType> thresholds;
  std::vector<FeatureIndexType> featureIndices;
  std::vector<int32_t> offsets;
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
    
    mlir::decisionforest::DecisionForestAttribute forestAttribute = ensembleConstOp.forest();
    mlir::decisionforest::DecisionForest<>& forest = forestAttribute.GetDecisionForest();
    
    // TODO We don't really need to do a full serialization here. We only need to know the size of the serialization.
    std::vector<ThresholdType> thresholds;
    std::vector<FeatureIndexType> featureIndices;
    std::vector<int32_t> treeOffsets;
    forest.GetDenseSerialization(thresholds, featureIndices, treeOffsets);
    
    // TODO the names of the model and offset global should be generated so they're unique for each ensemble constant
    // TODO the getter function names need to be persisted with the actual tree values in the JSON so the runtime can call them. 
    auto memrefTypes = AddGlobalMemrefs(owningModule, ensembleConstOp, rewriter, location, featureIndices.size());
    auto getModelGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<0>(memrefTypes), "model");
    auto getOffsetGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<1>(memrefTypes), "offsets");
    
    EnsembleConstantLoweringInfo info {static_cast<Value>(getModelGlobal), static_cast<Value>(getOffsetGlobal), std::get<0>(memrefTypes), std::get<1>(memrefTypes),
                                       thresholds, featureIndices, treeOffsets};
    ensembleConstantToMemrefsMap[op] = info;
    
    // rewriter.replaceOp(op, static_cast<Value>(getModelGlobal));
    rewriter.eraseOp(op);

    return mlir::success();
  }

  void AddGlobalMemrefGetter(mlir::ModuleOp module, std::string globalName, Type memrefType, ConversionPatternRewriter &rewriter, Location location) const {
    SaveAndRestoreInsertionPoint saveAndRestoreEntryPoint(rewriter);
    auto getMemrefFuncType = rewriter.getFunctionType(rewriter.getNoneType(), memrefType);
    std::string funcName = "Get_" + globalName;
    NamedAttribute visibilityAttribute{module.sym_visibilityAttrName(), rewriter.getStringAttr("public")};
    auto getGlobalMemrefFunc = FuncOp::create(location, funcName, getMemrefFuncType, ArrayRef<NamedAttribute>(visibilityAttribute));
    auto &entryBlock = *getGlobalMemrefFunc.addEntryBlock();
    rewriter.setInsertionPointToStart(&entryBlock);

    auto getGlobalOffsets = rewriter.create<memref::GetGlobalOp>(location, memrefType, globalName);
    rewriter.create<mlir::ReturnOp>(location, static_cast<Value>(getGlobalOffsets));

    module.push_back(getGlobalMemrefFunc);
  }
  
  // TODO we also need to write the serialized model to a file here.
  std::tuple<Type, Type> AddGlobalMemrefs(mlir::ModuleOp module, mlir::decisionforest::EnsembleConstantOp& ensembleConstOp,
                                          ConversionPatternRewriter &rewriter, Location location, int32_t modelMemrefSize) const {
    mlir::decisionforest::DecisionForestAttribute forestAttribute = ensembleConstOp.forest();
    mlir::decisionforest::DecisionForest<>& forest = forestAttribute.GetDecisionForest();

    SaveAndRestoreInsertionPoint saveAndRestoreInsertPoint(rewriter);
    rewriter.setInsertionPoint(&module.front());

    auto forestType = ensembleConstOp.getResult().getType().cast<decisionforest::TreeEnsembleType>();
    assert (forestType.doAllTreesHaveSameType()); // There is still an assumption here that all trees have the same type
    auto treeType = forestType.getTreeType(0).cast<decisionforest::TreeType>();

    auto thresholdType = treeType.getThresholdType();
    auto featureIndexType = treeType.getFeatureIndexType(); 
    auto tileSize = treeType.getTilingDescriptor().MaxTileSize();
    assert (tileSize == 1);
    Type memrefElementType = decisionforest::TiledNumericalNodeType::get(thresholdType, featureIndexType, tileSize);

    PersistDecisionForest(forest, forestType);
    
    // auto modelMemrefSize = (int32_t)featureIndices.size();
    auto modelMemrefType = MemRefType::get({modelMemrefSize}, memrefElementType);
    rewriter.create<memref::GlobalOp>(location, "model",
                                      /*sym_visibility=*/rewriter.getStringAttr("private"),
                                      /*type=*/modelMemrefType,
                                      /*initial_value=*/rewriter.getUnitAttr(),
                                      /*constant=*/false);
    AddGlobalMemrefGetter(module, "model", modelMemrefType, rewriter, location);
    
    auto offsetSize = (int32_t)forest.NumTrees();
    auto offsetMemrefType = MemRefType::get({offsetSize}, rewriter.getIndexType());
    rewriter.create<memref::GlobalOp>(location, "offsets", rewriter.getStringAttr("private"),
                                      offsetMemrefType, rewriter.getUnitAttr(), false);
    AddGlobalMemrefGetter(module, "offsets", offsetMemrefType, rewriter, location);

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
    int32_t treeLength = 100; // TODO Need to put this into the map too
    auto treeMemref = rewriter.create<memref::SubViewOp>(location, ensembleInfo.modelGlobal, ArrayRef<OpFoldResult>({static_cast<Value>(modelMemrefIndex)}),
                                                         ArrayRef<OpFoldResult>({rewriter.getIndexAttr(treeLength)}), ArrayRef<OpFoldResult>({rewriter.getIndexAttr(1)}));

    getTreeOperationMap[op] = static_cast<Value>(treeMemref);

    rewriter.replaceOp(op, static_cast<Value>(treeMemref));
    // rewriter.eraseOp(op);

    return mlir::success();
  }
};

Value GetTreeMemrefFromTreeOperand(Value treeValue) {
  // auto getTreeOp = treeValue.getDefiningOp();
  // AssertOpIsOfType<mlir::decisionforest::GetTreeFromEnsembleOp>(getTreeOp);
  // auto getTreeOperationMapIter = getTreeOperationMap.find(getTreeOp);
  // assert(getTreeOperationMapIter != getTreeOperationMap.end());
  // auto treeMemref = getTreeOperationMapIter->second;
  auto treeMemref = treeValue; // TODO This is just until we remove the "replaceOp" in the GetTree lowering rule
  return treeMemref;
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
    auto featureIndexType = treeTileType.getIndexType();
    assert (treeTileType.getTileSize() == 1);

    auto loadThresholdOp = rewriter.create<decisionforest::LoadTileFeatureIndicesOp>(location, featureIndexType, treeMemref, static_cast<Value>(nodeIndex));
    auto minusOneConstant = rewriter.create<ConstantIntOp>(location, int64_t(-1), featureIndexType);
    auto comparison = rewriter.create<CmpIOp>(location, mlir::CmpIPredicate::eq, static_cast<Value>(loadThresholdOp), static_cast<Value>(minusOneConstant));
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
    auto location = op->getLoc();

    auto treeMemref = GetTreeMemrefFromTreeOperand(operands[0]);
    auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
    assert (treeMemrefType);

    auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
    auto featureIndexType = treeTileType.getIndexType();
    auto thresholdType = treeTileType.getThresholdType();
    // Assert tile size is 1
    assert (treeTileType.getTileSize() == 1);

    auto node = operands[1];
    auto nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(location, rewriter.getIndexType(), treeMemref, node);
    // Load threshold
    auto loadThresholdOp = rewriter.create<decisionforest::LoadTileThresholdsOp>(location, thresholdType, treeMemref, static_cast<Value>(nodeIndex));
    // Load feature index
    auto loadFeatureIndexOp = rewriter.create<decisionforest::LoadTileFeatureIndicesOp>(location, featureIndexType, treeMemref, static_cast<Value>(nodeIndex));
    // Load feature value
    auto rowTensor = operands[2];
    auto rowTensorType = rowTensor.getType().cast<RankedTensorType>();
    auto rowIndex = rewriter.create<IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(loadFeatureIndexOp));
    auto feature = rewriter.create<tensor::ExtractOp>(location, rowTensorType.getElementType(), rowTensor, static_cast<Value>(rowIndex));
    // 
    // result = Compare
    // TODO we need a cast here to make sure the threshold and the row element are the same type. The op expects both operands to be the same type.
    auto comparison = rewriter.create<CmpFOp>(location, mlir::CmpFPredicate::ULE, static_cast<Value>(feature), static_cast<Value>(loadThresholdOp));
    
    // index = 2*index + result
    auto twoConstant = rewriter.create<ConstantIndexOp>(location, 2);
    auto twoTimesIndex = rewriter.create<MulIOp>(location, rewriter.getIndexType(), static_cast<Value>(nodeIndex), static_cast<Value>(twoConstant));
    auto comparisonResultIndex = rewriter.create<IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(comparison));
    auto newIndex = rewriter.create<AddIOp>(location, rewriter.getIndexType(), static_cast<Value>(twoTimesIndex), static_cast<Value>(comparisonResultIndex));
    
    // node = indexToNode(index)
    auto newNode = rewriter.create<decisionforest::IndexToNodeOp>(location, traverseTileOp.getResult().getType(), treeMemref, static_cast<Value>(newIndex));

    rewriter.replaceOp(op, static_cast<Value>(newNode));

    return mlir::success();
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
    auto thresholdType = treeTileType.getThresholdType();
    // Assert tile size is 1
    assert (treeTileType.getTileSize() == 1);

    auto node = operands[1];
    auto nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(location, rewriter.getIndexType(), treeMemref, node);
    // Load threshold
    // TODO Ideally, this should be a different op for when we deal with tile sizes != 1. We will then need to load 
    // a single threshold value and cast it the trees return type
    auto loadThresholdOp = rewriter.create<decisionforest::LoadTileThresholdsOp>(location, thresholdType, treeMemref, static_cast<Value>(nodeIndex));

    rewriter.replaceOp(op, static_cast<Value>(loadThresholdOp));
    return mlir::success();
  }
};

struct MidLevelIRToMemrefLoweringPass: public PassWrapper<MidLevelIRToMemrefLoweringPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, tensor::TensorDialect, StandardOpsDialect, scf::SCFDialect>();
  }
  void runOnFunction() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<AffineDialect, memref::MemRefDialect, tensor::TensorDialect, StandardOpsDialect, scf::SCFDialect, decisionforest::DecisionForestDialect>();

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