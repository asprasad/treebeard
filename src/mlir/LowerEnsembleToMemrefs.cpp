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

using namespace mlir;

namespace {

using ThresholdType = double;
using FeatureIndexType = int32_t;

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
    auto functionOp = AddGetEnsembleFunction(op, operands, rewriter);
    auto functionCall = rewriter.create<mlir::CallOp>(location, functionOp);
    rewriter.replaceOp(op, functionCall.getResults()[0]);

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
    auto returnOp = rewriter.create<mlir::ReturnOp>(location, static_cast<Value>(getGlobalOffsets));

    module.push_back(getGlobalMemrefFunc);
  }
  
  // TODO we also need to write the serialized model to a file here.
  void AddGlobalMemref(mlir::ModuleOp module, mlir::decisionforest::EnsembleConstantOp& ensembleConstOp,
                       ConversionPatternRewriter &rewriter, Location location) const {
    mlir::decisionforest::DecisionForestAttribute forestAttribute = ensembleConstOp.forest();
    mlir::decisionforest::DecisionForest<>& forest = forestAttribute.GetDecisionForest();

    SaveAndRestoreInsertionPoint saveAndRestoreInsertPoint(rewriter);
    rewriter.setInsertionPoint(&module.front());

    auto forestType = ensembleConstOp.getResult().getType().cast<decisionforest::TreeEnsembleType>();
    auto thresholdType = rewriter.getF64Type(); // TODO Need to implement forestType.getThresholdType();
    auto featureIndexType = rewriter.getI32Type(); // TODO Need to implement forestType.getFeatureIndexType();
    auto tileSize = 1; // TODO get the tile type from the ensemble type and use that to get tiling
    Type memrefElementType = decisionforest::TiledNumericalNodeType::get(thresholdType, featureIndexType, tileSize);

    std::vector<ThresholdType> thresholds;
    std::vector<FeatureIndexType> featureIndices;
    std::vector<int32_t> treeOffsets;
    forest.GetDenseSerialization(thresholds, featureIndices, treeOffsets);
    
    // TODO write the model values to a file.
    
    auto modelMemrefSize = (int32_t)featureIndices.size();
    auto modelMemrefType = MemRefType::get({modelMemrefSize}, memrefElementType);
    auto modelMemref = rewriter.create<memref::GlobalOp>(
      location, "model",
      /*sym_visibility=*/rewriter.getStringAttr("private"),
      /*type=*/modelMemrefType,
      /*initial_value=*/rewriter.getUnitAttr(),
      /*constant=*/false);
    AddGlobalMemrefGetter(module, "model", modelMemrefType, rewriter, location);
    
    auto offsetSize = (int32_t)treeOffsets.size();
    auto offsetMemrefType = MemRefType::get({offsetSize}, rewriter.getIndexType());
    auto offsetMemref = rewriter.create<memref::GlobalOp>(location, "offsets", rewriter.getStringAttr("private"),
                                                          offsetMemrefType, rewriter.getUnitAttr(), false);
    AddGlobalMemrefGetter(module, "offsets", offsetMemrefType, rewriter, location);
  }
  
  FuncOp AddGetEnsembleFunction(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const {
    mlir::decisionforest::EnsembleConstantOp ensembleConstOp = llvm::dyn_cast<mlir::decisionforest::EnsembleConstantOp>(op);
    SaveAndRestoreInsertionPoint saveAndRestoreInsertPoint(rewriter);

    auto location = op->getLoc();
    auto owningModule = op->getParentOfType<mlir::ModuleOp>();
    assert (owningModule);

    AddGlobalMemref(owningModule, ensembleConstOp, rewriter, location);

    // TODO the stuff below is just a hack to get something to replace the ConstEnsembleOp with. 
    // It will go away soon.
    auto returnType = op->getResults()[0].getType();
    auto functionType = rewriter.getFunctionType(rewriter.getNoneType(), returnType);
    auto functionOp = mlir::FuncOp::create(location, std::string("GetEnsemble"), functionType);
    
    auto &entryBlock = *functionOp.addEntryBlock();
    rewriter.setInsertionPointToStart(&entryBlock);
    
    owningModule.push_back(functionOp);
    return functionOp;
  }
};

struct MidLevelIRToMemrefLoweringPass: public PassWrapper<MidLevelIRToMemrefLoweringPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, tensor::TensorDialect, StandardOpsDialect, scf::SCFDialect>();
  }
  void runOnFunction() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<AffineDialect, memref::MemRefDialect, tensor::TensorDialect, StandardOpsDialect, scf::SCFDialect, decisionforest::DecisionForestDialect>();

    target.addIllegalOp<decisionforest::EnsembleConstantOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<EnsembleConstantOpLowering>(&getContext());

    // loweredSparseConstants.clear();

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
    llvm::errs() << "Lowering to mid level IR failed.\n";
  }
}

} // decisionforest
} // mlir