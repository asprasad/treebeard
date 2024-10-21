#ifdef TREEBEARD_GPU_SUPPORT
#include "Dialect.h"
#include <iostream>
#include <memory>
#include <vector>
// #include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

#include "CodeGenStateMachine.h"
#include "Dialect.h"
#include "LIRLoweringHelpers.h"
#include "MemrefTypes.h"
#include "ModelSerializers.h"
#include "OpLoweringUtils.h"
#include "Representations.h"
#include "TiledTree.h"
#include "TraverseTreeTileOpLowering.h"
#include "TreeTilingUtils.h"
#include "schedule.h"

using namespace mlir;

namespace mlir {
namespace decisionforest {

struct TraverseToCooperativeTraverseTreeTileOp : public ConversionPattern {
  std::unique_ptr<std::map<Operation *, std::vector<Value>>> m_subviewMap;
  TraverseToCooperativeTraverseTreeTileOp(MLIRContext *ctx)
      : ConversionPattern(
            mlir::decisionforest::TraverseTreeTileOp::getOperationName(),
            1 /*benefit*/, ctx) {
    m_subviewMap =
        std::make_unique<std::map<Operation *, std::vector<Value>>>();
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto traverseTileOp =
        AssertOpIsOfType<mlir::decisionforest::TraverseTreeTileOp>(op);
    auto traverseTileAdaptor =
        decisionforest::TraverseTreeTileOpAdaptor(traverseTileOp);
    assert(operands.size() == 3);
    if (!traverseTileOp)
      return mlir::failure();

    auto location = op->getLoc();
    auto treeType = traverseTileAdaptor.getTree()
                        .getType()
                        .cast<decisionforest::TreeType>();
    auto tileSize = treeType.getTileSize();
    if (tileSize == 1)
      return mlir::failure();

    // TODO_Ashwin Assume that successive threads are going to process
    // successive rows Find the subview op that is the source of the row for
    // this traverseTile
    auto rowVal = traverseTileAdaptor.getData();
    auto definingOp = rowVal.getDefiningOp();

    auto rowSubviewOp = AssertOpIsOfType<memref::SubViewOp>(definingOp);
    auto rowIndex = rowSubviewOp.getOffsets()[0];
    auto zeroIndexAttr = rewriter.getIndexAttr(0);

    auto sizes = rowSubviewOp.getStaticSizesAttr();
    auto size1Attr = rewriter.getIndexAttr(sizes[0]);
    auto size2Attr = rewriter.getIndexAttr(sizes[1]);

    auto strides = rowSubviewOp.getStaticStridesAttr();
    auto strides1Attr = rewriter.getIndexAttr(strides[0]);
    auto strides2Attr = rewriter.getIndexAttr(strides[1]);

    std::vector<Value> rows;
    auto mapIter = m_subviewMap->find(definingOp);
    if (mapIter == m_subviewMap->end()) {
      decisionforest::helpers::SaveAndRestoreInsertionPoint
          restoreInsertionPoint(rewriter);
      rewriter.setInsertionPointAfter(rowSubviewOp);

      auto tileSizeConst =
          rewriter.create<arith::ConstantIndexOp>(location, tileSize);
      auto rowIndByTileSize = rewriter.create<arith::FloorDivSIOp>(
          location, rowIndex, tileSizeConst.getResult());
      auto rowStartIndex = rewriter.create<arith::MulIOp>(
          location, rowIndByTileSize.getResult(), tileSizeConst.getResult());
      rowIndex = rowStartIndex.getResult();

      auto oneConst = rewriter.create<arith::ConstantIndexOp>(location, 1);
      for (auto i = 0; i < tileSize; ++i) {
        auto nextRow = rewriter.create<memref::SubViewOp>(
            location, rowSubviewOp.getSource(),
            ArrayRef<OpFoldResult>({rowIndex, zeroIndexAttr}),
            ArrayRef<OpFoldResult>({size1Attr, size2Attr}),
            ArrayRef<OpFoldResult>({strides1Attr, strides2Attr}));

        auto nextRowIndex = rewriter.create<arith::AddIOp>(
            location, rowIndex, oneConst.getResult());
        rowIndex = nextRowIndex;
        rows.push_back(nextRow.getResult());
      }
      m_subviewMap->insert(std::make_pair(definingOp, rows));
      rewriter.eraseOp(definingOp);
    } else {
      rows = mapIter->second;
    }

    auto simtTraverseTile =
        rewriter.create<decisionforest::CooperativeTraverseTreeTileOp>(
            location, traverseTileOp.getResult().getType(),
            traverseTileOp.getPredicate(), traverseTileOp.getTree(),
            traverseTileOp.getNode(), ValueRange(rows));
    rewriter.replaceOp(op, simtTraverseTile.getResult());
    return mlir::success();
  }
};

struct ConvertTraverseToCooperativeTraverse
    : public PassWrapper<ConvertTraverseToCooperativeTraverse,
                         OperationPass<mlir::ModuleOp>> {
  ConvertTraverseToCooperativeTraverse() {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, scf::SCFDialect,
                    decisionforest::DecisionForestDialect>();
  }

  void runOnOperation() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<
        AffineDialect, memref::MemRefDialect, scf::SCFDialect,
        decisionforest::DecisionForestDialect, vector::VectorDialect,
        math::MathDialect, arith::ArithDialect, func::FuncDialect,
        gpu::GPUDialect>();

    target.addIllegalOp<decisionforest::TraverseTreeTileOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<TraverseToCooperativeTraverseTreeTileOp>(
        patterns.getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

void ConvertTraverseToSimtTraverse(mlir::MLIRContext &context,
                                   mlir::ModuleOp module) {
  // llvm::DebugFlag = true;
  // Lower from high-level IR to mid-level IR
  
  // Check if the environment variable PRINT_AFTER_ALL is set
  const char *printAfterAllEnv = std::getenv("PRINT_AFTER_ALL");
  bool printAfterAll = printAfterAllEnv && std::string(printAfterAllEnv) == "true";

  if(printAfterAll)
    context.disableMultithreading();
  

  mlir::PassManager pm(&context);
  
  // If PRINT_AFTER_ALL is set to "true", enable IR printing
  if (printAfterAll) {
    /* Enable Print After All For Debugging */
    pm.enableIRPrinting(
        [=](mlir::Pass *a, Operation *b) { return false; },  // Don't print before passes
        [=](mlir::Pass *a, Operation *b) { return true; },   // Print after every pass
        true,  // Print at module scope
        false  // Print after every pass, regardless of changes
    );
  }

  pm.addPass(std::make_unique<ConvertTraverseToCooperativeTraverse>());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "GPU SIMT pass failed.\n";
  }
}

} // namespace decisionforest
} // namespace mlir

#endif