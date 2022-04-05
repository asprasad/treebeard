// Implementation of a transformation to uniformly tile all trees in a forest. 
// The "uniform" represents that assumption that all edge probabilities are the same. 
// The tile size also needs to be constant across trees.

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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <queue>
#include <cassert>
#include "TiledTree.h"

using namespace mlir;
namespace {

template<typename T>
T AssertOpIsOfType(Operation* operation) {
  T typedOp = llvm::dyn_cast<T>(operation);
  assert(typedOp);
  return typedOp;
}

struct ReorderEnsembleConstants : public RewritePattern {

  ReorderEnsembleConstants(MLIRContext *ctx) 
    : RewritePattern(mlir::decisionforest::PredictForestOp::getOperationName(), 1 /*benefit*/, ctx)
  {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const final {
    mlir::decisionforest::PredictForestOp predictOp = llvm::dyn_cast<mlir::decisionforest::PredictForestOp>(op);
    assert(predictOp);
    if (!predictOp)
         return mlir::failure();

    auto forestAttribute = predictOp.ensemble();
    auto forest = forestAttribute.GetDecisionForest();
    auto forestType = forestAttribute.getType().cast<decisionforest::TreeEnsembleType>();
    auto tilingDescriptor = forest.GetTree(0).TilingDescriptor();
    // If we've haven't tiled this forest, then we don't want to reorder.
    if (tilingDescriptor.MaxTileSize() == 1)
      return mlir::failure();

    std::vector<std::shared_ptr<decisionforest::DecisionTree<>>> newTrees;
    auto trees = forest.GetTrees();

    bool sorted = true;
    auto prevTreeDepth = trees.at(0)->GetTiledTree()->GetTreeDepth();
    for (size_t i=1 ; i<trees.size() ; ++i) {
      auto currTreeDepth = trees.at(i)->GetTiledTree()->GetTreeDepth(); 
      if (prevTreeDepth > currTreeDepth) {
        sorted = false;
        break;
      }
      prevTreeDepth = currTreeDepth;
    }
    if (sorted)
      return mlir::failure();
    
    std::vector<int32_t> depths;
    for (int64_t i=0 ; i<(int64_t)forest.NumTrees() ; ++i) {
      assert (newTrees.size() == depths.size());
      // TODO should we actually call this here?
      // trees.at(i)->GetTiledTree()->MakeAllLeavesSameDepth();
      auto depth = trees.at(i)->GetTiledTree()->GetTreeDepth();

      auto iter=depths.begin();
      for ( ; iter!=depths.end() && *iter<depth ; ++iter);
      auto insertionPoint = newTrees.begin() + (iter - depths.begin());
      depths.insert(iter, depth);
      newTrees.insert(insertionPoint, trees.at(i));
    }
    forest.GetTrees() = newTrees;
    auto newForestAttribute = decisionforest::DecisionForestAttribute::get(forestType, forest);
    auto reorderedPredictForestOp = rewriter.create<decisionforest::PredictForestOp>(op->getLoc(), predictOp.getResult().getType(), 
                                                                                 newForestAttribute, predictOp.data(), 
                                                                                 predictOp.result(), predictOp.schedule());
    rewriter.replaceOp(op, static_cast<Value>(reorderedPredictForestOp));
    return mlir::success();
  }

};

struct ReorderTreesByDepthPass : public PassWrapper<ReorderTreesByDepthPass, FunctionPass> {
  ReorderTreesByDepthPass() 
  { }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, StandardOpsDialect, scf::SCFDialect, math::MathDialect>();
  }
  void runOnFunction() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<ReorderEnsembleConstants>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getFunction(), std::move(patterns))))
        signalPassFailure();
  }
};

struct SplitTreeLoopsByTreeDepthPattern : public RewritePattern {

  int32_t m_pipelineSize;
  SplitTreeLoopsByTreeDepthPattern(MLIRContext *ctx, int32_t pipelineSize) 
    : RewritePattern(mlir::decisionforest::PredictForestOp::getOperationName(), 1 /*benefit*/, ctx), m_pipelineSize(pipelineSize)
  {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const final {
    mlir::decisionforest::PredictForestOp predictOp = llvm::dyn_cast<mlir::decisionforest::PredictForestOp>(op);
    assert(predictOp);
    if (!predictOp)
         return mlir::failure();

    auto scheduleAttribute = predictOp.schedule();
    auto schedule = scheduleAttribute.GetSchedule();
    auto forestAttribute = predictOp.ensemble();
    auto& forest = forestAttribute.GetDecisionForest();
    
    // Don't match if we've already modified the schedule on this op. Prevents
    // infinite loops
    if (!schedule->IsDefaultSchedule())
      return mlir::failure();

    // TODO we're assuming that the schedule is the default unmodified schedule
    auto& batchIndex = schedule->GetBatchIndex();
    auto& treeIndex = schedule->GetTreeIndex();
    schedule->Reorder({&treeIndex, &batchIndex});
    
    // Tile the batch loop so we can pipeline it
    // auto& b0 = schedule->NewIndexVariable("b0");
    // auto& b1 = schedule->NewIndexVariable("b1");
    // schedule->Tile(batchIndex, b0, b1, m_pipelineSize);
    // TODO check this API. Why do we need the second parameter?
    schedule->Pipeline(batchIndex, m_pipelineSize);
    
    int32_t currTreeIndex = 0;
    auto indexToSplit = &treeIndex;
    while (currTreeIndex < (int32_t)forest.NumTrees()) {
      int32_t currDepth = forest.GetTree(currTreeIndex).GetTiledTree()->GetTreeDepth();
      int32_t intervalEnd = currTreeIndex;
      int32_t intervalEndTreeDepth = currDepth;
      std::map<decisionforest::IndexVariable*, std::pair<decisionforest::IndexVariable*, decisionforest::IndexVariable*>> indexMap;
      while (currDepth == intervalEndTreeDepth && ++intervalEnd < (int32_t)forest.NumTrees()) {
        intervalEndTreeDepth = forest.GetTree(intervalEnd).GetTiledTree()->GetTreeDepth();
      }

      // No need to split if we're splitting the last index.
      if (intervalEnd == indexToSplit->GetRange().m_stop) {
        indexToSplit->SetUnrollFactor(currDepth);
        break;
      }

      // Split tree loop to go from currTreeIndex to intervalEnd
      auto& firstIndex = schedule->NewIndexVariable(std::string("tree_") + std::to_string(currTreeIndex));
      auto& secondIndex = schedule->NewIndexVariable(std::string("tree_") + std::to_string(intervalEnd));
      
      assert (indexToSplit->GetRange().m_start == currTreeIndex);
      schedule->Split(*indexToSplit, firstIndex, secondIndex, intervalEnd, indexMap);
      firstIndex.SetUnrollFactor(currDepth);

      indexToSplit = &secondIndex;
      currTreeIndex = intervalEnd;
    }

    return mlir::success();
  }

};

struct SplitTreeLoopByDepth : public PassWrapper<SplitTreeLoopByDepth, FunctionPass> {
  int32_t m_pipelineSize;
  SplitTreeLoopByDepth(int32_t pipelineSize) 
  :m_pipelineSize(pipelineSize)
  { }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, StandardOpsDialect, scf::SCFDialect, math::MathDialect>();
  }
  void runOnFunction() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<SplitTreeLoopsByTreeDepthPattern>(&getContext(), m_pipelineSize);

    if (failed(applyPatternsAndFoldGreedily(getFunction(), std::move(patterns))))
        signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir
{
namespace decisionforest
{
void DoReorderTreesByDepth(mlir::MLIRContext& context, mlir::ModuleOp module, int32_t pipelineSize) {
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  optPM.addPass(std::make_unique<ReorderTreesByDepthPass>());
  // TODO pipelineSize needs to be added to CompilerOptions
  if (pipelineSize != -1)
    optPM.addPass(std::make_unique<SplitTreeLoopByDepth>(pipelineSize));

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to mid level IR failed.\n";
  }
}

} // decisionforest
} // mlir
