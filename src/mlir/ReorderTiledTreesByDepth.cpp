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

  bool AreTreesSorted(decisionforest::DecisionForest<>& forest) const {
    auto trees = forest.GetTrees();
    bool sorted = true;
    size_t i = 0;
    auto prevTreePeelDepth = trees.at(0)->GetTiledTree()->GetLevelsToUnroll();
    for (; i<trees.size() && trees.at(i)->GetTiledTree()->IsProbabilisticallyTiled() ; ++i) {
      auto currTreePeelDepth = trees.at(i)->GetTiledTree()->GetLevelsToUnroll(); 
      if (prevTreePeelDepth > currTreePeelDepth) {
        sorted = false;
        break;
      }
      prevTreePeelDepth = currTreePeelDepth;
    }

    if (sorted && (i < trees.size())) {
      auto prevTreeDepth = trees.at(i)->GetTiledTree()->GetTreeDepth();
      for (; i<trees.size() ; ++i) {
        if (trees.at(i)->GetTiledTree()->IsProbabilisticallyTiled()) {
          sorted = false;
          break;
        }
        auto currTreeDepth = trees.at(i)->GetTiledTree()->GetTreeDepth(); 
        if (prevTreeDepth > currTreeDepth) {
          sorted = false;
          break;
        }
        prevTreeDepth = currTreeDepth;
      }
    }
    return sorted;
  }

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

    if (AreTreesSorted(forest))
      return mlir::failure();
    
    auto trees = forest.GetTrees();
    std::vector<std::shared_ptr<decisionforest::DecisionTree<>>> uniformTiledTrees, probTiledTrees, reorderedTrees;
    std::vector<int32_t> depths, peelDepths;
    for (int64_t i=0 ; i<(int64_t)forest.NumTrees() ; ++i) {
      assert (probTiledTrees.size() == peelDepths.size());
      assert (uniformTiledTrees.size() == depths.size());
      auto tiledTree = trees.at(i)->GetTiledTree();
      if (tiledTree->IsProbabilisticallyTiled()) {
        assert (tiledTree->GetLevelsToUnroll() != -1);
        auto levelsToPeel = tiledTree->GetLevelsToUnroll();

        auto iter=peelDepths.begin();
        for ( ; iter!=peelDepths.end() && *iter<levelsToPeel ; ++iter);
        auto insertionPoint = probTiledTrees.begin() + (iter - peelDepths.begin());
        peelDepths.insert(iter, levelsToPeel);
        probTiledTrees.insert(insertionPoint, trees.at(i));
      }
      else {
        // TODO should we actually call this here?
        // trees.at(i)->GetTiledTree()->MakeAllLeavesSameDepth();
        auto depth = tiledTree->GetTreeDepth();

        auto iter=depths.begin();
        for ( ; iter!=depths.end() && *iter<depth ; ++iter);
        auto insertionPoint = uniformTiledTrees.begin() + (iter - depths.begin());
        depths.insert(iter, depth);
        uniformTiledTrees.insert(insertionPoint, trees.at(i));
      }
    }
    reorderedTrees.insert(reorderedTrees.end(), probTiledTrees.begin(), probTiledTrees.end());
    reorderedTrees.insert(reorderedTrees.end(), uniformTiledTrees.begin(), uniformTiledTrees.end());
    forest.GetTrees() = reorderedTrees;

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
  int32_t m_numberOfCores;
  SplitTreeLoopsByTreeDepthPattern(MLIRContext *ctx, int32_t pipelineSize, int32_t numCores) 
    : RewritePattern(mlir::decisionforest::PredictForestOp::getOperationName(), 1 /*benefit*/, ctx), 
      m_pipelineSize(pipelineSize), m_numberOfCores(numCores)
  {}

  void SplitTreeLoopForProbAndUniformTiling(decisionforest::Schedule* schedule, decisionforest::DecisionForest<>& forest,
                                            decisionforest::IndexVariable* &probTreeIndex, decisionforest::IndexVariable* &probBatchIndex,
                                            decisionforest::IndexVariable* &unifTreeIndex, decisionforest::IndexVariable* &unifBatchIndex) const {
    // TODO What if there are no uniformly tiled trees?
    assert (forest.NumTrees() > 0);
    if (!forest.GetTree(0).GetTiledTree()->IsProbabilisticallyTiled()) {
      // There are no probabilistically tiled trees
      probTreeIndex = probBatchIndex = nullptr;
      unifBatchIndex = &schedule->GetBatchIndex();
      unifTreeIndex = &schedule->GetTreeIndex();
      return;
    }
    else if (forest.GetTrees().back()->GetTiledTree()->IsProbabilisticallyTiled()) {
      // All trees are probabilistically tiled
      unifTreeIndex = unifBatchIndex = nullptr;
      probBatchIndex = &schedule->GetBatchIndex();
      probTreeIndex = &schedule->GetTreeIndex();
      return;
    }
    int32_t splitPoint = 0;
    for (size_t i=0 ; i<forest.NumTrees() ; ++i) {
      if (!forest.GetTree(i).GetTiledTree()->IsProbabilisticallyTiled()) {
        splitPoint = (int32_t)i;
        break;
      }
    }
    assert (splitPoint != 0 && "Expected at least one tree to be probabilistically tiled");
    probTreeIndex = &schedule->NewIndexVariable("probTreeIndex");
    unifTreeIndex = &schedule->NewIndexVariable("unifTreeIndex");
    decisionforest::Schedule::IndexVariableMapType indexMap;
    schedule->Split(schedule->GetTreeIndex(), *probTreeIndex, *unifTreeIndex, splitPoint, indexMap);
    auto mapIter = indexMap.find(&schedule->GetBatchIndex());
    assert (mapIter != indexMap.end());
    probBatchIndex = mapIter->second.first;
    unifBatchIndex = mapIter->second.second;
  }

  void SplitTreeLoopForUniformTiling(decisionforest::Schedule *schedule, decisionforest::DecisionForest<>& forest,
                                     decisionforest::IndexVariable* batchIndexPtr, 
                                     decisionforest::IndexVariable* treeIndexPtr) const {
    if (batchIndexPtr==nullptr && treeIndexPtr==nullptr)
      return;
    if (m_pipelineSize == -1)
      return;
    assert (batchIndexPtr!=nullptr && treeIndexPtr!=nullptr);
    auto& batchIndex = *batchIndexPtr;
    auto& treeIndex = *treeIndexPtr;
    // TODO check this API. Why do we need the second parameter?
    schedule->Pipeline(batchIndex, m_pipelineSize);
    
    // This index may already have been split. So we need to start at the right place
    int32_t currTreeIndex = treeIndex.GetRange().m_start; 
    auto indexToSplit = &treeIndex;
    
    assert (treeIndex.GetRange().m_stop == (int32_t)forest.NumTrees());

    while (currTreeIndex < treeIndex.GetRange().m_stop) {
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
  }

  void SplitTreeLoopForProbabilityBasedTiling(decisionforest::Schedule *schedule, decisionforest::DecisionForest<>& forest,
                                              decisionforest::IndexVariable* batchIndexPtr, 
                                              decisionforest::IndexVariable* treeIndexPtr) const {
    if (batchIndexPtr==nullptr && treeIndexPtr==nullptr)
      return;
    
    assert (batchIndexPtr!=nullptr && treeIndexPtr!=nullptr);
    auto& treeIndex = *treeIndexPtr;
    
    int32_t currTreeIndex = 0;
    assert (treeIndex.GetRange().m_start == 0);
    auto indexToSplit = &treeIndex;
    auto currBatchIndex = batchIndexPtr;
    while (currTreeIndex < (int32_t)treeIndex.GetRange().m_stop) {
      auto tiledTree = forest.GetTree(currTreeIndex).GetTiledTree();
      assert (tiledTree->IsProbabilisticallyTiled());
      int32_t currDepth = tiledTree->GetLevelsToUnroll();
      int32_t intervalEnd = currTreeIndex;
      int32_t intervalEndTreeDepth = currDepth;
      std::map<decisionforest::IndexVariable*, std::pair<decisionforest::IndexVariable*, decisionforest::IndexVariable*>> indexMap;
      while (currDepth == intervalEndTreeDepth && ++intervalEnd < indexToSplit->GetRange().m_stop) {
        intervalEndTreeDepth = forest.GetTree(intervalEnd).GetTiledTree()->GetLevelsToUnroll();
      }

      // No need to split if we're splitting the last index.
      if (intervalEnd == indexToSplit->GetRange().m_stop) {
        schedule->PeelWalk(*currBatchIndex, currDepth);
        break;
      }

      // Split tree loop to go from currTreeIndex to intervalEnd
      auto& firstIndex = schedule->NewIndexVariable(std::string("tree_") + std::to_string(currTreeIndex));
      auto& secondIndex = schedule->NewIndexVariable(std::string("tree_") + std::to_string(intervalEnd));
      
      assert (indexToSplit->GetRange().m_start == currTreeIndex);
      schedule->Split(*indexToSplit, firstIndex, secondIndex, intervalEnd, indexMap);
      // Figure out the set of batch index variables after the split
      auto mapIter = indexMap.find(currBatchIndex);
      assert (mapIter != indexMap.end());

      schedule->PeelWalk(*(mapIter->second.first), currDepth);

      indexToSplit = &secondIndex;
      currBatchIndex = mapIter->second.second;
      currTreeIndex = intervalEnd;
    }
  }

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

    auto* batchIndexPtr = &(schedule->GetBatchIndex());
    auto& treeIndex = schedule->GetTreeIndex();
    if (m_numberOfCores != -1) {
      auto& batchIndex = schedule->GetBatchIndex();
      auto& b0 = schedule->NewIndexVariable("b0");
      auto& b1_parallel = schedule->NewIndexVariable("b1_parallel");
      schedule->Tile(batchIndex, b0, b1_parallel, m_numberOfCores);
      schedule->Reorder({&b1_parallel, &b0, &treeIndex});
      schedule->Parallel(b1_parallel);
      batchIndexPtr = &b0;
    }

    // TODO we're assuming that the schedule is the default unmodified schedule
    
    
    schedule->Reorder({&treeIndex, batchIndexPtr});
    decisionforest::IndexVariable* probTreeIndex=nullptr, *probBatchIndex=nullptr;
    decisionforest::IndexVariable* unifTreeIndex=nullptr, *unifBatchIndex=nullptr;
    SplitTreeLoopForProbAndUniformTiling(schedule, forest, probTreeIndex, probBatchIndex, unifTreeIndex, unifBatchIndex);
    // Tile the batch loop so we can pipeline it
    // auto& b0 = schedule->NewIndexVariable("b0");
    // auto& b1 = schedule->NewIndexVariable("b1");
    // schedule->Tile(batchIndex, b0, b1, m_pipelineSize);
    SplitTreeLoopForProbabilityBasedTiling(schedule, forest, probBatchIndex, probTreeIndex);
    SplitTreeLoopForUniformTiling(schedule, forest, unifBatchIndex, unifTreeIndex);
    return mlir::success();
  }

};

struct SplitTreeLoopByDepth : public PassWrapper<SplitTreeLoopByDepth, FunctionPass> {
  int32_t m_pipelineSize;
  int32_t m_numCores;
  SplitTreeLoopByDepth(int32_t pipelineSize, int32_t numCores) 
  :m_pipelineSize(pipelineSize), m_numCores(numCores)
  { }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, StandardOpsDialect, scf::SCFDialect, math::MathDialect>();
  }
  void runOnFunction() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<SplitTreeLoopsByTreeDepthPattern>(&getContext(), m_pipelineSize, m_numCores);

    if (failed(applyPatternsAndFoldGreedily(getFunction(), std::move(patterns))))
        signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir
{
namespace decisionforest
{
void DoReorderTreesByDepth(mlir::MLIRContext& context, mlir::ModuleOp module, int32_t pipelineSize, int32_t numCores) {
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  optPM.addPass(std::make_unique<ReorderTreesByDepthPass>());
  // TODO pipelineSize needs to be added to CompilerOptions
  optPM.addPass(std::make_unique<SplitTreeLoopByDepth>(pipelineSize, numCores));

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to mid level IR failed.\n";
  }
}

} // decisionforest
} // mlir
