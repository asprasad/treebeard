// Implementation of a transformation to uniformly tile all trees in a forest.
// The "uniform" represents that assumption that all edge probabilities are the
// same. The tile size also needs to be constant across trees.

#include "Dialect.h"
// #include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "TiledTree.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cassert>
#include <queue>

#include "GPUCompileUtils.h"
#include "GPUModelSerializers.h"
#include "Logger.h"
#include "Representations.h"
#include "TreebeardContext.h"
#include "xgboostparser.h"

#include "GPUBenchmarkUtils.h"

using namespace mlir;
using namespace TreeBeard::test;

namespace {

template <typename T> T AssertOpIsOfType(Operation *operation) {
  T typedOp = llvm::dyn_cast<T>(operation);
  assert(typedOp);
  return typedOp;
}

struct ReorderEnsembleConstants : public RewritePattern {

  ReorderEnsembleConstants(MLIRContext *ctx)
      : RewritePattern(
            mlir::decisionforest::PredictForestOp::getOperationName(),
            1 /*benefit*/, ctx) {}

  bool AreUntiledTreesSorted(decisionforest::DecisionForest &forest) const {
    auto &trees = forest.GetTrees();
    assert(trees.at(0)->TilingDescriptor().MaxTileSize() == 1);
    bool sorted = true;
    auto prevTreePeelDepth = trees.at(0)->GetTreeDepth();
    for (size_t i = 0; i < trees.size(); ++i) {
      auto currTreePeelDepth = trees.at(i)->GetTreeDepth();
      if (prevTreePeelDepth > currTreePeelDepth) {
        sorted = false;
        break;
      }
      prevTreePeelDepth = currTreePeelDepth;
    }
    return sorted;
  }

  bool AreTiledTreesSorted(decisionforest::DecisionForest &forest) const {
    auto trees = forest.GetTrees();
    bool sorted = true;
    size_t i = 0;
    auto prevTreePeelDepth = trees.at(0)->GetTiledTree()->GetLevelsToUnroll();
    for (; i < trees.size() &&
           trees.at(i)->GetTiledTree()->IsProbabilisticallyTiled();
         ++i) {
      auto currTreePeelDepth = trees.at(i)->GetTiledTree()->GetLevelsToUnroll();
      if (prevTreePeelDepth > currTreePeelDepth) {
        sorted = false;
        break;
      }
      prevTreePeelDepth = currTreePeelDepth;
    }

    if (sorted && (i < trees.size())) {
      auto prevTreeDepth = trees.at(i)->GetTiledTree()->GetTreeDepth();
      for (; i < trees.size(); ++i) {
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

  LogicalResult handleUntiledForest(Operation *op,
                                    PatternRewriter &rewriter) const {
    mlir::decisionforest::PredictForestOp predictOp =
        llvm::dyn_cast<mlir::decisionforest::PredictForestOp>(op);
    assert(predictOp);
    if (!predictOp)
      return mlir::failure();

    auto forestAttribute = predictOp.getEnsemble();
    auto forest = forestAttribute.GetDecisionForest();
    auto forestType =
        forestAttribute.getType().cast<decisionforest::TreeEnsembleType>();
    auto tilingDescriptor = forest.GetTree(0).TilingDescriptor();

    assert(tilingDescriptor.MaxTileSize() == 1);

    if (AreUntiledTreesSorted(forest))
      return mlir::failure();

    auto trees = forest.GetTrees();
    std::sort(trees.begin(), trees.end(),
              [](const std::shared_ptr<decisionforest::DecisionTree> &a,
                 const std::shared_ptr<decisionforest::DecisionTree> &b) {
                return a->GetTreeDepth() < b->GetTreeDepth();
              });
    forest.GetTrees() = trees;

    auto newForestAttribute =
        decisionforest::DecisionForestAttribute::get(forestType, forest);
    auto reorderedPredictForestOp =
        rewriter.create<decisionforest::PredictForestOp>(
            op->getLoc(), predictOp.getResult().getType(), newForestAttribute,
            predictOp.getPredicateAttr(), predictOp.getData(),
            predictOp.getResult(), predictOp.getSchedule());
    rewriter.replaceOp(op, static_cast<Value>(reorderedPredictForestOp));
    return mlir::success();
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    mlir::decisionforest::PredictForestOp predictOp =
        llvm::dyn_cast<mlir::decisionforest::PredictForestOp>(op);
    assert(predictOp);
    if (!predictOp)
      return mlir::failure();

    auto forestAttribute = predictOp.getEnsemble();
    auto forest = forestAttribute.GetDecisionForest();
    auto forestType =
        forestAttribute.getType().cast<decisionforest::TreeEnsembleType>();
    auto tilingDescriptor = forest.GetTree(0).TilingDescriptor();
    // If we've haven't tiled this forest, then we don't want to reorder.
    if (tilingDescriptor.MaxTileSize() == 1)
      return handleUntiledForest(op, rewriter);

    if (AreTiledTreesSorted(forest))
      return mlir::failure();

    auto trees = forest.GetTrees();
    std::vector<std::shared_ptr<decisionforest::DecisionTree>>
        uniformTiledTrees, probTiledTrees, reorderedTrees;
    std::vector<int32_t> depths, peelDepths;
    for (int64_t i = 0; i < (int64_t)forest.NumTrees(); ++i) {
      assert(probTiledTrees.size() == peelDepths.size());
      assert(uniformTiledTrees.size() == depths.size());
      auto tiledTree = trees.at(i)->GetTiledTree();
      if (tiledTree->IsProbabilisticallyTiled()) {
        assert(tiledTree->GetLevelsToUnroll() != -1);
        auto levelsToPeel = tiledTree->GetLevelsToUnroll();

        auto iter = peelDepths.begin();
        for (; iter != peelDepths.end() && *iter < levelsToPeel; ++iter)
          ;
        auto insertionPoint =
            probTiledTrees.begin() + (iter - peelDepths.begin());
        peelDepths.insert(iter, levelsToPeel);
        probTiledTrees.insert(insertionPoint, trees.at(i));
      } else {
        // TODO should we actually call this here?
        // trees.at(i)->GetTiledTree()->MakeAllLeavesSameDepth();
        auto depth = tiledTree->GetTreeDepth();

        auto iter = depths.begin();
        for (; iter != depths.end() && *iter < depth; ++iter)
          ;
        auto insertionPoint =
            uniformTiledTrees.begin() + (iter - depths.begin());
        depths.insert(iter, depth);
        uniformTiledTrees.insert(insertionPoint, trees.at(i));
      }
    }
    reorderedTrees.insert(reorderedTrees.end(), probTiledTrees.begin(),
                          probTiledTrees.end());
    reorderedTrees.insert(reorderedTrees.end(), uniformTiledTrees.begin(),
                          uniformTiledTrees.end());
    forest.GetTrees() = reorderedTrees;

    auto newForestAttribute =
        decisionforest::DecisionForestAttribute::get(forestType, forest);
    auto reorderedPredictForestOp =
        rewriter.create<decisionforest::PredictForestOp>(
            op->getLoc(), predictOp.getResult().getType(), newForestAttribute,
            predictOp.getPredicateAttr(), predictOp.getData(),
            predictOp.getResult(), predictOp.getSchedule());
    rewriter.replaceOp(op, static_cast<Value>(reorderedPredictForestOp));
    return mlir::success();
  }
};

struct ReorderTreesByDepthPass
    : public PassWrapper<ReorderTreesByDepthPass,
                         OperationPass<mlir::func::FuncOp>> {
  ReorderTreesByDepthPass() {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, scf::SCFDialect,
                    math::MathDialect>();
  }
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<ReorderEnsembleConstants>(&getContext());

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

struct SplitTreeLoopsByTreeDepthPattern : public RewritePattern {

  TreeBeard::GPUAutoScheduleOptions m_options;

  SplitTreeLoopsByTreeDepthPattern(MLIRContext *ctx,
                                   TreeBeard::GPUAutoScheduleOptions &options)
      : RewritePattern(
            mlir::decisionforest::PredictForestOp::getOperationName(),
            1 /*benefit*/, ctx),
        m_options(options) {}

  bool doesTreeLoopNeedSpecialization(decisionforest::DecisionForest &forest,
                                      int32_t numParallelTreeBatches) const {
    auto &trees = forest.GetTrees();

    std::list<int32_t> depths[numParallelTreeBatches];
    for (size_t i = 0; i < trees.size(); i += numParallelTreeBatches) {
      for (auto j = 0; j < numParallelTreeBatches; ++j) {
        int32_t depth = trees.at(i + j)->GetTreeDepth();
        depths[j].push_back(depth);
      }
    }
    for (auto j = 1; j < numParallelTreeBatches; ++j)
      if (depths[0] != depths[j]) {
        TreeBeard::Logging::Log(
            "GPU Autoschedule: Tree loop needs specialization");
        return true;
      }
    TreeBeard::Logging::Log(
        "GPU Autoschedule:Tree loop does not need specialization");
    return false;
  }

  void SplitTreeLoopForUnrolling(decisionforest::Schedule *schedule,
                                 decisionforest::DecisionForest &forest,
                                 decisionforest::IndexVariable *batchIndexPtr,
                                 decisionforest::IndexVariable *treeIndexPtr,
                                 int32_t startIndex, int32_t stride) const {
    if (batchIndexPtr == nullptr && treeIndexPtr == nullptr)
      return;
    assert(batchIndexPtr != nullptr && treeIndexPtr != nullptr);
    auto &treeIndex = *treeIndexPtr;

    assert(treeIndex.GetRange().m_start == 0);
    assert(treeIndex.GetRange().m_step == stride);

    int32_t currTreeIndex = 0;
    auto indexToSplit = &treeIndex;

    assert(treeIndex.GetRange().m_stop == (int32_t)forest.NumTrees());

    while (currTreeIndex < treeIndex.GetRange().m_stop) {
      int32_t currDepth =
          forest.GetTree(currTreeIndex + startIndex).GetTreeDepth();
      int32_t intervalEnd = currTreeIndex;
      int32_t intervalEndTreeDepth = currDepth;
      std::map<decisionforest::IndexVariable *,
               std::pair<decisionforest::IndexVariable *,
                         decisionforest::IndexVariable *>>
          indexMap;
      while (currDepth == intervalEndTreeDepth &&
             (intervalEnd += stride) < (int32_t)forest.NumTrees()) {
        intervalEndTreeDepth =
            forest.GetTree(intervalEnd + startIndex).GetTreeDepth();
      }

      // No need to split if we're splitting the last index.
      if (intervalEnd == indexToSplit->GetRange().m_stop) {
        indexToSplit->SetTreeWalkUnrollFactor(currDepth - 1);
        if (m_options.treeWalkInterleaveFactor != -1)
          schedule->Pipeline(*indexToSplit, m_options.treeWalkInterleaveFactor);
        break;
      }

      // Split tree loop to go from currTreeIndex to intervalEnd
      auto &firstIndex = schedule->NewIndexVariable(
          std::string("tree_") + std::to_string(currTreeIndex));
      auto &secondIndex = schedule->NewIndexVariable(
          std::string("tree_") + std::to_string(intervalEnd));

      assert(indexToSplit->GetRange().m_start == currTreeIndex);
      schedule->Split(*indexToSplit, firstIndex, secondIndex, intervalEnd,
                      indexMap);
      firstIndex.SetTreeWalkUnrollFactor(currDepth - 1);
      if (m_options.treeWalkInterleaveFactor != -1)
        schedule->Pipeline(firstIndex, m_options.treeWalkInterleaveFactor);

      indexToSplit = &secondIndex;
      currTreeIndex = intervalEnd;
    }
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    mlir::decisionforest::PredictForestOp predictOp =
        llvm::dyn_cast<mlir::decisionforest::PredictForestOp>(op);
    assert(predictOp);
    if (!predictOp)
      return mlir::failure();

    auto scheduleAttribute = predictOp.getSchedule();
    auto schedule = scheduleAttribute.GetSchedule();
    auto forestAttribute = predictOp.getEnsemble();
    auto &forest = forestAttribute.GetDecisionForest();

    // Don't match if we've already modified the schedule on this op. Prevents
    // infinite loops
    // TODO we're assuming that the schedule is the default unmodified schedule
    if (!schedule->IsDefaultSchedule())
      return mlir::failure();

    decisionforest::IndexVariable *treeLoopToSpecialize = nullptr;
    decisionforest::IndexVariable *treeLoopToSplitAndUnroll = nullptr;
    decisionforest::IndexVariable *batchLoopToSpecialize = nullptr;

    auto *batchIndexPtr = &(schedule->GetBatchIndex());
    auto &treeIndex = schedule->GetTreeIndex();
    std::vector<decisionforest::IndexVariable *> gridLoops, threadBlockLoops,
        perThreadLoops;

    auto &gridXIndex = schedule->NewIndexVariable("gridXIndex_batch");
    gridLoops.push_back(&gridXIndex);

    auto &threadXIndex = schedule->NewIndexVariable("threadXIndex_batch");
    threadBlockLoops.push_back(&threadXIndex);
    // TODO_Ashwin RowTileSize is not being used
    // Construct another loop with this and then set a variable to it so that we
    // can consistently place the per thread tree loop before this loop.
    // Also this is the loop we need to enable caching for
    if (true || m_options.numRowsPerThread != 1) {
      auto &tempBatchIndex = schedule->NewIndexVariable("tempBatchIndex_batch");
      auto &perThreadBatchIndex =
          schedule->NewIndexVariable("perThreadBatchIndex_batch");

      schedule->Tile(*batchIndexPtr, gridXIndex, tempBatchIndex,
                     m_options.numRowsPerTB);
      schedule->Tile(tempBatchIndex, threadXIndex, perThreadBatchIndex,
                     m_options.numRowsPerThread);

      perThreadLoops.push_back(&perThreadBatchIndex);
      batchLoopToSpecialize = &perThreadBatchIndex;
    } else {
      schedule->Tile(*batchIndexPtr, gridXIndex, threadXIndex,
                     m_options.numRowsPerTB);
    }
    if (m_options.cacheRows)
      schedule->Cache(gridXIndex);

    gridXIndex.SetGPUDimension(
        decisionforest::IndexVariable::GPUConstruct::Grid,
        decisionforest::IndexVariable::Dimension::X);
    threadXIndex.SetGPUDimension(
        decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
        decisionforest::IndexVariable::Dimension::X);

    if (m_options.numTreeThreads != 1) {
      auto &threadYIndex = schedule->NewIndexVariable("threadYIndex_tree");
      auto &perThreadTreeIndex =
          schedule->NewIndexVariable("perThreadTreeIndex_tree");
      schedule->Tile(treeIndex, perThreadTreeIndex, threadYIndex,
                     m_options.numTreeThreads);

      if (m_options.treeWalkInterleaveFactor == -1) {
        perThreadLoops.insert(perThreadLoops.begin(), &perThreadTreeIndex);
      } else {
        perThreadLoops.push_back(&perThreadTreeIndex);
      }

      threadBlockLoops.push_back(&threadYIndex);
      if (m_options.cacheTrees)
        schedule->Cache(perThreadTreeIndex);

      if (m_options.sharedMemoryReduce)
        schedule->SharedReduce(threadYIndex);

      threadYIndex.SetGPUDimension(
          decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
          decisionforest::IndexVariable::Dimension::Y);

      treeLoopToSpecialize = &threadYIndex;
      treeLoopToSplitAndUnroll = &perThreadTreeIndex;
    } else {
      if (m_options.treeWalkInterleaveFactor == -1) {
        perThreadLoops.insert(perThreadLoops.begin(), &treeIndex);
      } else {
        perThreadLoops.push_back(&treeIndex);
      }
      if (m_options.cacheTrees)
        schedule->Cache(treeIndex);
      treeLoopToSpecialize = nullptr;
      treeLoopToSplitAndUnroll = &treeIndex;
    }

    std::vector<decisionforest::IndexVariable *> loopOrder;
    // Add gridLoops, then threadBlockLoops, then perThreadLoops
    loopOrder.insert(loopOrder.end(), gridLoops.begin(), gridLoops.end());
    loopOrder.insert(loopOrder.end(), threadBlockLoops.begin(),
                     threadBlockLoops.end());
    loopOrder.insert(loopOrder.end(), perThreadLoops.begin(),
                     perThreadLoops.end());
    schedule->Reorder(loopOrder);

    if (m_options.unrollTreeWalks) {

      // if there is parallelism across trees, we need to specialize the tree
      // index if not, then we can just split and unroll the tree loop
      if (m_options.numTreeThreads != 1) {
        assert(treeLoopToSpecialize != nullptr);
        assert(batchLoopToSpecialize != nullptr);
        assert(treeLoopToSplitAndUnroll != nullptr);
        if (doesTreeLoopNeedSpecialization(forest, m_options.numTreeThreads)) {
          decisionforest::Schedule::IterationSpecializationInfo
              specializationInfo;
          schedule->SpecializeIterations(*treeLoopToSpecialize,
                                         specializationInfo);
          auto numIterations =
              (int32_t)specializationInfo.m_iterationMaps.size();
          assert(numIterations == m_options.numTreeThreads);
          auto start = 0;
          auto stride = numIterations;
          for (auto iter = 0; iter < numIterations; ++iter) {
            auto *treeIndexPtr =
                specializationInfo
                    .m_iterationMaps[iter][treeLoopToSplitAndUnroll];
            auto *batchIndexPtr =
                specializationInfo.m_iterationMaps[iter][batchLoopToSpecialize];
            // We're going to start at a tree with an offset equal to the
            // iteration number
            start = iter;
            SplitTreeLoopForUnrolling(schedule, forest, batchIndexPtr,
                                      treeIndexPtr, start, stride);
          }
        } else {
          SplitTreeLoopForUnrolling(schedule, forest, batchLoopToSpecialize,
                                    treeLoopToSplitAndUnroll, 0,
                                    m_options.numTreeThreads);
        }
      } else {
        assert(treeLoopToSplitAndUnroll != nullptr);
        assert(treeLoopToSpecialize == nullptr);
        auto start = 0, stride = 1;
        SplitTreeLoopForUnrolling(schedule, forest, batchLoopToSpecialize,
                                  treeLoopToSplitAndUnroll, start, stride);
      }
    }
    return mlir::success();
  }
};

struct SplitTreeLoopByDepth
    : public PassWrapper<SplitTreeLoopByDepth,
                         OperationPass<mlir::func::FuncOp>> {
  TreeBeard::GPUAutoScheduleOptions m_options;
  SplitTreeLoopByDepth(const TreeBeard::GPUAutoScheduleOptions &options)
      : m_options(options) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, scf::SCFDialect,
                    math::MathDialect>();
  }
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<SplitTreeLoopsByTreeDepthPattern>(&getContext(), m_options);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

// ===---------------------------------------------------=== //
// GPU Auto-tuning Heuristic
// ===---------------------------------------------------=== //

class GPUAutoTuner {
  std::vector<int32_t> rowsPerTB;
  std::vector<int32_t> rowsPerThread;
  std::vector<int32_t> numTreeThreads;
  bool m_shouldTryUnroll = true;
  bool m_shouldCacheRows = true;
  const bool m_verboseLogs = false;

  int32_t m_numFeatures;
  int32_t m_numTrees;
  int32_t m_batchSize;
  bool m_multiClass;

  TreeBeard::GPUAutoScheduleOptions m_bestOptions;
  std::string m_bestRep;
  std::string m_model;

  void updateConfig(const TreeBeard::GPUAutoScheduleOptions &options,
                    const std::string &rep) {
    m_bestOptions = options;
    m_bestRep = rep;

    if (m_verboseLogs) {
      std::cerr << "\nUpdating best configuration\n";
      this->printBestSchedule();
    }
  }

  void computeScheduleSubset() {
    // if batchSize < 1k, then focus on larger number of tree threads
    if (m_batchSize <= 1024) {
      numTreeThreads = {20, 50};
      rowsPerTB = {8, 32};
      rowsPerThread = {1};
    } else {
      const int32_t featureThreshold = 100;
      // if we have a large batch sizes, with a large
      // feature count, use a large number of tree threads
      if (m_numFeatures > featureThreshold) {
        numTreeThreads = {20, 50};
        rowsPerTB = {8, 32};
        rowsPerThread = {1};
      } else {
        numTreeThreads = {1, 2, 10};
        rowsPerTB = {32, 64};
        rowsPerThread = {1};
      }
    }
  }

  void initParameters(const std::string &modelName, int32_t batchSize) {
    std::vector<std::string> benchmarks{"abalone", "airline", "airline-ohe",
                                        // "bosch",
                                        "covtype", "epsilon", "higgs",
                                        "letters", "year_prediction_msd"};

    std::vector<bool> isMultiClass{false, false, false, // false,
                                   true,  false, false, true, false};

    std::vector<int32_t> numFeatures{8, 13, 692, 54, 2000, 28, 16, 90};

    std::vector<int32_t> numTrees{1000, 100, 1000, 800, 100, 100, 26000, 100};

    auto it = std::find_if(benchmarks.begin(), benchmarks.end(),
                           [&](const std::string &arg) {
                             return modelName.find(arg) != std::string::npos;
                           });
    assert(it != benchmarks.end());
    auto index = std::distance(benchmarks.begin(), it);

    m_numFeatures = numFeatures[index];
    m_numTrees = numTrees[index];
    m_multiClass = isMultiClass[index];

    computeScheduleSubset();

    // If the num features is too large to fit 8 rows
    // into shared mem, then don't try to cache
    if (m_numFeatures > 1500)
      m_shouldCacheRows = false;
  }

  void trySharedReduce(double &bestKernelTime) {
    if (!m_multiClass)
      return;
    std::cerr << "Checking shared reduce\n";
    printBestSchedule();
    std::cerr << "*************\n";

    if (m_bestOptions.unrollTreeWalks == false) {
      constexpr bool unrollTreeWalks = false;
      TreeBeard::GPUAutoScheduleOptions options = m_bestOptions;
      options.sharedMemoryReduce = true;

      auto time = BenchmarkIfNoSharedMemOverflow<float, int8_t, true, false,
                                                 unrollTreeWalks, true>(
          m_model, m_bestRep, m_batchSize, options.numRowsPerTB,
          options.numRowsPerThread, options.numTreeThreads,
          options.treeWalkInterleaveFactor, true);
      std::cerr << time.kernelTimePerSample << std::endl;
      if (time.kernelTimePerSample != -1 &&
          time.kernelTimePerSample < bestKernelTime) {
        updateConfig(options, m_bestRep);
        bestKernelTime = time.kernelTimePerSample;
      }

      options.cacheRows = false;
      time = BenchmarkIfNoSharedMemOverflow<float, int8_t, false, false,
                                            unrollTreeWalks, true>(
          m_model, m_bestRep, m_batchSize, options.numRowsPerTB,
          options.numRowsPerThread, options.numTreeThreads,
          options.treeWalkInterleaveFactor, true);
      std::cerr << time.kernelTimePerSample << std::endl;
      if (time.kernelTimePerSample != -1 &&
          time.kernelTimePerSample < bestKernelTime) {
        updateConfig(options, m_bestRep);
        bestKernelTime = time.kernelTimePerSample;
      }

    } else {
      constexpr bool unrollTreeWalks = true;
      TreeBeard::GPUAutoScheduleOptions options = m_bestOptions;
      options.sharedMemoryReduce = true;

      auto time = BenchmarkIfNoSharedMemOverflow<float, int8_t, true, false,
                                                 unrollTreeWalks, true>(
          m_model, m_bestRep, m_batchSize, options.numRowsPerTB,
          options.numRowsPerThread, options.numTreeThreads,
          options.treeWalkInterleaveFactor, true);

      if (time.kernelTimePerSample != -1 &&
          time.kernelTimePerSample < bestKernelTime) {
        updateConfig(options, m_bestRep);
        bestKernelTime = time.kernelTimePerSample;
      }

      options.cacheRows = false;
      time = BenchmarkIfNoSharedMemOverflow<float, int8_t, false, false,
                                            unrollTreeWalks, true>(
          m_model, m_bestRep, m_batchSize, options.numRowsPerTB,
          options.numRowsPerThread, options.numTreeThreads,
          options.treeWalkInterleaveFactor, true);

      if (time.kernelTimePerSample != -1 &&
          time.kernelTimePerSample < bestKernelTime) {
        updateConfig(options, m_bestRep);
        bestKernelTime = time.kernelTimePerSample;
      }
    }
  }

  template <bool cacheRows>
  void runBenchmarks(double &bestKernelTime, const std::string &rep,
                     int32_t numRowsPerTB, int32_t numRowsPerThread,
                     int32_t treeThreads) {
    constexpr bool cacheTrees = false;
    constexpr bool sharedMemoryReduce = false;
    {
      constexpr bool unrollTreeWalks = false;
      TreeBeard::GPUAutoScheduleOptions options{
          numRowsPerTB, numRowsPerThread, 1,  treeThreads,       1, cacheRows,
          cacheTrees,   unrollTreeWalks,  -1, sharedMemoryReduce};

      auto time = m_multiClass
                      ? BenchmarkIfNoSharedMemOverflow<float, int8_t, cacheRows,
                                                       false, unrollTreeWalks>(
                            m_model, rep, m_batchSize, numRowsPerTB,
                            numRowsPerThread, treeThreads, -1, true)
                      : BenchmarkIfNoSharedMemOverflow<float, float, cacheRows,
                                                       false, unrollTreeWalks>(
                            m_model, rep, m_batchSize, numRowsPerTB,
                            numRowsPerThread, treeThreads, -1, true);
      if (time.kernelTimePerSample != -1 &&
          time.kernelTimePerSample < bestKernelTime) {
        updateConfig(options, rep);
        bestKernelTime = time.kernelTimePerSample;
      }
    }

    // TODO we need to check if we need to check unrolling for the
    // current model and config
    {
      constexpr bool unrollTreeWalks = true;
      constexpr int32_t interleaveDepth = 2;

      TreeBeard::GPUAutoScheduleOptions options{numRowsPerTB,
                                                numRowsPerThread,
                                                1,
                                                treeThreads,
                                                1,
                                                cacheRows,
                                                cacheTrees,
                                                unrollTreeWalks,
                                                interleaveDepth,
                                                sharedMemoryReduce};

      auto time =
          m_multiClass
              ? BenchmarkIfNoSharedMemOverflow<float, int8_t, cacheRows, false,
                                               unrollTreeWalks>(
                    m_model, rep, m_batchSize, numRowsPerTB, numRowsPerThread,
                    treeThreads, interleaveDepth, true)
              : BenchmarkIfNoSharedMemOverflow<float, float, cacheRows, false,
                                               unrollTreeWalks>(
                    m_model, rep, m_batchSize, numRowsPerTB, numRowsPerThread,
                    treeThreads, interleaveDepth, true);
      if (time.kernelTimePerSample != -1 &&
          time.kernelTimePerSample < bestKernelTime) {
        bestKernelTime = time.kernelTimePerSample;
        updateConfig(options, rep);
      }
    }

    {
      constexpr bool unrollTreeWalks = true;
      constexpr int32_t interleaveDepth = 4;

      TreeBeard::GPUAutoScheduleOptions options{numRowsPerTB,
                                                numRowsPerThread,
                                                1,
                                                treeThreads,
                                                1,
                                                cacheRows,
                                                cacheTrees,
                                                unrollTreeWalks,
                                                interleaveDepth,
                                                sharedMemoryReduce};

      auto time =
          m_multiClass
              ? BenchmarkIfNoSharedMemOverflow<float, int8_t, cacheRows, false,
                                               unrollTreeWalks>(
                    m_model, rep, m_batchSize, numRowsPerTB, numRowsPerThread,
                    treeThreads, interleaveDepth, true)
              : BenchmarkIfNoSharedMemOverflow<float, float, cacheRows, false,
                                               unrollTreeWalks>(
                    m_model, rep, m_batchSize, numRowsPerTB, numRowsPerThread,
                    treeThreads, interleaveDepth, true);
      if (time.kernelTimePerSample != -1 &&
          time.kernelTimePerSample < bestKernelTime) {
        bestKernelTime = time.kernelTimePerSample;
        updateConfig(options, rep);
      }
    }
  }

public:
  GPUAutoTuner(const std::string &modelName, int32_t batchSize)
      : m_batchSize(batchSize), m_model(modelName) {
    initParameters(modelName, batchSize);
  }

  void exploreSchedules() {

    const auto numKernelRuns = mlir::decisionforest::numberOfKernelRuns;
    mlir::decisionforest::numberOfKernelRuns = NUM_RUNS;

    double bestKernelTime = std::numeric_limits<double>::max();
    // Caching rows seems generally better than not. So we will
    // first explore schedules with row caching and then disable
    // if needed if we want to enable shared reduction.
    for (auto numRowsPerTB : rowsPerTB) {
      for (auto numRowsPerThread : rowsPerThread) {
        if (numRowsPerThread >= numRowsPerTB)
          break;
        for (auto treeThreads : numTreeThreads) {
          auto tbSize = numRowsPerTB / numRowsPerThread * treeThreads;
          if (tbSize > MAX_TB_SIZE)
            continue;
          if (tbSize < MIN_TB_SIZE)
            continue;
          for (auto rep : {"gpu_array", "gpu_sparse", "gpu_reorg"}) {
            if (m_shouldCacheRows)
              runBenchmarks<true>(bestKernelTime, rep, numRowsPerTB,
                                  numRowsPerThread, treeThreads);
            else
              runBenchmarks<false>(bestKernelTime, rep, numRowsPerTB,
                                   numRowsPerThread, treeThreads);
          }
        }
      }
    }

    // Now check if we need to do a shared reduction.
    trySharedReduce(bestKernelTime);

    mlir::decisionforest::numberOfKernelRuns = numKernelRuns;

    std::cerr << m_model << "\t" << m_batchSize << std::endl;
    std::cerr << "Best kernel execution time: " << bestKernelTime << std::endl;
  }

  void printBestSchedule() {
    std::cerr << "Best schedule for " << m_model << " is: " << std::endl;
    std::cerr << "\tnumRowsPerTB: " << m_bestOptions.numRowsPerTB << std::endl;
    std::cerr << "\tnumRowsPerThread: " << m_bestOptions.numRowsPerThread
              << std::endl;
    std::cerr << "\tnumTreeThreads: " << m_bestOptions.numTreeThreads
              << std::endl;
    std::cerr << "\tcacheRows: " << m_bestOptions.cacheRows << std::endl;
    std::cerr << "\tcacheTrees: " << m_bestOptions.cacheTrees << std::endl;
    std::cerr << "\tunrollTreeWalks: " << m_bestOptions.unrollTreeWalks
              << std::endl;
    std::cerr << "\tinterleaveDepth: " << m_bestOptions.treeWalkInterleaveFactor
              << std::endl;
    std::cerr << "\tsharedMemoryReduce: " << m_bestOptions.sharedMemoryReduce
              << std::endl;
    std::cerr << "\tRepresentation: " << m_bestRep << std::endl;
  }

  TreeBeard::GPUAutoSchedulerResults getResult() {
    return TreeBeard::GPUAutoSchedulerResults{
        m_bestOptions, m_bestOptions.unrollTreeWalks, m_bestRep};
  }
};

} // anonymous namespace

namespace TreeBeard {

void DoGPUAutoSchedule(mlir::MLIRContext &context, mlir::ModuleOp module,
                       const TreeBeard::GPUAutoScheduleOptions &options) {
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(std::make_unique<ReorderTreesByDepthPass>());
  // TODO pipelineSize needs to be added to CompilerOptions
  optPM.addPass(std::make_unique<SplitTreeLoopByDepth>(options));

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to mid level IR failed.\n";
  }
}

GPUAutoSchedulerResults findBestGPUSchedule(const std::string &benchmark,
                                            int32_t batchSize) {
  GPUAutoTuner tuner(benchmark, batchSize);
  tuner.exploreSchedules();
  tuner.printBestSchedule();
  return tuner.getResult();
}

} // namespace TreeBeard
