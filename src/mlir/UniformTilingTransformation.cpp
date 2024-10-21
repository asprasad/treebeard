// Implementation of a transformation to uniformly tile all trees in a forest.
// The "uniform" represents that assumption that all edge probabilities are the
// same. The tile size also needs to be constant across trees.

#include <optional>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "Dialect.h"
#include "OpLoweringUtils.h"
#include "TiledTree.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cassert>
#include <queue>

namespace mlir {
namespace decisionforest {

struct PadEnsemblePattern : public RewritePattern {

  PadEnsemblePattern(MLIRContext *ctx)
      : RewritePattern(
            mlir::decisionforest::PredictForestOp::getOperationName(),
            1 /*benefit*/, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    auto predictForestOp =
        llvm::dyn_cast<mlir::decisionforest::PredictForestOp>(op);
    assert(predictForestOp);
    if (!predictForestOp)
      return mlir::failure();

    const std::string allLeavesSameDepthAttrName = "allLeavesSameDepth";
    if (predictForestOp->getAttr(allLeavesSameDepthAttrName))
      return mlir::failure();

    auto forestAttribute = predictForestOp.getEnsemble();
    auto forest = forestAttribute.GetDecisionForest();
    auto forestType =
        forestAttribute.getType().cast<decisionforest::TreeEnsembleType>();
    auto tilingDescriptor = forest.GetTree(0).TilingDescriptor();
    auto tileSize = tilingDescriptor.MaxTileSize();
    for (int64_t i = 0; i < (int64_t)forest.NumTrees(); ++i) {
      if (tileSize > 1) {
        forest.GetTree(i).GetTiledTree()->MakeAllLeavesSameDepth();
      } else {
        forest.GetTree(i).MakeAllLeavesSameDepth();
      }
    }

    auto newForestAttribute =
        decisionforest::DecisionForestAttribute::get(forestType, forest);
    auto tiledPredictForestOp =
        rewriter.create<decisionforest::PredictForestOp>(
            op->getLoc(), predictForestOp.getResult().getType(),
            newForestAttribute, predictForestOp.getPredicateAttr(),
            predictForestOp.getData(), predictForestOp.getResult(),
            predictForestOp.getSchedule());

    tiledPredictForestOp->setAttr(allLeavesSameDepthAttrName,
                                  rewriter.getUnitAttr());

    rewriter.replaceOp(op, static_cast<Value>(tiledPredictForestOp));
    return mlir::success();
  }
};

struct TileEnsembleAttribute : public RewritePattern {
  int32_t m_tileSize;
  Type m_tileShapeType;
  bool m_makeAllLeavesSameDepth;

  TileEnsembleAttribute(MLIRContext *ctx, int32_t tileSize, Type tileShapeType,
                        bool makeAllLeavesSameDepth)
      : RewritePattern(
            mlir::decisionforest::PredictForestOp::getOperationName(),
            1 /*benefit*/, ctx),
        m_tileSize(tileSize), m_tileShapeType(tileShapeType),
        m_makeAllLeavesSameDepth(makeAllLeavesSameDepth) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    auto predictForestOp =
        llvm::dyn_cast<mlir::decisionforest::PredictForestOp>(op);
    assert(predictForestOp);
    if (!predictForestOp)
      return mlir::failure();

    auto forestAttribute = predictForestOp.getEnsemble();
    auto forest = forestAttribute.GetDecisionForest();
    auto forestType =
        forestAttribute.getType().cast<decisionforest::TreeEnsembleType>();
    auto tilingDescriptor = forest.GetTree(0).TilingDescriptor();
    // If we've already tiled this forest, then don't tile it again!
    if (tilingDescriptor.MaxTileSize() == m_tileSize)
      return mlir::failure();

    assert(tilingDescriptor.MaxTileSize() == 1 &&
           "Forest shouldn't already be tiled!");
    std::vector<Type> treeTypes;
    for (int64_t i = 0; i < (int64_t)forest.NumTrees(); ++i) {
      forest.GetTree(i).InitializeInternalNodeHitCounts();
      TileSingleDecisionTree(forest.GetTree(i));
      auto treeType =
          forestType.getTreeType(i).cast<decisionforest::TreeType>();
      auto newTreeType = decisionforest::TreeType::get(
          treeType.getResultType(),
          forest.GetTree(i).TilingDescriptor().MaxTileSize(),
          treeType.getThresholdType(), treeType.getFeatureIndexType(),
          m_tileShapeType, treeType.getChildIndexType());
      treeTypes.push_back(newTreeType);
      if (i != 0)
        assert(treeTypes.at(0) == newTreeType);

      if (m_makeAllLeavesSameDepth) {
        forest.GetTree(i).GetTiledTree()->MakeAllLeavesSameDepth();
      }
    }
    // Tile this forest uniformly
    auto newForestType = decisionforest::TreeEnsembleType::get(
        forestType.getResultType(), forestType.getNumberOfTrees(),
        forestType.getRowType(), forestType.getReductionType(),
        treeTypes.at(0));

    auto newForestAttribute =
        decisionforest::DecisionForestAttribute::get(newForestType, forest);
    auto tiledPredictForestOp =
        rewriter.create<decisionforest::PredictForestOp>(
            op->getLoc(), predictForestOp.getResult().getType(),
            newForestAttribute, predictForestOp.getPredicateAttr(),
            predictForestOp.getData(), predictForestOp.getResult(),
            predictForestOp.getSchedule());

    rewriter.replaceOp(op, static_cast<Value>(tiledPredictForestOp));
    return mlir::success();
  }

  void DoTileTraversalForNode(
      const std::vector<decisionforest::DecisionTree::Node> &nodes,
      int32_t currentNode, int32_t tileID,
      std::vector<int32_t> &tileIDs) const {
    std::queue<int32_t> nodeQ;
    nodeQ.push(currentNode);
    int32_t numNodes = 0;
    while (!nodeQ.empty() && numNodes < m_tileSize) {
      auto node = nodeQ.front();
      nodeQ.pop();
      ++numNodes;
      tileIDs.at(node) = tileID;
      auto leftChild = nodes.at(node).leftChild;
      if (leftChild != decisionforest::DecisionTree::INVALID_NODE_INDEX &&
          !nodes.at(leftChild).IsLeaf())
        nodeQ.push(leftChild);
      auto rightChild = nodes.at(node).rightChild;
      if (rightChild != decisionforest::DecisionTree::INVALID_NODE_INDEX &&
          !nodes.at(rightChild).IsLeaf())
        nodeQ.push(rightChild);
    }
  }

  void ConstructTileIDVector(
      const std::vector<decisionforest::DecisionTree::Node> &nodes,
      int32_t currentNode, int32_t &tileID,
      std::vector<int32_t> &tileIDs) const {
    if (currentNode == decisionforest::DecisionTree::INVALID_NODE_INDEX)
      return;
    if (tileIDs.at(currentNode) == -1) {
      DoTileTraversalForNode(nodes, currentNode, tileID, tileIDs);
      ++tileID;
    }
    ConstructTileIDVector(nodes, nodes.at(currentNode).leftChild, tileID,
                          tileIDs);
    ConstructTileIDVector(nodes, nodes.at(currentNode).rightChild, tileID,
                          tileIDs);
  }

  void TileSingleDecisionTree(decisionforest::DecisionTree &tree) const {
    const auto &nodes = tree.GetNodes();
    std::vector<int32_t> tileIDs(nodes.size(), -1);
    int32_t tileID = 0;
    ConstructTileIDVector(nodes, 0, tileID, tileIDs);
    decisionforest::TreeTilingDescriptor tilingDescriptor(
        m_tileSize, -1, tileIDs, decisionforest::TilingType::kRegular);
    tree.SetTilingDescriptor(tilingDescriptor);
  }
};

struct UniformTilingPass
    : public PassWrapper<UniformTilingPass, OperationPass<mlir::ModuleOp>> {
  int32_t m_tileSize;
  int32_t m_tileShapeBitWidth;
  bool m_makeAllLeavesSameDepth;
  UniformTilingPass(int32_t tileSize, int32_t tileShapeBitWidth,
                    bool makeAllLeavesSameDepth)
      : m_tileSize(tileSize), m_tileShapeBitWidth(tileShapeBitWidth),
        m_makeAllLeavesSameDepth(makeAllLeavesSameDepth) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, scf::SCFDialect,
                    math::MathDialect>();
  }
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    auto tileShapeType = IntegerType::get(&getContext(), m_tileShapeBitWidth);
    patterns.add<TileEnsembleAttribute>(
        &getContext(), m_tileSize, tileShapeType, m_makeAllLeavesSameDepth);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

struct PadTreesToMakeAllLeavesSameDepthPass
    : public PassWrapper<PadTreesToMakeAllLeavesSameDepthPass,
                         OperationPass<mlir::func::FuncOp>> {
  PadTreesToMakeAllLeavesSameDepthPass() {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, scf::SCFDialect,
                    math::MathDialect>();
  }
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<PadEnsemblePattern>(&getContext());

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace decisionforest
} // namespace mlir

namespace mlir {
namespace decisionforest {
void DoUniformTiling(mlir::MLIRContext &context, mlir::ModuleOp module,
                     int32_t tileSize, int32_t tileShapeBitWidth,
                     bool makeAllLeavesSameDepth) {
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
  pm.addPass(std::make_unique<UniformTilingPass>(tileSize, tileShapeBitWidth,
                                                 makeAllLeavesSameDepth));

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Uniform tiling pass failed.\n";
  }
}

void padTreesToMakeAllLeavesSameDepth(mlir::MLIRContext &context,
                                      mlir::ModuleOp module) {
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
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(std::make_unique<PadTreesToMakeAllLeavesSameDepthPass>());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Padding trees to make leaves same depth failed.\n";
  }
}

} // namespace decisionforest
} // namespace mlir
