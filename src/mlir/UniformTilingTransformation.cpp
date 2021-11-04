// Implementation of a transformation to uniformly tile all trees in a forest. 
// The "uniform" represents that assumption that all edge probabilities are the same. 
// The tile size also needs to be constant across trees.

#include "Dialect.h"
// #include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <queue>
#include <cassert>

using namespace mlir;
namespace {

template<typename T>
T AssertOpIsOfType(Operation* operation) {
  T typedOp = llvm::dyn_cast<T>(operation);
  assert(typedOp);
  return typedOp;
}

struct TileEnsembleConstants : public RewritePattern {
  int32_t m_tileSize;
  TileEnsembleConstants(MLIRContext *ctx, int32_t tileSize) 
    : RewritePattern(mlir::decisionforest::EnsembleConstantOp::getOperationName(), 1 /*benefit*/, ctx), m_tileSize(tileSize)
  {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const final {
    Location location = op->getLoc();
    mlir::decisionforest::EnsembleConstantOp constantOp = llvm::dyn_cast<mlir::decisionforest::EnsembleConstantOp>(op);
    assert(constantOp);
    if (!constantOp)
         return mlir::failure();

    auto forestAttribute = constantOp.forest();
    auto forest = forestAttribute.GetDecisionForest();
    auto forestType = constantOp.getType().cast<decisionforest::TreeEnsembleType>();
    auto tilingDescriptor = forest.GetTree(0).TilingDescriptor();
    // If we've already tiled this forest, then don't tile it again!
    if (tilingDescriptor.MaxTileSize() == m_tileSize)
      return mlir::failure();

    assert (tilingDescriptor.MaxTileSize() == 1 && "Forest shouldn't already be tiled!");
    std::vector<Type> treeTypes;
    for (int64_t i=0 ; i<(int64_t)forest.NumTrees() ; ++i) {
      TileSingleDecisionTree(forest.GetTree(i));
      auto treeType = forestType.getTreeType(i).cast<decisionforest::TreeType>();
      auto newTreeType = decisionforest::TreeType::get(treeType.getResultType(), forest.GetTree(i).TilingDescriptor().MaxTileSize(), 
                                                       treeType.getThresholdType(), treeType.getFeatureIndexType());
      treeTypes.push_back(newTreeType);
    }
    // Tile this forest uniformly
    auto newForestType = decisionforest::TreeEnsembleType::get(forestType.getResultType(), forestType.getNumberOfTrees(),
                                                               forestType.getRowType(), forestType.getReductionType(), treeTypes);

    auto newForestAttribute = decisionforest::DecisionForestAttribute::get(newForestType, forest);
    auto newConstant = rewriter.create<decisionforest::EnsembleConstantOp>(op->getLoc(), newForestType, newForestAttribute);

    // Go over all the uses of the constant. These must be "GetTree". Change the result type
    auto uses = op->getResult(0).getUses();
    for (auto& use : uses) {
      Operation *useOp = use.getOwner();
      auto getTreeOp = AssertOpIsOfType<decisionforest::GetTreeFromEnsembleOp>(useOp);
      auto oldTreeType = getTreeOp.getResult().getType().cast<decisionforest::TreeType>();
      auto newTreeType = decisionforest::TreeType::get(oldTreeType.getResultType(), m_tileSize, oldTreeType.getThresholdType(), oldTreeType.getFeatureIndexType());
      
      mlir::OpBuilder::InsertionGuard insertionGuard(rewriter);
      rewriter.setInsertionPointAfter(useOp);
      auto newGetTreeOp = rewriter.create<decisionforest::GetTreeFromEnsembleOp>(location, newTreeType, static_cast<Value>(newConstant), getTreeOp.treeIndex());
      rewriter.replaceOp(useOp, static_cast<Value>(newGetTreeOp));
    }
    rewriter.replaceOp(op, static_cast<Value>(newConstant));
    return mlir::success();
  }

  void DoTileTraversalForNode(const std::vector<decisionforest::DecisionTree<>::Node>& nodes, 
                              int32_t currentNode, int32_t tileID, std::vector<int32_t>& tileIDs) const {
    std::queue<int32_t> nodeQ;
    nodeQ.push(currentNode);
    int32_t numNodes = 0;
    while (!nodeQ.empty() && numNodes < m_tileSize) {
      auto node = nodeQ.front();
      nodeQ.pop();
      ++numNodes;
      tileIDs.at(node) = tileID;
      auto leftChild = nodes.at(node).leftChild;
      if (leftChild != decisionforest::DecisionTree<>::INVALID_NODE_INDEX && !nodes.at(leftChild).IsLeaf())
        nodeQ.push(leftChild);
      auto rightChild = nodes.at(node).rightChild;
      if (rightChild != decisionforest::DecisionTree<>::INVALID_NODE_INDEX && !nodes.at(rightChild).IsLeaf())
        nodeQ.push(rightChild);
    }
  }

  void ConstructTileIDVector(const std::vector<decisionforest::DecisionTree<>::Node>& nodes,
                             int32_t currentNode, int32_t& tileID, std::vector<int32_t>& tileIDs) const {
    if (currentNode == decisionforest::DecisionTree<>::INVALID_NODE_INDEX)
      return;
    if (tileIDs.at(currentNode) == -1) {
      DoTileTraversalForNode(nodes, currentNode, tileID, tileIDs);
      ++tileID;
    }
    ConstructTileIDVector(nodes, nodes.at(currentNode).leftChild, tileID, tileIDs);
    ConstructTileIDVector(nodes, nodes.at(currentNode).rightChild, tileID, tileIDs);
  }

  void TileSingleDecisionTree(decisionforest::DecisionTree<>& tree) const {
    const auto& nodes = tree.GetNodes();
    std::vector<int32_t> tileIDs(nodes.size(), -1);
    int32_t tileID = 0;
    ConstructTileIDVector(nodes, 0, tileID, tileIDs);
    decisionforest::TreeTilingDescriptor tilingDescriptor(m_tileSize, -1, tileIDs, decisionforest::TilingType::kRegular);
    tree.SetTilingDescriptor(tilingDescriptor);
  }
};

struct UniformTilingPass : public PassWrapper<UniformTilingPass, FunctionPass> {
  int32_t m_tileSize;
  UniformTilingPass(int32_t tileSize) : m_tileSize(tileSize) { }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, StandardOpsDialect, scf::SCFDialect>();
  }
  void runOnFunction() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TileEnsembleConstants>(&getContext(), m_tileSize);

    if (failed(applyPatternsAndFoldGreedily(getFunction(), std::move(patterns))))
        signalPassFailure();
  }
};
} // anonymous namespace

namespace mlir
{
namespace decisionforest
{
void DoUniformTiling(mlir::MLIRContext& context, mlir::ModuleOp module, int32_t tileSize) {
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  optPM.addPass(std::make_unique<UniformTilingPass>(tileSize));

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to mid level IR failed.\n";
  }
}

} // decisionforest
} // mlir
