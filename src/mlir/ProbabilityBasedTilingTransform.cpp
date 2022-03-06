// Implementation of a transformation to tile all trees in a forest based on edge probabilities. 
// The tile needs to be constant across trees.

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
#include <set>
#include <cassert>
#include "Logger.h"

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
  Type m_tileShapeType;
  bool m_hybrid;
  double m_tilingThreshold;

  TileEnsembleConstants(MLIRContext *ctx, int32_t tileSize, Type tileShapeType) 
    : RewritePattern(mlir::decisionforest::EnsembleConstantOp::getOperationName(), 1 /*benefit*/, ctx),
      m_tileSize(tileSize), m_tileShapeType(tileShapeType), m_hybrid(false), m_tilingThreshold(0.0)
  {}

  TileEnsembleConstants(MLIRContext *ctx, int32_t tileSize, Type tileShapeType, bool hybrid, double tilingThreshold) 
    : RewritePattern(mlir::decisionforest::EnsembleConstantOp::getOperationName(), 1 /*benefit*/, ctx),
      m_tileSize(tileSize), m_tileShapeType(tileShapeType), m_hybrid(hybrid), m_tilingThreshold(tilingThreshold)
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
    int32_t numTreesTiledProbabilistically = 0;
    for (int64_t i=0 ; i<(int64_t)forest.NumTrees() ; ++i) {
      forest.GetTree(i).InitializeInternalNodeHitCounts();
      
      auto tiledProbabilistically = TileSingleDecisionTree(forest.GetTree(i));
      numTreesTiledProbabilistically += tiledProbabilistically ? 1 : 0;
      // if (tiledProbabilistically) 
      //   std::cout << i << " ";

      auto treeType = forestType.getTreeType(i).cast<decisionforest::TreeType>();
      auto newTreeType = decisionforest::TreeType::get(treeType.getResultType(), forest.GetTree(i).TilingDescriptor().MaxTileSize(), 
                                                       treeType.getThresholdType(), treeType.getFeatureIndexType(), m_tileShapeType, 
                                                       treeType.isSparseRepresentation(), treeType.getChildIndexType());
      treeTypes.push_back(newTreeType);
    }
    // std::cout << std::endl;

    TreeBeard::Logging::Log("Number of trees tiled probabilistically : " + std::to_string(numTreesTiledProbabilistically));
    // Tile this forest uniformly
    auto newForestType = decisionforest::TreeEnsembleType::get(forestType.getResultType(), forestType.getNumberOfTrees(),
                                                               forestType.getRowType(), forestType.getReductionType(), treeTypes);

    auto newForestAttribute = decisionforest::DecisionForestAttribute::get(newForestType, forest);
    auto newConstant = rewriter.create<decisionforest::EnsembleConstantOp>(op->getLoc(), newForestType, newForestAttribute);

    // Go over all the uses of the constant. These must be "GetTree". Change the result type
    auto uses = op->getResult(0).getUses();
    for (auto& use : uses) {
      Operation *useOp = use.getOwner();
      if (auto getTreeOp = llvm::dyn_cast<decisionforest::GetTreeFromEnsembleOp>(useOp)) {
        auto oldTreeType = getTreeOp.getResult().getType().cast<decisionforest::TreeType>();
        auto newTreeType = decisionforest::TreeType::get(oldTreeType.getResultType(), m_tileSize, oldTreeType.getThresholdType(), oldTreeType.getFeatureIndexType());

        mlir::OpBuilder::InsertionGuard insertionGuard(rewriter);
        rewriter.setInsertionPointAfter(useOp);
        auto newGetTreeOp = rewriter.create<decisionforest::GetTreeFromEnsembleOp>(location, newTreeType, static_cast<Value>(newConstant), getTreeOp.treeIndex());
        rewriter.replaceOp(useOp, static_cast<Value>(newGetTreeOp));
      }
    }
    rewriter.replaceOp(op, static_cast<Value>(newConstant));
    return mlir::success();
  }

  void DoTileTraversalForNode_Uniform(const std::vector<decisionforest::DecisionTree<>::Node>& nodes, 
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

  void DoTileTraversalForNode(const std::vector<decisionforest::DecisionTree<>::Node>& nodes, 
                              int32_t currentNode, int32_t tileID, std::vector<int32_t>& tileIDs) const {
    auto& currNode = nodes.at(currentNode);
    auto& rootNode = nodes.at(0);
    double nodeProbability = (double)currNode.hitCount/(double)rootNode.hitCount;
    if (m_hybrid && nodeProbability<m_tilingThreshold) {
      DoTileTraversalForNode_Uniform(nodes, currentNode, tileID, tileIDs);
      return;
    }
    std::set<int32_t> tileNodes;
    tileNodes.insert(currentNode);
    tileIDs.at(currentNode) = tileID;
    int32_t numNodes = 1;
    while (numNodes < m_tileSize) {
      int32_t maxHitCount = -1;
      int32_t maxProbabilityNode = -1;
      for (auto nodeIndex : tileNodes) {
        auto& node = nodes.at(nodeIndex);
        if (node.leftChild != decisionforest::DecisionTree<>::INVALID_NODE_INDEX) {
          auto &child = nodes.at(node.leftChild);
          if (!child.IsLeaf() && child.hitCount > maxHitCount && tileNodes.find(node.leftChild)==tileNodes.end()) {
            maxHitCount = child.hitCount;
            maxProbabilityNode = node.leftChild;
          }
        }
        if (node.rightChild != decisionforest::DecisionTree<>::INVALID_NODE_INDEX) {
          auto &child = nodes.at(node.rightChild);
          if (!child.IsLeaf() && child.hitCount > maxHitCount && tileNodes.find(node.rightChild)==tileNodes.end()) {
            maxHitCount = child.hitCount;
            maxProbabilityNode = node.rightChild;
          }
        }
      }
      if (maxProbabilityNode == -1)
        break;
      tileNodes.insert(maxProbabilityNode);
      tileIDs.at(maxProbabilityNode) = tileID;
      ++numNodes;
      assert ((int32_t)tileNodes.size() == numNodes);
    }
  }

  void ConstructTileIDVector(const std::vector<decisionforest::DecisionTree<>::Node>& nodes,
                             int32_t currentNode, int32_t& tileID, std::vector<int32_t>& tileIDs, bool skewed) const {
    if (currentNode == decisionforest::DecisionTree<>::INVALID_NODE_INDEX)
      return;
    if (tileIDs.at(currentNode) == -1) {
      if (skewed)
        DoTileTraversalForNode(nodes, currentNode, tileID, tileIDs);
      else
        DoTileTraversalForNode_Uniform(nodes, currentNode, tileID, tileIDs);
      ++tileID;
    }
    ConstructTileIDVector(nodes, nodes.at(currentNode).leftChild, tileID, tileIDs, skewed);
    ConstructTileIDVector(nodes, nodes.at(currentNode).rightChild, tileID, tileIDs, skewed);
  }

  bool IsDecisionTreeSkewed(decisionforest::DecisionTree<>& tree) const {
    auto& root = tree.GetNodes().at(0);
    std::vector<mlir::decisionforest::DecisionTree<>::Node> leaves;
    for (auto& node : tree.GetNodes()) {
      if (node.IsLeaf())
        leaves.push_back(node);
    }
    std::sort(leaves.begin(), leaves.end(), [](mlir::decisionforest::DecisionTree<>::Node& n1, mlir::decisionforest::DecisionTree<>::Node& n2) {
      return n1.hitCount > n2.hitCount;
    });
    double fractionOfLeaves=0.10, threshold=0.9;
    int32_t numLeavesToConsider = leaves.size() * fractionOfLeaves;
    int32_t hits = 0;
    for (int32_t i=0 ; i<numLeavesToConsider ; ++i) {
      hits += leaves.at(i).hitCount;
    }
    return (double)hits/(double)root.hitCount > threshold;
  }

  bool TileSingleDecisionTree(decisionforest::DecisionTree<>& tree) const {
    const auto& nodes = tree.GetNodes();
    std::vector<int32_t> tileIDs(nodes.size(), -1);
    int32_t tileID = 0;
    bool isTreeSkewed = IsDecisionTreeSkewed(tree);
    ConstructTileIDVector(nodes, 0, tileID, tileIDs, isTreeSkewed);
    assert (std::find(tileIDs.begin(), tileIDs.end(), -1) == tileIDs.end());
    decisionforest::TreeTilingDescriptor tilingDescriptor(m_tileSize, -1, tileIDs, decisionforest::TilingType::kRegular);
    tree.SetTilingDescriptor(tilingDescriptor);
    return isTreeSkewed;
  }
};

struct ProbabilityBasedTilingPass : public PassWrapper<ProbabilityBasedTilingPass, FunctionPass> {
  int32_t m_tileSize;
  int32_t m_tileShapeBitWidth;
  bool m_hybrid;
  double m_tilingThreshold;
  ProbabilityBasedTilingPass(int32_t tileSize, int32_t tileShapeBitWidth) 
    : m_tileSize(tileSize), m_tileShapeBitWidth(tileShapeBitWidth), m_hybrid(false), m_tilingThreshold(0.0)
  { }
  ProbabilityBasedTilingPass(int32_t tileSize, int32_t tileShapeBitWidth, bool hybrid, double tilingThreshold) 
    : m_tileSize(tileSize), m_tileShapeBitWidth(tileShapeBitWidth), m_hybrid(hybrid), m_tilingThreshold(tilingThreshold)
  { }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, StandardOpsDialect, scf::SCFDialect, math::MathDialect>();
  }
  void runOnFunction() final {
    RewritePatternSet patterns(&getContext());
    auto tileShapeType = IntegerType::get(&getContext(), m_tileShapeBitWidth);
    patterns.add<TileEnsembleConstants>(&getContext(), m_tileSize, tileShapeType, m_hybrid, m_tilingThreshold);

    if (failed(applyPatternsAndFoldGreedily(getFunction(), std::move(patterns))))
        signalPassFailure();
  }
};
} // anonymous namespace

namespace mlir
{
namespace decisionforest
{
void DoProbabilityBasedTiling(mlir::MLIRContext& context, mlir::ModuleOp module, int32_t tileSize, int32_t tileShapeBitWidth) {
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  optPM.addPass(std::make_unique<ProbabilityBasedTilingPass>(tileSize, tileShapeBitWidth));

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to mid level IR failed.\n";
  }
}

void DoHybridTiling(mlir::MLIRContext& context, mlir::ModuleOp module, int32_t tileSize, int32_t tileShapeBitWidth) {
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  optPM.addPass(std::make_unique<ProbabilityBasedTilingPass>(tileSize, tileShapeBitWidth, true, 0.20));

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to mid level IR failed.\n";
  }
}

} // decisionforest
} // mlir
