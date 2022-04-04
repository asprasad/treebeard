#ifndef TRAVERSE_TREE_TILE_OP_LOWERING
#define TRAVERSE_TREE_TILE_OP_LOWERING

#include "CodeGenStateMachine.h"

namespace mlir
{
namespace decisionforest
{
    // TODO - Unify this with the regular traverse tile lowering.
    class InterleavedTraverseTreeTileOpLoweringHelper {
        private:
            std::function<Value(Value)> m_getTreeMemref;
            std::function<Value(Value)> m_getLutFromTree;
        public:
        InterleavedTraverseTreeTileOpLoweringHelper(std::function<Value(Value)> getTreeMemref, std::function<Value(Value)> getLutFromTree) : m_getTreeMemref(getTreeMemref), m_getLutFromTree(getLutFromTree) {}

        LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,ConversionPatternRewriter &rewriter) const final {
            auto traverseTileOp = AssertOpIsOfType<mlir::decisionforest::InterleavedTraverseTreeTileOp>(op);
            if (!traverseTileOp)
                return mlir::failure();
            
            auto trees = traverseTileOp.trees();
            auto nodes = traverseTileOp.nodes();
            auto dataRows = traverseTileOp.data();

            assert(nodes.size() == trees.size());
            assert(trees.size() == dataRows.size());

            decisionforest::InterleavedCodeGenStateMachine codeGenStateMachine;
            for (size_t i = 0; i < trees.size(); i++) {
            auto tree = trees[i];
            auto node = nodes[i];
            auto data = dataRows[i];

            auto treeMemref = m_getTreeMemref(tree);
            auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
            assert (treeMemrefType);

            auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();

            // TODO - tile size should be same for all iterations. Need to assert this somehow.
            if (treeTileType.getTileSize() == 1) {
                codeGenStateMachine.AddStateMachine(
                std::make_unique<decisionforest::ScalarTraverseTileCodeGenerator>(
                    treeMemref,
                    data,
                    node,
                    traverseTileOp.getResult(i).getType(),
                    decisionforest::Representation::kArray));
            }
            else {
                codeGenStateMachine.AddStateMachine(
                std::make_unique<decisionforest::VectorTraverseTileCodeGenerator>(
                    tree,
                    treeMemref,
                    data,
                    node,
                    traverseTileOp.getResult(i).getType(),
                    decisionforest::Representation::kArray,
                    m_getLutFromTree));
            }
            }

            auto location = op->getLoc();
            while (codeGenStateMachine.EmitNext(rewriter, location));

            rewriter.replaceOp(op, codeGenStateMachine.GetResult());
            return mlir::success();
        }
    };
} // decisionforest
} // mlir

#endinf // TRAVERSE_TREE_TILE_OP_LOWERING