#ifndef TRAVERSE_TREE_TILE_OP_LOWERING
#define TRAVERSE_TREE_TILE_OP_LOWERING

#include "CodeGenStateMachine.h"
#include "OpLoweringUtils.h"

namespace mlir
{
namespace decisionforest
{
    // TODO - Unify this with the regular traverse tile lowering.
    class InterleavedTraverseTreeTileOpLoweringHelper {
        private:
        std::function<Value(Value)> m_getLutFromTree;
        std::shared_ptr<decisionforest::IRepresentation> m_representation;
        
        public:
        InterleavedTraverseTreeTileOpLoweringHelper(
            std::function<Value(Value)> getLutFromTree,
            std::shared_ptr<decisionforest::IRepresentation> representation)
            : m_getLutFromTree(getLutFromTree),
              m_representation(representation) {}

        LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,ConversionPatternRewriter &rewriter) const {
            auto traverseTileOp = AssertOpIsOfType<mlir::decisionforest::InterleavedTraverseTreeTileOp>(op);
            if (!traverseTileOp)
                return mlir::failure();
            
            auto trees = traverseTileOp.getTrees();
            auto nodes = traverseTileOp.getNodes();
            auto dataRows = traverseTileOp.getData();

            assert(nodes.size() == trees.size());
            assert(trees.size() == dataRows.size());

            decisionforest::InterleavedCodeGenStateMachine codeGenStateMachine;
            for (size_t i = 0; i < trees.size(); i++) {
                auto tree = trees[i];
                auto node = nodes[i];
                auto data = dataRows[i];

                auto treeMemref = m_representation->GetTreeMemref(tree);
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
                        m_representation));
                }
                else {
                    codeGenStateMachine.AddStateMachine(
                    std::make_unique<decisionforest::VectorTraverseTileCodeGenerator>(
                        tree,
                        treeMemref,
                        data,
                        node,
                        traverseTileOp.getResult(i).getType(),
                        m_representation,
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

#endif // TRAVERSE_TREE_TILE_OP_LOWERING