#include <iostream>
#include "Dialect.h"
#include <mutex>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/Debug.h"

#include "MemrefTypes.h"
#include "Dialect.h"
#include "TreeTilingUtils.h"

#include "Logger.h"
#include "OpLoweringUtils.h"

namespace mlir {
namespace decisionforest {

struct NodeToIndexOpLowering : public ConversionPattern {
  NodeToIndexOpLowering(MLIRContext *ctx) : ConversionPattern(mlir::decisionforest::NodeToIndexOp::getOperationName(), 1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    auto nodeToIndexOp = AssertOpIsOfType<mlir::decisionforest::NodeToIndexOp>(op);
    assert(operands.size() == 2);
    if (!nodeToIndexOp)
        return mlir::failure();
    // The argument of a NodeToIndex op can only be 
    // 1. An IndexToNodeOp
    // 2. A block argument
    // If it is #1, then the IndextoNodeOp should already have been replaced and we should replace the NodeToIndexOp with the 
    // corresponding index value of the indexToNode op. If it is #2, then we should have already replaced the node argument 
    // with an index value and we just replace the Op with this index value.
    auto indexValue = operands[1];
    if (indexValue.getType().isa<mlir::IndexType>()) {
        rewriter.replaceOp(op, indexValue);
    }
    else {
        assert (false && "Expected node value to be replaced by an index value by now!");
        return mlir::failure();
    }
    return mlir::success();
  }
};

struct IndexToNodeOpLowering : public ConversionPattern {
  IndexToNodeOpLowering(MLIRContext *ctx) : ConversionPattern(mlir::decisionforest::IndexToNodeOp::getOperationName(), 1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    auto indexToNodeOp = AssertOpIsOfType<mlir::decisionforest::IndexToNodeOp>(op);
    assert(operands.size() == 2);
    if (!indexToNodeOp)
        return mlir::failure();
    // The uses of an IndexToNode op can only be 
    // 1. A NodeToIndexOp
    // 2. A block argument
    // If the source of the node argument is a NodeToIndexOp, we need to replace the IndexToNodeOp with the index from nodeToIndexOpToIndexValueMap.
    // If the source is anything else, then just replace uses with argument.
    auto indexValue = operands[1];
    if (auto nodeToIndexOp = llvm::dyn_cast<decisionforest::NodeToIndexOp>(op)) {
      assert (false && "The source of an index value should not be a NodeToIndexOp here!");
      return mlir::failure();
    }
    else {
      assert (indexValue.getType().isa<mlir::IndexType>());
      rewriter.replaceOp(op, indexValue);
    }
    return mlir::success();
  }
};

struct ConvertNodeTypeToIndexTypePass : public PassWrapper<ConvertNodeTypeToIndexTypePass, OperationPass<mlir::func::FuncOp>> {
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, tensor::TensorDialect, scf::SCFDialect>();
  }
  
  void VisitOp(Operation* op, OpBuilder& builder) {
    for (auto &region : op->getRegions()) {
        for (auto &block : region.getBlocks()) {
            // Collect the set of arguments for which index arguments are needed
            std::list<size_t> indexArgInsertionIndices;
            for (size_t i=0 ; i<block.getNumArguments() ; ++i) {
                auto arg = block.getArgument(i);
                if (arg.getType().isa<decisionforest::NodeType>())
                    indexArgInsertionIndices.push_back(i);
            }

            for (auto index : indexArgInsertionIndices){
                auto newArgument = block.insertArgument(index, builder.getIndexType(), op->getLoc());
                auto oldArgument = block.getArgument(index+1);
                assert(oldArgument.getType().isa<decisionforest::NodeType>());
                oldArgument.replaceAllUsesWith(newArgument);
                block.eraseArgument(index+1);
            }

            for (auto& op : block.getOperations()) {
                VisitOp(&op, builder);
            }  
        }
    }
    if (!(llvm::dyn_cast<decisionforest::NodeToIndexOp>(op)) && !(llvm::dyn_cast<decisionforest::IndexToNodeOp>(op))) {
        for (auto result : op->getResults()) {
            if (result.getType().isa<decisionforest::NodeType>()) {
                assert (llvm::dyn_cast<scf::WhileOp>(op) && "Expected op to be an scf::WhileOp");
                // TreeBeard::Logging::Log("Calling setType on result of : " + op->getName().getStringRef().str());
                // TODO check that this is some op with non trivial control flow. 
                // assert(op->getRegions().size() > 1);
                // TODO Is there a way we can avoid calling setType here?
                result.setType(builder.getIndexType());
            }
        }
        for (auto operand : op->getOperands()) {
          if (operand.getType().isa<decisionforest::NodeType>()) {
            // TreeBeard::Logging::Log("Calling setType on argument of : " + op->getName().getStringRef().str());
            operand.setType(mlir::IndexType::get(op->getContext()));
          }
        }
    }
  }

  void AddIndexArgumentsForAllBlocks() {
      OpBuilder builder(&getContext());
      auto func = this->getOperation();
      VisitOp(func, builder);
  }

  void runOnOperation() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<AffineDialect, memref::MemRefDialect, tensor::TensorDialect, 
                           scf::SCFDialect, decisionforest::DecisionForestDialect,
                           math::MathDialect, arith::ArithDialect, func::FuncDialect>();

    target.addIllegalOp<decisionforest::EnsembleConstantOp,
                        decisionforest::GetTreeFromEnsembleOp,
                        decisionforest::GetTreeClassIdOp,
                        decisionforest::GetRootOp,
                        decisionforest::IsLeafOp,
                        decisionforest::TraverseTreeTileOp,
                        decisionforest::GetLeafValueOp,
                        decisionforest::NodeToIndexOp,
                        decisionforest::IndexToNodeOp>();
    
    AddIndexArgumentsForAllBlocks();
    
    RewritePatternSet patterns(&getContext());
    patterns.add<NodeToIndexOpLowering,
                 IndexToNodeOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
  }
};
} // namespace decisionforest
} // namespace mlir

namespace mlir
{
namespace decisionforest
{
void ConvertNodeTypeToIndexType(mlir::MLIRContext& context, mlir::ModuleOp module) {
  // llvm::DebugFlag = true;
  // Lower from high-level IR to mid-level IR
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(std::make_unique<ConvertNodeTypeToIndexTypePass>());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Conversion from NodeType to Index failed.\n";
  }
}

} // decisionforest
} // mlir