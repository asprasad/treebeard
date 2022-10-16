#include "Dialect.h"
// #include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <unordered_map>

using namespace mlir;

namespace
{

struct WalkDecisionTreeOpLowering: public ConversionPattern {
  WalkDecisionTreeOpLowering(MLIRContext *ctx) : ConversionPattern(mlir::decisionforest::WalkDecisionTreeOp::getOperationName(), 1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    mlir::decisionforest::WalkDecisionTreeOp walkTreeOp = llvm::dyn_cast<mlir::decisionforest::WalkDecisionTreeOp>(op);
    assert(walkTreeOp);
    assert(operands.size() == 2);
    if (!walkTreeOp)
        return mlir::failure();

    auto tree = operands[0];
    auto inputRow = operands[1];
    
    auto location = op->getLoc();
    auto context = inputRow.getContext();
    auto treeType = tree.getType().cast<mlir::decisionforest::TreeType>();

    auto nodeType = mlir::decisionforest::NodeType::get(context);
    auto node = rewriter.create<decisionforest::GetRootOp>(location, nodeType, tree);

    scf::WhileOp whileLoop = rewriter.create<scf::WhileOp>(location, nodeType, static_cast<Value>(node));
    Block *before = rewriter.createBlock(&whileLoop.getBefore(), {}, nodeType, location);
    Block *after = rewriter.createBlock(&whileLoop.getAfter(), {}, nodeType, location);

    // Create the 'do' part for the condition.
    {
        rewriter.setInsertionPointToStart(&whileLoop.getBefore().front());
        auto node = before->getArguments()[0];
        auto isLeaf = rewriter.create<decisionforest::IsLeafOp>(location, rewriter.getI1Type(), tree, node);
        auto falseConstant = rewriter.create<arith::ConstantIntOp>(location, int64_t(0), rewriter.getI1Type());
        auto equalTo = rewriter.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, static_cast<Value>(isLeaf), static_cast<Value>(falseConstant));
        rewriter.create<scf::ConditionOp>(location, equalTo, ValueRange({node})); // this is the terminator
    }
    // Create the loop body
    {
        rewriter.setInsertionPointToStart(&whileLoop.getAfter().front());
        auto node = after->getArguments()[0];
        
        auto traverseTile = rewriter.create<decisionforest::TraverseTreeTileOp>(
          location,
          nodeType,
          tree,
          node,
          inputRow);

        rewriter.create<scf::YieldOp>(location, static_cast<Value>(traverseTile));
    }
    rewriter.setInsertionPointAfter(whileLoop);
    auto treePrediction = rewriter.create<decisionforest::GetLeafValueOp>(location, treeType.getThresholdType(), tree, whileLoop.getResults()[0]);
    rewriter.replaceOp(op, static_cast<Value>(treePrediction));

    return mlir::success();
  }
};

struct PipelinedWalkDecisionTreeOpLowering: public ConversionPattern {
  PipelinedWalkDecisionTreeOpLowering(MLIRContext *ctx) : ConversionPattern(mlir::decisionforest::PipelinedWalkDecisionTreeOp::getOperationName(), 1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    mlir::decisionforest::PipelinedWalkDecisionTreeOp walkTreeOp = llvm::dyn_cast<mlir::decisionforest::PipelinedWalkDecisionTreeOp>(op);
    
    assert(walkTreeOp);
    if (!walkTreeOp)
        return mlir::failure();

    auto trees = walkTreeOp.trees();
    auto dataRows = walkTreeOp.dataRows();
    auto unrollLoopAttr = walkTreeOp.UnrollLoopAttr();
    assert(trees.size() == dataRows.size());

    auto location = op->getLoc();
    auto context = op->getContext();

    std::vector<Value> nodes;
    std::vector<Type> nodeTypes;
    std::unordered_map<void*, decisionforest::GetRootOp> treeRootMap;
    auto nodeType = mlir::decisionforest::NodeType::get(context);
    for (size_t i = 0; i < trees.size(); i++) {
      if (treeRootMap.find(trees[i].getAsOpaquePointer()) == treeRootMap.end())
        treeRootMap[trees[i].getAsOpaquePointer()] = rewriter.create<decisionforest::GetRootOp>(location, nodeType, trees[i]);
      
      nodes.push_back(treeRootMap[trees[i].getAsOpaquePointer()]);
      nodeTypes.push_back(nodeType);
    }

    std::vector<Value> predictions;

    // Unroll the loop.
    if (unrollLoopAttr.GetUnrollFactor() > 1) {
      int32_t unrollFactor = unrollLoopAttr.GetUnrollFactor();
      ValueRange nodeArgs = nodes;

      for (int32_t i = 0; i < unrollFactor - 1; i++) {
        auto traverseTile = rewriter.create<decisionforest::InterleavedTraverseTreeTileOp>(
          location,
          nodeTypes,
          trees,
          nodeArgs,
          dataRows);
        
        nodeArgs = ValueRange(traverseTile.getResults());
      }

      for (size_t i = 0; i < trees.size(); i++) {
        auto treeType = trees[i].getType().cast<mlir::decisionforest::TreeType>();
        auto treePrediction = rewriter.create<decisionforest::GetLeafValueOp>(location, treeType.getThresholdType(), trees[i], nodeArgs[i]);
        predictions.push_back(treePrediction);
      }
    }
    else { 
      scf::WhileOp whileLoop = rewriter.create<scf::WhileOp>(location, nodeTypes, nodes);
      std::vector<mlir::Location> locations(nodeTypes.size(), location);
      Block *before = rewriter.createBlock(&whileLoop.getBefore(), {}, nodeTypes, locations);
      Block *after = rewriter.createBlock(&whileLoop.getAfter(), {}, nodeTypes, locations);

      // Create the 'do' part for the condition.
      {
          rewriter.setInsertionPointToStart(&whileLoop.getBefore().front());
          auto nodeArgs = before->getArguments();

          assert(trees.size() == nodeArgs.size() && "Number of arguments should be the same as number of trees.");

          std::vector<Value> isLeafResults;
          for (size_t i = 0; i < trees.size(); i++) {
            auto isLeaf = rewriter.create<decisionforest::IsLeafOp>(location, rewriter.getI1Type(), trees[i], nodeArgs[i]);
            isLeafResults.push_back(isLeaf);
          }
          
          auto falseConstant = rewriter.create<arith::ConstantIntOp>(location, int64_t(0), rewriter.getI1Type());
          auto trueConstant = rewriter.create<arith::ConstantIntOp>(location, int64_t(1), rewriter.getI1Type());

          // TODO - This will not work when trees have different depths. We're yet to handle that case here.
          // TODO - This does a bitwise and. That works in this case but I would preffered to use a logical AND to make semantics clearer. Yet to find on op for that.
          Value allLeaf = trueConstant;
          for (size_t i = 0; i < trees.size(); i++) {
            allLeaf = rewriter.create<arith::AndIOp>(location, isLeafResults[i], allLeaf);
          }

          auto equalTo = rewriter.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, static_cast<Value>(allLeaf), static_cast<Value>(falseConstant));
          rewriter.create<scf::ConditionOp>(location, equalTo, nodeArgs); // this is the terminator
      }
      // Create the loop body
      {
          rewriter.setInsertionPointToStart(&whileLoop.getAfter().front());
          auto nodeArgs = after->getArguments();
          
          auto traverseTile = rewriter.create<decisionforest::InterleavedTraverseTreeTileOp>(
            location,
            nodeTypes,
            trees,
            nodeArgs,
            dataRows);

          rewriter.create<scf::YieldOp>(location, ValueRange(traverseTile.getResults()));
      }
      rewriter.setInsertionPointAfter(whileLoop);

      for (size_t i = 0; i < trees.size(); i++) {
        auto treeType = trees[i].getType().cast<mlir::decisionforest::TreeType>();
        auto treePrediction = rewriter.create<decisionforest::GetLeafValueOp>(location, treeType.getThresholdType(), trees[i], whileLoop.getResults()[i]);
        predictions.push_back(treePrediction);
      }
    }

    rewriter.replaceOp(op, predictions);
    return mlir::success();
  }
};

struct WalkDecisionTreePeeledOpLowering: public ConversionPattern {
  WalkDecisionTreePeeledOpLowering(MLIRContext *ctx) : ConversionPattern(mlir::decisionforest::WalkDecisionTreePeeledOp::getOperationName(), 1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    mlir::decisionforest::WalkDecisionTreePeeledOp walkTreeOp = llvm::dyn_cast<mlir::decisionforest::WalkDecisionTreePeeledOp>(op);
    assert(walkTreeOp);
    assert(operands.size() == 2);
    if (!walkTreeOp)
        return mlir::failure();

    auto tree = operands[0];
    auto inputRow = operands[1];
    auto iterationsToPeelAttr = walkTreeOp.iterationsToPeelAttr();
    auto iterationsToPeel = iterationsToPeelAttr.getValue().getSExtValue();

    auto location = op->getLoc();
    auto context = inputRow.getContext();
    auto treeType = tree.getType().cast<mlir::decisionforest::TreeType>();

    auto nodeType = mlir::decisionforest::NodeType::get(context);
    Value node = rewriter.create<decisionforest::GetRootOp>(location, nodeType, tree);
    assert (iterationsToPeel > 1);
    Value walkResult;
    for (int64_t iteration=0 ; iteration<iterationsToPeel-1 ; ++iteration) {
      node = rewriter.create<decisionforest::TraverseTreeTileOp>(
        location,
        nodeType,
        tree,
        node,
        inputRow);
      // TODO this needs to change to a different op that always checks if a tile is a leaf
      auto isLeaf = rewriter.create<decisionforest::IsLeafTileOp>(location, rewriter.getI1Type(), tree, node);
      auto ifElse = rewriter.create<scf::IfOp>(location, walkTreeOp.getResult().getType(), isLeaf, true);

      // generate the if case
      auto thenBuilder = ifElse.getThenBodyBuilder();
      auto getLeafValue = thenBuilder.create<decisionforest::GetLeafTileValueOp>(location, treeType.getThresholdType(), tree, node);
      thenBuilder.create<scf::YieldOp>(location, static_cast<Value>(getLeafValue));
      if (iteration==0) {
        walkResult = ifElse.getResult(0);
      }
      else {
        rewriter.create<scf::YieldOp>(location, ifElse.getResult(0));
      }
      rewriter.setInsertionPointToStart(ifElse.elseBlock());
    }
    scf::WhileOp whileLoop = rewriter.create<scf::WhileOp>(location, nodeType, static_cast<Value>(node));
    Block *before = rewriter.createBlock(&whileLoop.getBefore(), {}, nodeType, location);
    Block *after = rewriter.createBlock(&whileLoop.getAfter(), {}, nodeType, location);

    // Create the 'do' part for the condition.
    {
        rewriter.setInsertionPointToStart(&whileLoop.getBefore().front());
        auto node = before->getArguments()[0];
        auto isLeaf = rewriter.create<decisionforest::IsLeafOp>(location, rewriter.getI1Type(), tree, node);
        auto falseConstant = rewriter.create<arith::ConstantIntOp>(location, int64_t(0), rewriter.getI1Type());
        auto equalTo = rewriter.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, static_cast<Value>(isLeaf), static_cast<Value>(falseConstant));
        rewriter.create<scf::ConditionOp>(location, equalTo, ValueRange({node})); // this is the terminator
    }
    // Create the loop body
    {
        rewriter.setInsertionPointToStart(&whileLoop.getAfter().front());
        auto node = after->getArguments()[0];
        
        auto traverseTile = rewriter.create<decisionforest::TraverseTreeTileOp>(
          location,
          nodeType,
          tree,
          node,
          inputRow);

        rewriter.create<scf::YieldOp>(location, static_cast<Value>(traverseTile));
    }
    rewriter.setInsertionPointAfter(whileLoop);
    auto treePrediction = rewriter.create<decisionforest::GetLeafValueOp>(location, treeType.getThresholdType(), tree, whileLoop.getResults()[0]);
    rewriter.create<scf::YieldOp>(location, static_cast<Value>(treePrediction));
    rewriter.replaceOp(op, walkResult);

    return mlir::success();
  }
};

struct WalkDecisionTreeOpLoweringPass: public PassWrapper<WalkDecisionTreeOpLoweringPass, OperationPass<mlir::func::FuncOp>> {
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<memref::MemRefDialect, scf::SCFDialect, 
                           decisionforest::DecisionForestDialect, math::MathDialect, arith::ArithmeticDialect>();
    target.addIllegalOp<decisionforest::WalkDecisionTreeOp, decisionforest::WalkDecisionTreePeeledOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<WalkDecisionTreeOpLowering>(&getContext());
    patterns.add<WalkDecisionTreePeeledOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
  }
};

struct PipelinedWalkDecisionTreeOpLoweringPass: public PassWrapper<PipelinedWalkDecisionTreeOpLoweringPass, OperationPass<mlir::func::FuncOp>> {
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<memref::MemRefDialect, scf::SCFDialect, 
                           decisionforest::DecisionForestDialect, math::MathDialect, arith::ArithmeticDialect>();
    target.addIllegalOp<decisionforest::PipelinedWalkDecisionTreeOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<PipelinedWalkDecisionTreeOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
  }
};

}

namespace mlir
{
namespace decisionforest
{

void AddWalkDecisionTreeOpLoweringPass(mlir::OpPassManager &optPM) {
  optPM.addPass(std::make_unique<WalkDecisionTreeOpLoweringPass>());
  optPM.addPass(std::make_unique<PipelinedWalkDecisionTreeOpLoweringPass>());
}

}
}