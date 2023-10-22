#include "Dialect.h"
#include <iostream>
#include <mutex>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/Linalg/Passes.h"

#include "llvm/Support/Debug.h"

#include "Dialect.h"
#include "MemrefTypes.h"
#include "TreeTilingUtils.h"

#include "GPUSupportUtils.h"
#include "LIRLoweringHelpers.h"
#include "Logger.h"
#include "OpLoweringUtils.h"

using namespace mlir;

namespace mlir {
namespace decisionforest {

struct ReduceToCooperativeReduceOp : public ConversionPattern {
  ReduceToCooperativeReduceOp(MLIRContext *ctx)
      : ConversionPattern(
            mlir::decisionforest::ReduceDimensionOp::getOperationName(),
            1 /*benefit*/, ctx) {}

  std::vector<scf::ParallelOp>
  getNestedBlockLoops(scf::ParallelOp reductionLoop) const {
    std::vector<scf::ParallelOp> blockLoops;
    // iterate over the ops in reductionLoop and push any block loops into
    // blockLoops
    reductionLoop->walk([&](scf::ParallelOp parFor) {
      if (parFor != reductionLoop && decisionforest::isThreadLoop(parFor)) {
        blockLoops.push_back(parFor);
      }
    });
    return blockLoops;
  }

  void orderBlockLoops(std::vector<scf::ParallelOp> &blockLoops) const {
    // Sort the block loops in the order of their mapping attribute (x, y, z)
    std::sort(blockLoops.begin(), blockLoops.end(),
              [](scf::ParallelOp a, scf::ParallelOp b) {
                auto firstMappingAttr = getMappingAttr(a);
                auto secondMappingAttr = getMappingAttr(b);
                assert(isThreadLoop(a) && isThreadLoop(b) &&
                       "Expected thread loops");
                assert(firstMappingAttr && secondMappingAttr &&
                       "Expected mapping attrs");
                return static_cast<int>(firstMappingAttr.getProcessor()) <
                       static_cast<int>(secondMappingAttr.getProcessor());
              });
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto reduceOp =
        AssertOpIsOfType<mlir::decisionforest::ReduceDimensionOp>(op);
    if (!reduceOp)
      return mlir::failure();
    // Find the parallel loop for which this op is doing a partial reduction
    // i.e. the parallel loop immediately before this op
    // auto &operationList = op->getBlock()->getOperations();
    // get a block iterator for this op
    auto opIt = op->getReverseIterator();
    // std::find(operationList.begin(), operationList.end(), op);
    scf::ParallelOp reductionTreeLoop;
    do {
      auto &currOp = *opIt;
      if ((reductionTreeLoop = dyn_cast<scf::ParallelOp>(&currOp)))
        break;
      opIt++;
    } while (opIt != op->getBlock()->rend());
    assert(reductionTreeLoop && "No parallel loop found for reduction op");
    // Find the parent parallel loops for this op which correspond to block dims
    std::vector<scf::ParallelOp> blockLoops =
        getNestedBlockLoops(reductionTreeLoop);
    blockLoops.push_back(reductionTreeLoop);
    auto parentOp = reductionTreeLoop->getParentOfType<scf::ParallelOp>();
    while (parentOp && decisionforest::isThreadLoop(parentOp)) {
      blockLoops.insert(blockLoops.begin(), parentOp);
      parentOp = parentOp->getParentOfType<scf::ParallelOp>();
    }

    orderBlockLoops(blockLoops);

    // Say this loops bounds are s_t and e_t
    // Find the parent parallel loops for this op which correspond to block dims
    // Say this loops indices and steps are i0, i1... and s0, s1 ...
    // Then the set of threads to be used for this reduction is (i0, i1, ...,
    // s_t, *) -> (i0 + s0, i1 + s1, ..., e_t, *)
    auto location = op->getLoc();
    std::vector<Value> startIndices, endIndices;
    for (auto blockLoop : blockLoops) {
      if (blockLoop == reductionTreeLoop) {
        // Add the start and end indices for the reduction tree loop
        startIndices.push_back(reductionTreeLoop.getLowerBound()[0]);
        endIndices.push_back(reductionTreeLoop.getUpperBound()[0]);
      } else {
        startIndices.push_back(blockLoop.getInductionVars()[0]);
        auto endIndex = rewriter.create<arith::AddIOp>(
            location, blockLoop.getInductionVars()[0], blockLoop.getStep()[0]);
        endIndices.push_back(endIndex);
      }
    }
    // Add 0, 1 pairs as required
    while (startIndices.size() < 3) {
      // Push consts 0 and 1 into start and end respectively
      startIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(location, 0));
      endIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(location, 1));
    }

    assert(startIndices.size() == endIndices.size());

    if (reduceOp.getPreReductionDimensionStart().size() > 0 ||
        reduceOp.getPreReductionDimensionEnd().size() > 0) {
      llvm::errs() << "Pre reduction dimensions not supported yet\n";
      return mlir::failure();
    }

    if (reduceOp.getPostReductionDimensionStart().size() != 1 ||
        reduceOp.getPostReductionDimensionEnd().size() != 1) {
      llvm::errs() << "Number of post-reduction dimensions should be 1\n";
      return mlir::failure();
    }

    Value rangeStart = reduceOp.getPostReductionDimensionStart().front();
    Value rangeEnd = reduceOp.getPostReductionDimensionEnd().front();
    // Create the cooperative reduce op
    rewriter.create<decisionforest::CooperativeReduceDimensionOp>(
        location, reduceOp.getTargetMemref(), reduceOp.getSourceMemref(),
        reduceOp.getReductionDimension(), ValueRange{}, rangeStart, rangeEnd,
        startIndices.at(0), startIndices.at(1), startIndices.at(2),
        endIndices.at(0), endIndices.at(1), endIndices.at(2),
        reduceOp.getInitialValueAttr());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ConvertReductionsToCooperativeReductions
    : public PassWrapper<ConvertReductionsToCooperativeReductions,
                         OperationPass<mlir::func::FuncOp>> {
  ConvertReductionsToCooperativeReductions() {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<AffineDialect, memref::MemRefDialect, scf::SCFDialect,
                decisionforest::DecisionForestDialect, arith::ArithDialect>();
  }

  void runOnOperation() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<
        AffineDialect, memref::MemRefDialect, scf::SCFDialect,
        decisionforest::DecisionForestDialect, vector::VectorDialect,
        math::MathDialect, arith::ArithDialect, func::FuncDialect,
        gpu::GPUDialect>();

    target.addDynamicallyLegalOp<decisionforest::ReduceDimensionOp>(
        [](decisionforest::ReduceDimensionOp op) {
          return op.getReductionTypeAttr().getReductionType() ==
                 decisionforest::Reduction::kArgMax;
        });
    target.addIllegalOp<decisionforest::ReduceDimensionInplaceOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ReduceToCooperativeReduceOp>(patterns.getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

void runConvertToCooperativeReducePass(mlir::MLIRContext &context,
                                       mlir::ModuleOp module) {
  // llvm::DebugFlag = true;
  mlir::PassManager pm(&context);
  auto &nestedPM = pm.nest<func::FuncOp>();
  nestedPM.addPass(
      std::make_unique<ConvertReductionsToCooperativeReductions>());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Convert to cooperative reduce pass failed.\n";
  }
  // llvm::DebugFlag = false;
}

void convertToCooperativeReduceAndCanonicalize(mlir::MLIRContext &context,
                                               mlir::ModuleOp module) {
  runConvertToCooperativeReducePass(context, module);
  RunCanonicalizerPass(context, module);
}

} // namespace decisionforest
} // namespace mlir