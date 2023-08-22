#include "Dialect.h"
#include "OpLoweringUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Target/TargetMachine.h"

#include "GPUSupportUtils.h"
#include "LowerReduceOps.h"

using namespace mlir;

namespace mlir {
namespace decisionforest {

// Defined in GPURepresentations.cpp
int64_t GetConstantIntValueFromMLIRValue(Value val);

Value createFloatConst(Location location, ConversionPatternRewriter &rewriter,
                       Type type, double floatVal) {
  auto floatType = type.cast<mlir::FloatType>();
  if (type.isa<mlir::Float32Type>()) {
    return rewriter.create<arith::ConstantFloatOp>(
        location, llvm::APFloat((float)floatVal), floatType);
  } else if (type.isa<mlir::Float64Type>()) {
    return rewriter.create<arith::ConstantFloatOp>(
        location, llvm::APFloat(floatVal), floatType);
  } else {
    llvm_unreachable("Unsupported type");
  }
  return Value();
}

} // namespace decisionforest
} // namespace mlir

namespace {

struct ReduceOpLowering : public ConversionPattern {
  ReduceOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::decisionforest::ReduceOp::getOperationName(),
                          1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto reduceOp = AssertOpIsOfType<decisionforest::ReduceOp>(op);
    auto location = reduceOp.getLoc();
    decisionforest::ReduceOpAdaptor adaptor(operands);

    auto loadVal = rewriter.create<memref::LoadOp>(
        location, adaptor.getTargetMemref(), adaptor.getIndices());
    auto addVal = rewriter.create<arith::AddFOp>(location, loadVal.getResult(),
                                                 adaptor.getValue());
    rewriter.create<memref::StoreOp>(location, addVal.getResult(),
                                     adaptor.getTargetMemref(),
                                     adaptor.getIndices());
    // rewriter.create<decisionforest::PrintfOp>(
    //     location, "Index: (%ld, %ld, %ld) -> %lf\n",
    //     ValueRange{adaptor.getIndices()[0], adaptor.getIndices()[1],
    //                adaptor.getIndices()[2], addVal.getResult()});

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ReduceDimensionOpLowering : public ConversionPattern {
  void generateSimpleReductionLoopNest(
      Location location, ConversionPatternRewriter &rewriter,
      Value sourceMemref, Value targetMemref,
      mlir::Operation::operand_range reducedDims,
      std::vector<Value> &rangeStart, std::vector<Value> &rangeEnd,
      int64_t reductionDim) const {

    // Iterate [rangeStart, rangeEnd)
    auto oneIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 1);
    Operation *outermostLoop = nullptr;
    std::list<Value> loopIVs;
    for (auto startEndPair : llvm::zip(rangeStart, rangeEnd)) {
      auto start = std::get<0>(startEndPair);
      auto end = std::get<1>(startEndPair);
      auto loop = rewriter.create<scf::ForOp>(location, start, end,
                                              oneIndexConst.getResult());
      if (outermostLoop == nullptr) {
        outermostLoop = loop.getOperation();
      }
      loopIVs.push_back(loop.getInductionVar());
      rewriter.setInsertionPointToStart(loop.getBody());
    }
    // for each value, sum over the reduction dimension
    auto zeroIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
    auto reductionDimSize =
        sourceMemref.getType().cast<MemRefType>().getDimSize(reductionDim);
    auto reductionDimSizeConst =
        rewriter.create<arith::ConstantIndexOp>(location, reductionDimSize);
    auto memrefElemType =
        sourceMemref.getType().cast<MemRefType>().getElementType();
    auto zeroFloatConst = decisionforest::createFloatConst(location, rewriter,
                                                           memrefElemType, 0.0);
    auto reductionLoop = rewriter.create<scf::ForOp>(
        location, zeroIndexConst.getResult(), reductionDimSizeConst.getResult(),
        oneIndexConst.getResult(), ValueRange{zeroFloatConst});
    rewriter.setInsertionPointToStart(reductionLoop.getBody());

    auto reductionLoopIV = reductionLoop.getInductionVar();
    std::vector<Value> memrefIndex;
    memrefIndex.push_back(reductionLoopIV);
    memrefIndex.insert(memrefIndex.end(), reducedDims.size(), zeroIndexConst);
    memrefIndex.insert(memrefIndex.end(), loopIVs.begin(), loopIVs.end());

    auto loadVal =
        rewriter.create<memref::LoadOp>(location, sourceMemref, memrefIndex);

    // rewriter.create<decisionforest::PrintfOp>(
    //     location, "Index: %ld, %ld, %ld, Value: %lf\n",
    //     ValueRange{memrefIndex[0], memrefIndex[1], memrefIndex[2],
    //                loadVal.getResult()});
    auto accumulator = reductionLoop.getRegionIterArgs().front();
    auto addVal = rewriter.create<arith::AddFOp>(location, loadVal.getResult(),
                                                 accumulator);
    rewriter.create<scf::YieldOp>(location, addVal.getResult());

    rewriter.setInsertionPointAfter(reductionLoop);

    // Write value to target memref
    std::vector<Value> resultIndex(loopIVs.begin(), loopIVs.end());

    rewriter.create<memref::StoreOp>(location,
                                     reductionLoop.getResults().front(),
                                     targetMemref, resultIndex);

    rewriter.setInsertionPointAfter(outermostLoop);
  }

  ReduceDimensionOpLowering(MLIRContext *ctx)
      : ConversionPattern(
            mlir::decisionforest::ReduceDimensionOp::getOperationName(),
            1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto reduceOp = AssertOpIsOfType<decisionforest::ReduceDimensionOp>(op);
    auto location = reduceOp.getLoc();

    // The values for the outer indices. These are fixed
    auto reducedDims = reduceOp.getReducedDimensions();

    auto reductionDim = reduceOp.getReductionDimension();
    auto reductionDimVal =
        decisionforest::GetConstantIntValueFromMLIRValue(reductionDim);
    auto rangeStart = reduceOp.getRangeStart();
    auto rangeEnd = reduceOp.getRangeEnd();

    auto sourceMemref = reduceOp.getSourceMemref();
    // auto sourceMemrefType = sourceMemref.getType().cast<MemRefType>();
    auto targetMemref = reduceOp.getTargetMemref();

    assert(rangeStart.size() != 0);
    assert(rangeStart.size() == rangeEnd.size());
    std::vector<Value> rangeStartVec(rangeStart.begin(), rangeStart.end());
    std::vector<Value> rangeEndVec(rangeEnd.begin(), rangeEnd.end());
    // If no range is specified, then we need to walk the full range
    // of the dimensions with dimension num > reductionDim
    // if (rangeStart.size() == 0) {
    //   auto zeroIndexConst =
    //       rewriter.create<arith::ConstantIndexOp>(location, 0);
    //   for (auto i = reductionDimVal + reducedDims.size() + 1;
    //        i < sourceMemrefType.getRank(); ++i) {
    //     rangeStartVec.push_back(zeroIndexConst);
    //     auto dimSize = sourceMemrefType.getDimSize(i);
    //     auto dimSizeConst =
    //         rewriter.create<arith::ConstantIndexOp>(location, dimSize);
    //     rangeEndVec.push_back(dimSizeConst);
    //   }
    // }
    generateSimpleReductionLoopNest(location, rewriter, sourceMemref,
                                    targetMemref, reducedDims, rangeStartVec,
                                    rangeEndVec, reductionDimVal);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ReduceInplaceOpLowering : public ConversionPattern {
  void generateSimpleReductionLoopNest(Location location,
                                       ConversionPatternRewriter &rewriter,
                                       Value targetMemref,
                                       mlir::Operation::operand_range indices,
                                       std::vector<Value> &rangeStart,
                                       std::vector<Value> &rangeEnd,
                                       int64_t reductionDim) const {
    // Iterate [rangeStart, rangeEnd)
    auto oneIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 1);
    Operation *outermostLoop = nullptr;
    std::list<Value> loopIVs;
    for (auto startEndPair : llvm::zip(rangeStart, rangeEnd)) {
      auto start = std::get<0>(startEndPair);
      auto end = std::get<1>(startEndPair);
      auto loop = rewriter.create<scf::ForOp>(location, start, end,
                                              oneIndexConst.getResult());
      if (outermostLoop == nullptr) {
        outermostLoop = loop.getOperation();
      }
      loopIVs.push_back(loop.getInductionVar());
      rewriter.setInsertionPointToStart(loop.getBody());
    }
    // for each value, sum over the reduction dimension
    auto zeroIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
    auto reductionDimSize =
        targetMemref.getType().cast<MemRefType>().getDimSize(reductionDim);
    auto reductionDimSizeConst =
        rewriter.create<arith::ConstantIndexOp>(location, reductionDimSize);
    auto memrefElemType =
        targetMemref.getType().cast<MemRefType>().getElementType();
    auto zeroFloatConst = decisionforest::createFloatConst(location, rewriter,
                                                           memrefElemType, 0.0);
    auto reductionLoop = rewriter.create<scf::ForOp>(
        location, zeroIndexConst.getResult(), reductionDimSizeConst.getResult(),
        oneIndexConst.getResult(), ValueRange{zeroFloatConst});
    rewriter.setInsertionPointToStart(reductionLoop.getBody());

    auto reductionLoopIV = reductionLoop.getInductionVar();
    std::vector<Value> memrefIndex(indices.begin(), indices.end());
    memrefIndex.push_back(reductionLoopIV);
    memrefIndex.insert(memrefIndex.end(), loopIVs.begin(), loopIVs.end());

    auto loadVal =
        rewriter.create<memref::LoadOp>(location, targetMemref, memrefIndex);
    auto accumulator = reductionLoop.getRegionIterArgs().front();
    auto addVal = rewriter.create<arith::AddFOp>(location, loadVal.getResult(),
                                                 accumulator);
    rewriter.create<scf::YieldOp>(location, addVal.getResult());

    rewriter.setInsertionPointAfter(reductionLoop);

    // Write value to target memref
    std::vector<Value> resultIndex(indices.begin(), indices.end());
    resultIndex.push_back(zeroIndexConst);
    resultIndex.insert(resultIndex.end(), loopIVs.begin(), loopIVs.end());

    rewriter.create<memref::StoreOp>(location,
                                     reductionLoop.getResults().front(),
                                     targetMemref, resultIndex);

    rewriter.setInsertionPointAfter(outermostLoop);
  }

  ReduceInplaceOpLowering(MLIRContext *ctx)
      : ConversionPattern(
            mlir::decisionforest::ReduceDimensionInplaceOp::getOperationName(),
            1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto reduceOp =
        AssertOpIsOfType<decisionforest::ReduceDimensionInplaceOp>(op);
    auto location = reduceOp.getLoc();

    // The values for the outer indices. These are fixed
    auto indices = reduceOp.getIndices();
    auto reductionDim = reduceOp.getDimension();
    auto reductionDimVal =
        decisionforest::GetConstantIntValueFromMLIRValue(reductionDim);
    auto rangeStart = reduceOp.getRangeStart();
    auto rangeEnd = reduceOp.getRangeEnd();
    auto targetMemref = reduceOp.getTargetMemref();
    // auto targetMemrefType = targetMemref.getType().cast<MemRefType>();

    assert(rangeStart.size() != 0);
    assert(rangeStart.size() == rangeEnd.size());
    std::vector<Value> rangeStartVec(rangeStart.begin(), rangeStart.end());
    std::vector<Value> rangeEndVec(rangeEnd.begin(), rangeEnd.end());
    // If no range is specified, then we need to walk the full range
    // of the dimensions with dimension num > reductionDim
    // if (rangeStart.size() == 0) {
    //   auto zeroIndexConst =
    //       rewriter.create<arith::ConstantIndexOp>(location, 0);
    //   for (auto i = reductionDimVal + 1; i < targetMemrefType.getRank(); ++i)
    //   {
    //     rangeStartVec.push_back(zeroIndexConst);
    //     auto dimSize = targetMemrefType.getDimSize(i);
    //     auto dimSizeConst =
    //         rewriter.create<arith::ConstantIndexOp>(location, dimSize);
    //     rangeEndVec.push_back(dimSizeConst);
    //   }
    // }
    generateSimpleReductionLoopNest(location, rewriter, targetMemref, indices,
                                    rangeStartVec, rangeEndVec,
                                    reductionDimVal);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct LowerReductionOps
    : public PassWrapper<LowerReductionOps, OperationPass<mlir::func::FuncOp>> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, arith::ArithDialect>();
  }

  void runOnOperation() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<memref::MemRefDialect, scf::SCFDialect,
                           decisionforest::DecisionForestDialect,
                           math::MathDialect, arith::ArithDialect,
                           func::FuncDialect, gpu::GPUDialect>();

    target.addIllegalOp<decisionforest::ReduceOp,
                        decisionforest::ReduceDimensionInplaceOp,
                        decisionforest::ReduceDimensionOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ReduceOpLowering, ReduceInplaceOpLowering,
                 ReduceDimensionOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace decisionforest {

void lowerReduceToMemref(mlir::MLIRContext &context, mlir::ModuleOp module) {
  // llvm::DebugFlag = true;
  // Lower from high-level IR to mid-level IR
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(std::make_unique<LowerReductionOps>());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering reduction ops failed.\n";
  }
  // llvm::DebugFlag = false;
}

void legalizeReductionsAndCanonicalize(mlir::MLIRContext &context,
                                       mlir::ModuleOp module) {
  decisionforest::legalizeReductions(context, module);
  decisionforest::RunCanonicalizerPass(context, module);
}

void lowerReductionsAndCanonicalize(mlir::MLIRContext &context,
                                    mlir::ModuleOp module) {
  decisionforest::lowerLinalgToLoops(context, module);
  decisionforest::lowerReduceToMemref(context, module);
  decisionforest::RunCanonicalizerPass(context, module);
}

} // namespace decisionforest
} // namespace mlir