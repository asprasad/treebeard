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

mlir::gpu::KernelDim3 GetThreadID(mlir::Operation *op);
mlir::gpu::KernelDim3 GetBlockID(mlir::Operation *op);

Value createFloatConst(Location location, OpBuilder &rewriter, Type type,
                       double floatVal) {
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
      int64_t reductionDim, double initialValue) const {

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
    auto initialValueConst = decisionforest::createFloatConst(
        location, rewriter, memrefElemType, initialValue);
    auto reductionLoop = rewriter.create<scf::ForOp>(
        location, zeroIndexConst.getResult(), reductionDimSizeConst.getResult(),
        oneIndexConst.getResult(), ValueRange{initialValueConst});
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
    generateSimpleReductionLoopNest(
        location, rewriter, sourceMemref, targetMemref, reducedDims,
        rangeStartVec, rangeEndVec, reductionDimVal,
        reduceOp.getInitialValue().convertToDouble());
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
    auto initialValConst = decisionforest::createFloatConst(
        location, rewriter, memrefElemType, 0.0);
    auto reductionLoop = rewriter.create<scf::ForOp>(
        location, zeroIndexConst.getResult(), reductionDimSizeConst.getResult(),
        oneIndexConst.getResult(), ValueRange{initialValConst});
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

struct CooperativeReduceDimensionOpLowering : public ConversionPattern {

  CooperativeReduceDimensionOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::decisionforest::CooperativeReduceDimensionOp::
                              getOperationName(),
                          1 /*benefit*/, ctx) {}

  std::tuple<Value, Value> generateLocalThreadId(
      Location location, ConversionPatternRewriter &rewriter,
      decisionforest::CooperativeReduceDimensionOp reduceOp) const {

    auto threadId = decisionforest::GetThreadID(reduceOp);
    auto localThreadIdX = rewriter.create<arith::SubIOp>(
        location, threadId.x, reduceOp.getBlockXStart());
    auto localThreadIdY = rewriter.create<arith::SubIOp>(
        location, threadId.y, reduceOp.getBlockYStart());
    // auto localThreadIdZ = rewriter.create<arith::SubIOp>(
    //     location, threadId.z, reduceOp.getBlockZStart());

    auto startZ = decisionforest::GetConstantIntValueFromMLIRValue(
        reduceOp.getBlockZStart());
    auto endZ = decisionforest::GetConstantIntValueFromMLIRValue(
        reduceOp.getBlockZEnd());
    assert(startZ == 0 && endZ == 1 &&
           "Only 2D thread blocks supported for now");

    auto numThreadsX = rewriter.create<arith::SubIOp>(
        location, reduceOp.getBlockXEnd(), reduceOp.getBlockXStart());
    // index = numThreadsX*threadNum.Y + threadNum.X
    auto nxTimesTy =
        rewriter.create<arith::MulIOp>(location, numThreadsX, localThreadIdY);
    auto localThreadId = rewriter.create<arith::AddIOp>(
        location, static_cast<Value>(nxTimesTy), localThreadIdX);

    auto numThreadsY = rewriter.create<arith::SubIOp>(
        location, reduceOp.getBlockYEnd(), reduceOp.getBlockYStart());
    auto numThreads = rewriter.create<arith::MulIOp>(
        location, numThreadsX, static_cast<Value>(numThreadsY));

    return std::make_tuple(localThreadId, numThreads);
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto reduceOp =
        AssertOpIsOfType<decisionforest::CooperativeReduceDimensionOp>(op);
    auto location = reduceOp.getLoc();

    // The values for the outer indices. These are fixed
    auto reducedDims = reduceOp.getReducedDimensions();
    assert(reducedDims.empty());

    auto reductionDim = reduceOp.getReductionDimension();
    auto reductionDimVal =
        decisionforest::GetConstantIntValueFromMLIRValue(reductionDim);
    assert(reductionDimVal == 0);

    auto rangeStart = reduceOp.getRangeStart();
    auto rangeEnd = reduceOp.getRangeEnd();

    auto sourceMemref = reduceOp.getSourceMemref();
    // auto sourceMemrefType = sourceMemref.getType().cast<MemRefType>();
    auto targetMemref = reduceOp.getTargetMemref();

    assert(rangeStart.size() == 1);
    assert(rangeStart.size() == rangeEnd.size());
    std::vector<Value> rangeStartVec(rangeStart.begin(), rangeStart.end());
    std::vector<Value> rangeEndVec(rangeEnd.begin(), rangeEnd.end());

    auto localThreadIdAndNumThreads =
        generateLocalThreadId(location, rewriter, reduceOp);
    auto localThreadId = std::get<0>(localThreadIdAndNumThreads);
    auto numThreads = std::get<1>(localThreadIdAndNumThreads);

    rewriter.create<gpu::BarrierOp>(location);

    auto elemType = sourceMemref.getType().cast<MemRefType>().getElementType();
    // for i = start[0] to end[0] step numThreads
    auto loop = rewriter.create<scf::ForOp>(location, rangeStartVec[0],
                                            rangeEndVec[0], numThreads);
    rewriter.setInsertionPointToStart(loop.getBody());
    auto loopIV = loop.getInductionVar();
    auto threadIndex =
        rewriter.create<arith::AddIOp>(location, loopIV, localThreadId);
    // // if (threadIndex < end[0])
    auto cmpOp = rewriter.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::slt, threadIndex, rangeEndVec[0]);
    auto ifOp = rewriter.create<scf::IfOp>(location, cmpOp,
                                           /*hasElseRegion=*/false);
    {
      auto ifBodyBuilder = ifOp.getThenBodyBuilder();
      auto initialValConst = decisionforest::createFloatConst(
          location, ifBodyBuilder, elemType,
          reduceOp.getInitialValue().convertToDouble());
      auto reductionDimSize =
          sourceMemref.getType().cast<MemRefType>().getDimSize(reductionDimVal);
      auto reductionDimSizeConst = ifBodyBuilder.create<arith::ConstantIndexOp>(
          location, reductionDimSize);
      auto zeroIndexConst =
          ifBodyBuilder.create<arith::ConstantIndexOp>(location, 0);
      auto oneIndexConst =
          ifBodyBuilder.create<arith::ConstantIndexOp>(location, 1);
      auto reductionLoop = ifBodyBuilder.create<scf::ForOp>(
          location, zeroIndexConst.getResult(),
          reductionDimSizeConst.getResult(), oneIndexConst.getResult(),
          ValueRange{initialValConst});
      ifBodyBuilder.setInsertionPointToStart(reductionLoop.getBody());
      std::vector<Value> memrefIndex{reductionLoop.getInductionVar(),
                                     threadIndex};
      auto loadVal = ifBodyBuilder.create<memref::LoadOp>(
          location, sourceMemref, memrefIndex);
      auto accumulator = reductionLoop.getBody()->getArguments()[1];
      auto addVal = ifBodyBuilder.create<arith::AddFOp>(
          location, loadVal.getResult(), accumulator);
      ifBodyBuilder.create<scf::YieldOp>(location, addVal.getResult());
      ifBodyBuilder.setInsertionPointAfter(reductionLoop);
      // Write value to target memref
      std::vector<Value> resultIndex{threadIndex};
      ifBodyBuilder.create<memref::StoreOp>(location,
                                            reductionLoop.getResults().front(),
                                            targetMemref, resultIndex);
    }
    // rewriter.setInsertionPointAfter(loop);
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
                        decisionforest::ReduceDimensionOp,
                        decisionforest::CooperativeReduceDimensionOp,
                        decisionforest::CooperativeReduceInplaceOp>();

    RewritePatternSet patterns(&getContext());
    patterns
        .add<ReduceOpLowering, ReduceInplaceOpLowering,
             ReduceDimensionOpLowering, CooperativeReduceDimensionOpLowering>(
            &getContext());

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
  // module->dump();
  decisionforest::RunCanonicalizerPass(context, module);
}

} // namespace decisionforest
} // namespace mlir