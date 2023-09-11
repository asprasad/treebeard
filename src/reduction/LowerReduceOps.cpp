#include "Dialect.h"
#include "OpLoweringUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
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

bool isLoopRangeOne(Value start, Value end) {
  if (start.getDefiningOp() && end.getDefiningOp() &&
      start.getDefiningOp()->hasTrait<OpTrait::ConstantLike>() &&
      end.getDefiningOp()->hasTrait<OpTrait::ConstantLike>()) {

    auto startConst = decisionforest::GetConstantIntValueFromMLIRValue(start);
    auto endConst = decisionforest::GetConstantIntValueFromMLIRValue(end);
    return (endConst - startConst) == 1;
  }

  return false;
}

void generateArgMax(ConversionPatternRewriter &rewriter, Location location,
                    scf::ForOp reductionLoop, Value targetMemref,
                    Value sourceMemref, std::vector<Value> &&targetIndices,
                    std::vector<Value> &&sourceIndices) {
  auto sourceMemrefType =
      sourceMemref.getType().cast<MemRefType>().getElementType();

  auto k = reductionLoop.getInductionVar();
  auto currentValue = rewriter.create<memref::LoadOp>(
      location, sourceMemrefType, static_cast<Value>(sourceMemref),
      sourceIndices);

  auto maxValue = reductionLoop.getLoopBody().getArgument(1);
  auto maxIndex = reductionLoop.getLoopBody().getArgument(2);

  auto compareResult = rewriter.create<arith::CmpFOp>(
      location, mlir::arith::CmpFPredicate::OGT, maxValue, currentValue);

  auto ifElse = rewriter.create<scf::IfOp>(
      location, TypeRange({sourceMemrefType, k.getType()}), compareResult,
      true);
  {
    auto thenBodyBuilder = ifElse.getThenBodyBuilder();
    thenBodyBuilder.create<scf::YieldOp>(location,
                                         ValueRange({maxValue, maxIndex}));
  }
  {
    auto elseBodyBuilder = ifElse.getElseBodyBuilder();
    elseBodyBuilder.create<scf::YieldOp>(
        location,
        ValueRange({static_cast<Value>(currentValue), static_cast<Value>(k)}));
  }

  rewriter.create<scf::YieldOp>(location, ifElse.getResults());

  rewriter.setInsertionPointAfter(reductionLoop);

  auto result = rewriter.create<arith::IndexCastOp>(
      location, targetMemref.getType().cast<MemRefType>().getElementType(),
      static_cast<Value>(reductionLoop.getResult(1)));
  rewriter.create<memref::StoreOp>(location, result, targetMemref,
                                   targetIndices);
}

std::set<int32_t> getMappedDimensionAsInteger(
    const mlir::Operation::operand_range &mappedDimensions) {
  std::set<int32_t> mappedDims;
  for (auto dim : mappedDimensions) {
    auto dimVal = decisionforest::GetConstantIntValueFromMLIRValue(dim);
    mappedDims.insert(dimVal);
  }
  return mappedDims;
}

void addLoopIfNeededAndUpdateIndices(
    ConversionPatternRewriter &rewriter, Location location,
    std::vector<Value> &loopIVs, std::vector<Value> &resultIndex,
    std::set<int32_t> &mappedDimsSet, Value start, Value end,
    Value oneIndexConst, Operation *outermostLoop) {
  if (isLoopRangeOne(start, end)) {
    loopIVs.push_back(start);
  } else {
    auto loop =
        rewriter.create<scf::ForOp>(location, start, end, oneIndexConst);
    if (!outermostLoop) {
      outermostLoop = loop.getOperation();
    }
    loopIVs.push_back(loop.getInductionVar());
    rewriter.setInsertionPointToStart(loop.getBody());
  }

  // If the last dimension is mapped, include it in the result index.
  if (mappedDimsSet.find(loopIVs.size() - 1) != mappedDimsSet.end()) {
    resultIndex.push_back(loopIVs.back());
  }
}

LogicalResult generateSimpleReductionLoopNest(
    Location location, ConversionPatternRewriter &rewriter, Value targetMemref,
    Value sourceMemref, std::set<int32_t> &&mappedDimSet,
    mlir::Operation::operand_range &&preReductionDimStart,
    mlir::Operation::operand_range &&preReductionDimEnd, int64_t reductionDim,
    mlir::Operation::operand_range &&postReductionDimStart,
    mlir::Operation::operand_range &&postReductionDimEnd,
    decisionforest::Reduction reduction) {

  // Iterate [rangeStart, rangeEnd)
  auto oneIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 1);
  auto zeroIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
  bool inplace = (sourceMemref == targetMemref);

  Operation *outermostLoop = nullptr;
  std::vector<Value> loopIVs;
  std::vector<Value> resultIndex;
  for (auto startEndPair :
       llvm::zip(preReductionDimStart, preReductionDimEnd)) {
    auto start = std::get<0>(startEndPair);
    auto end = std::get<1>(startEndPair);
    addLoopIfNeededAndUpdateIndices(rewriter, location, loopIVs, resultIndex,
                                    mappedDimSet, start, end, oneIndexConst,
                                    outermostLoop);
  }

  if (preReductionDimStart.size() < (size_t)reductionDim) {
    for (auto i = preReductionDimStart.size(); i < (size_t)reductionDim; i++) {

      auto dimSize = sourceMemref.getType().cast<MemRefType>().getDimSize(i);
      auto dimSizeConst =
          rewriter.create<arith::ConstantIndexOp>(location, dimSize);
      addLoopIfNeededAndUpdateIndices(
          rewriter, location, loopIVs, resultIndex, mappedDimSet,
          zeroIndexConst, dimSizeConst, oneIndexConst, outermostLoop);
    }
  }

  // for each value, sum over the reduction dimension
  auto sourceMemrefType = sourceMemref.getType().cast<MemRefType>();
  auto targetMemrefType = targetMemref.getType().cast<MemRefType>();

  auto reductionDimSize = sourceMemrefType.getDimSize(reductionDim);
  auto reductionDimSizeConst =
      rewriter.create<arith::ConstantIndexOp>(location, reductionDimSize);
  auto memrefElemType = sourceMemrefType.getElementType();

  ValueRange reductionLoopArgs;
  if (reduction == decisionforest::Reduction::kAdd) {
    reductionLoopArgs = ValueRange{};
  } else {
    if (!targetMemrefType.getElementType().isIntOrIndex()) {
      llvm::errs() << "Illegal result type for ArgMax reduction. Should be "
                      "an Integer type.\n";
      return mlir::failure();
    }
    auto indexConst = rewriter.create<arith::ConstantIndexOp>(location, -1);
    auto minusInfConst = decisionforest::createFloatConst(
        location, rewriter, memrefElemType, -INFINITY);
    reductionLoopArgs = ValueRange{minusInfConst, indexConst};
  }

  auto reductionLoop = rewriter.create<scf::ForOp>(
      location,
      inplace ? oneIndexConst.getResult() : zeroIndexConst.getResult(),
      reductionDimSizeConst.getResult(), oneIndexConst.getResult(),
      reductionLoopArgs);
  if (outermostLoop == nullptr) {
    outermostLoop = reductionLoop.getOperation();
  }

  // If in-place reduction, include the result dimension.
  if (inplace) {
    resultIndex.push_back(zeroIndexConst);
  }

  loopIVs.push_back(reductionLoop.getInductionVar());
  rewriter.setInsertionPointToStart(reductionLoop.getBody());
  for (auto startEndPair :
       llvm::zip(postReductionDimStart, postReductionDimEnd)) {

    auto start = std::get<0>(startEndPair);
    auto end = std::get<1>(startEndPair);

    addLoopIfNeededAndUpdateIndices(rewriter, location, loopIVs, resultIndex,
                                    mappedDimSet, start, end, oneIndexConst,
                                    outermostLoop);
  }

  if (reduction == decisionforest::Reduction::kAdd) {

    for (size_t j = loopIVs.size(); j < (size_t)sourceMemrefType.getRank();
         ++j) {
      auto start = rewriter.create<arith::ConstantIndexOp>(location, 0);
      auto end = rewriter.create<arith::ConstantIndexOp>(
          location, sourceMemrefType.getDimSize(j));
      addLoopIfNeededAndUpdateIndices(rewriter, location, loopIVs, resultIndex,
                                      mappedDimSet, start, end, oneIndexConst,
                                      outermostLoop);
    }

    if ((size_t)targetMemrefType.getRank() != resultIndex.size()) {
      llvm::errs() << "Target memref index and result index do not match\n";
      return mlir::failure();
    }

    auto loadVal =
        rewriter.create<memref::LoadOp>(location, sourceMemref, loopIVs);

    auto loadResultIndexVal =
        rewriter.create<memref::LoadOp>(location, targetMemref, resultIndex);
    auto addVal = rewriter.create<arith::AddFOp>(
        location, loadVal.getResult(), loadResultIndexVal.getResult());
    rewriter.create<memref::StoreOp>(location, addVal.getResult(), targetMemref,
                                     resultIndex);
  } else {
    if ((size_t)reductionDim != (loopIVs.size() - 1)) {
      llvm::errs()
          << "Wrong reduction dimension. It should be the last dimension\n";
      return mlir::failure();
    }

    if (reductionLoop.getInitArgs().size() != 2) {
      llvm::errs() << "Expected 2 args for ArgMax reduction.\n";
      return mlir::failure();
    }

    generateArgMax(rewriter, location, reductionLoop, targetMemref,
                   sourceMemref, std::move(resultIndex), std::move(loopIVs));
  }

  rewriter.setInsertionPointAfter(outermostLoop);
  return mlir::success();
}

struct ReduceDimensionOpLowering : public ConversionPattern {
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
    auto preReducedDimStart = reduceOp.getPreReductionDimensionStart();
    auto preReducedDimEnd = reduceOp.getPreReductionDimensionEnd();
    auto postReducedDimStart = reduceOp.getPostReductionDimensionStart();
    auto postReducedDimEnd = reduceOp.getPostReductionDimensionEnd();
    auto mappedDimensions = reduceOp.getMappedDimensions();
    auto mappedDimSet = getMappedDimensionAsInteger(mappedDimensions);

    auto reductionDim = reduceOp.getReductionDimension();
    auto reductionDimVal =
        decisionforest::GetConstantIntValueFromMLIRValue(reductionDim);

    auto sourceMemref = reduceOp.getSourceMemref();
    auto targetMemref = reduceOp.getTargetMemref();

    auto reductionType = reduceOp.getReductionTypeAttr().getReductionType();

    auto result = generateSimpleReductionLoopNest(
        location, rewriter, targetMemref, sourceMemref, std::move(mappedDimSet),
        std::move(preReducedDimStart), std::move(preReducedDimEnd),
        reductionDimVal, std::move(postReducedDimStart),
        std::move(postReducedDimEnd), reductionType);

    if (failed(result)) {
      return result;
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ReduceInplaceOpLowering : public ConversionPattern {
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
    auto preReducedDimStart = reduceOp.getPreReductionDimensionStart();
    auto preReducedDimEnd = reduceOp.getPreReductionDimensionEnd();
    auto postReducedDimStart = reduceOp.getPostReductionDimensionStart();
    auto postReducedDimEnd = reduceOp.getPostReductionDimensionEnd();

    auto reductionDim = reduceOp.getReductionDimension();
    auto reductionDimVal =
        decisionforest::GetConstantIntValueFromMLIRValue(reductionDim);

    auto sourceMemref = reduceOp.getTargetMemref();
    auto reductionType = reduceOp.getReductionTypeAttr().getReductionType();

    std::set<int32_t> mappedDimSet;
    auto sourceType = sourceMemref.getType().cast<MemRefType>();
    for (auto i = 0; i < sourceType.getRank(); ++i) {
      if (i != reductionDimVal) {
        mappedDimSet.insert(i);
      }
    }

    auto result = generateSimpleReductionLoopNest(
        location, rewriter, sourceMemref, sourceMemref, std::move(mappedDimSet),
        std::move(preReducedDimStart), std::move(preReducedDimEnd),
        reductionDimVal, std::move(postReducedDimStart),
        std::move(postReducedDimEnd), reductionType);

    if (failed(result)) {
      return result;
    }

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