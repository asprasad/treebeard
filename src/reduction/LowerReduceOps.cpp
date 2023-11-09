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
#include "LIRLoweringHelpers.h"
#include "LowerReduceOps.h"

using namespace mlir;

int64_t getConstantStepBetweenValues(mlir::Value start, mlir::Value end) {
  int64_t endValConst;
  int64_t startValConst;
  if (isContantInt(end, endValConst)) {
    startValConst = GetConstantIntValueFromMLIRValue(start);
    return endValConst - startValConst;
  } else {
    // the defining op of the end value must be an add op
    auto addOp = AssertOpIsOfType<arith::AddIOp>(end.getDefiningOp());
    auto constIncrement =
        addOp.getLhs() == start ? addOp.getRhs() : addOp.getLhs();
    return GetConstantIntValueFromMLIRValue(constIncrement);
  }
}

namespace mlir {
namespace decisionforest {

// Defined in GPURepresentations.cpp
mlir::gpu::KernelDim3 GetThreadID(mlir::Operation *op);
mlir::gpu::KernelDim3 GetBlockID(mlir::Operation *op);
int64_t GetNumberOfThreadsInThreadBlock(gpu::LaunchOp gpuLaunchOp);

// Defined in CodeGenStateMachine.cpp
Value GenerateLocalThreadId(ConversionPatternRewriter &rewriter,
                            Location location, gpu::LaunchOp launchOp);

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

void generateArgMax(OpBuilder &rewriter, Location location,
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
    auto dimVal = GetConstantIntValueFromMLIRValue(dim);
    mappedDims.insert(dimVal);
  }
  return mappedDims;
}

void addLoopIfNeededAndUpdateIndices(ConversionPatternRewriter &rewriter,
                                     Location location,
                                     std::vector<Value> &loopIVs,
                                     std::vector<Value> &resultIndex,
                                     std::set<int32_t> &mappedDimsSet,
                                     Value start, Value end, Value stepConst,
                                     Operation *outermostLoop) {

  auto loop = rewriter.create<scf::ForOp>(location, start, end, stepConst);
  if (!outermostLoop) {
    outermostLoop = loop.getOperation();
  }

  loopIVs.push_back(loop.getInductionVar());
  rewriter.setInsertionPointToStart(loop.getBody());

  // If the last dimension is mapped, include it in the result index.
  if (mappedDimsSet.find(loopIVs.size() - 1) != mappedDimsSet.end()) {
    resultIndex.push_back(loopIVs.back());
  }
}

LogicalResult generateAtomiceductionLoopNest(
    Location location, ConversionPatternRewriter &rewriter, Value targetMemref,
    Value sourceMemref, std::set<int32_t> &&mappedDimSet,
    mlir::Operation::operand_range &&preReductionDimStart,
    mlir::Operation::operand_range &&preReductionDimEnd, int64_t reductionDim,
    Value reductionDimIndex,
    mlir::Operation::operand_range &&postReductionDimStart,
    mlir::Operation::operand_range &&postReductionDimEnd,
    decisionforest::Reduction reduction, Value initialValueConst = Value()) {

  decisionforest::helpers::SaveAndRestoreInsertionPoint saveInsertionPoint(
      rewriter);

  // Iterate [rangeStart, rangeEnd)
  auto oneIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 1);

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

  // All the dimensions prior to the reduction dimension should be specified.
  assert(preReductionDimStart.size() == (size_t)reductionDim);

  // Insert at the point where reduction dim is supposed to appear.
  loopIVs.push_back(reductionDimIndex);

  // If we're doing an inplace atomic reduction, we need to check if the
  // all preReduction IVs are 0 and the reductionDimIndex==0. If they are,
  // we need to skip accumulating this value (the value at this index is a
  // destination value and we would be accumulating twice)
  if (sourceMemref == targetMemref) {
    auto trueConst = rewriter.create<arith::ConstantIntOp>(
        location, 1, rewriter.getIntegerType(1));
    auto currCondVal = trueConst.getResult();
    for (auto iv : loopIVs) {
      auto isZeroIndexCond = rewriter.create<arith::CmpIOp>(
          location, arith::CmpIPredicate::eq, iv,
          rewriter.create<arith::ConstantIndexOp>(location, 0));
      auto andCond = rewriter.create<arith::AndIOp>(location, currCondVal,
                                                    isZeroIndexCond);
      currCondVal = andCond.getResult();
    }
    // Negate currCondVal
    auto notAllZeros = rewriter.create<arith::XOrIOp>(location, currCondVal,
                                                      trueConst.getResult());
    auto ifOp = rewriter.create<scf::IfOp>(location, TypeRange({}),
                                           notAllZeros.getResult(), false);
    auto thenBodyBuilder = ifOp.getThenBodyBuilder();
    // auto yieldOp = thenBodyBuilder.create<scf::YieldOp>(location);
    rewriter.setInsertionPointToStart(thenBodyBuilder.getInsertionBlock());
  }

  // for each value, sum over the reduction dimension
  auto sourceMemrefType = sourceMemref.getType().cast<MemRefType>();
  auto targetMemrefType = targetMemref.getType().cast<MemRefType>();

  for (auto startEndPair :
       llvm::zip(postReductionDimStart, postReductionDimEnd)) {

    auto start = std::get<0>(startEndPair);
    auto end = std::get<1>(startEndPair);

    addLoopIfNeededAndUpdateIndices(rewriter, location, loopIVs, resultIndex,
                                    mappedDimSet, start, end, oneIndexConst,
                                    outermostLoop);
  }

  // All dimensions have to be specified.
  assert(static_cast<int64_t>(loopIVs.size()) == sourceMemrefType.getRank());

  assert(reduction == decisionforest::Reduction::kAdd);

  auto loadVal =
      rewriter.create<memref::LoadOp>(location, sourceMemref, loopIVs);

  if (sourceMemref == targetMemref) {
    // resultIndex' = [(0's as required), resultIndex]
    // size of resultIndex' = targetMemrefType.getRank()
    std::vector<Value> resultIndexPrime;
    auto zeroIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
    for (size_t i = 0; i < targetMemrefType.getRank() - resultIndex.size(); ++i)
      resultIndexPrime.push_back(zeroIndexConst);
    resultIndexPrime.insert(resultIndexPrime.end(), resultIndex.begin(),
                            resultIndex.end());

    if ((size_t)targetMemrefType.getRank() != resultIndexPrime.size()) {
      llvm::errs() << "Target memref index and result index do not match\n";
      return mlir::failure();
    }

    rewriter.create<memref::AtomicRMWOp>(
        location, mlir::arith::AtomicRMWKind::addf, loadVal.getResult(),
        targetMemref, resultIndexPrime);
  } else {
    if ((size_t)targetMemrefType.getRank() != resultIndex.size()) {
      llvm::errs() << "Target memref index and result index do not match\n";
      return mlir::failure();
    }
    rewriter.create<memref::AtomicRMWOp>(
        location, mlir::arith::AtomicRMWKind::addf, loadVal.getResult(),
        targetMemref, resultIndex);
  }

  return mlir::success();
}

LogicalResult generateSimpleReductionLoopNest(
    Location location, ConversionPatternRewriter &rewriter, Value targetMemref,
    Value sourceMemref, std::set<int32_t> &&mappedDimSet,
    mlir::Operation::operand_range &&preReductionDimStart,
    mlir::Operation::operand_range &&preReductionDimEnd, int64_t reductionDim,
    mlir::Operation::operand_range &&postReductionDimStart,
    mlir::Operation::operand_range &&postReductionDimEnd,
    decisionforest::Reduction reduction, int32_t vectorWidth = -1,
    Value initialValueConst = Value()) {

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

  if (preReductionDimEnd.size() != (size_t)reductionDim) {
    llvm::errs() << "Reduction dimension should should immediately follow the "
                    "Pre-Reduction dimensions\n";
    return mlir::failure();
  }

  // for each value, sum over the reduction dimension
  auto sourceMemrefType = sourceMemref.getType().cast<MemRefType>();
  auto targetMemrefType = targetMemref.getType().cast<MemRefType>();

  auto reductionDimSize = sourceMemrefType.getDimSize(reductionDim);
  auto reductionDimSizeConst =
      rewriter.create<arith::ConstantIndexOp>(location, reductionDimSize);
  auto memrefElemType = sourceMemrefType.getElementType();

  // If in-place reduction, include the result dimension.
  if (inplace) {
    resultIndex.push_back(zeroIndexConst);
  }

  // Push-back empty value for the reduction dimension. Will be filled later.
  loopIVs.push_back(Value());

  for (auto startEndPair :
       llvm::zip(postReductionDimStart, postReductionDimEnd)) {
    auto start = std::get<0>(startEndPair);
    auto end = std::get<1>(startEndPair);
    bool lastLoop = (end == postReductionDimEnd.back()) &&
                    (start == postReductionDimStart.back());
    auto stepConst = oneIndexConst;
    // If
    if (lastLoop && vectorWidth > 1) {
      stepConst =
          rewriter.create<arith::ConstantIndexOp>(location, vectorWidth);
    }
    addLoopIfNeededAndUpdateIndices(rewriter, location, loopIVs, resultIndex,
                                    mappedDimSet, start, end, stepConst,
                                    outermostLoop);
  }

  std::vector<Value> reductionLoopArgs;
  if (reduction == decisionforest::Reduction::kAdd) {
    if (!initialValueConst)
      initialValueConst = decisionforest::createFloatConst(location, rewriter,
                                                           memrefElemType, 0.0);
    if (vectorWidth > 1) {
      auto vectorType = VectorType::get(vectorWidth, memrefElemType);
      auto vectorInitialValue = rewriter.create<vector::SplatOp>(
          location, vectorType, initialValueConst);
      reductionLoopArgs.push_back(vectorInitialValue);
    } else {
      reductionLoopArgs.push_back(initialValueConst);
    }
  } else {
    assert(vectorWidth == -1);
    if (!targetMemrefType.getElementType().isIntOrIndex()) {
      llvm::errs() << "Illegal result type for ArgMax reduction. Should be "
                      "an Integer type.\n";
      return mlir::failure();
    }
    auto indexConst =
        rewriter.create<arith::ConstantIndexOp>(location, 0).getResult();
    auto minusInfConst = decisionforest::createFloatConst(
        location, rewriter, memrefElemType, -INFINITY);
    reductionLoopArgs.push_back(minusInfConst);
    reductionLoopArgs.push_back(indexConst);
  }

  auto reductionLoop = rewriter.create<scf::ForOp>(
      location, zeroIndexConst, reductionDimSizeConst, oneIndexConst,
      reductionLoopArgs);
  if (outermostLoop == nullptr) {
    outermostLoop = reductionLoop.getOperation();
  }

  rewriter.setInsertionPointToStart(reductionLoop.getBody());
  // Insert at the point where reduction dim is supposed to appear.
  loopIVs[reductionDim] = reductionLoop.getInductionVar();

  if (loopIVs.size() != (size_t)sourceMemrefType.getRank()) {
    llvm::errs() << "LowerReduceOps: Not enough indices passed to address "
                    "sourceMemref.\n";
    return mlir::failure();
  }

  if (reduction == decisionforest::Reduction::kAdd) {

    if ((size_t)targetMemrefType.getRank() != resultIndex.size()) {
      llvm::errs() << "Target memref index and result index do not match\n";
      return mlir::failure();
    }

    if (vectorWidth > 1) {
      auto vectorType = VectorType::get(vectorWidth, memrefElemType);
      auto loadVal = rewriter.create<vector::LoadOp>(location, vectorType,
                                                     sourceMemref, loopIVs);

      auto accumulator = reductionLoop.getLoopBody().getArgument(1);
      auto addVal = rewriter.create<arith::AddFOp>(
          location, loadVal.getResult(), accumulator);
      rewriter.create<scf::YieldOp>(location, addVal.getResult());
      rewriter.setInsertionPointAfter(reductionLoop);
      rewriter.create<vector::StoreOp>(location, reductionLoop.getResult(0),
                                       targetMemref, resultIndex);
    } else {
      auto loadVal =
          rewriter.create<memref::LoadOp>(location, sourceMemref, loopIVs);

      auto accumulator = reductionLoop.getLoopBody().getArgument(1);
      auto addVal = rewriter.create<arith::AddFOp>(
          location, loadVal.getResult(), accumulator);
      rewriter.create<scf::YieldOp>(location, addVal.getResult());
      rewriter.setInsertionPointAfter(reductionLoop);
      rewriter.create<memref::StoreOp>(location, reductionLoop.getResult(0),
                                       targetMemref, resultIndex);
    }
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

    // TODO_Ashwin offsets not yet supported here. Probably not needed for CPU?
    auto targetOffsets = reduceOp.getTargetMemrefOffsets();
    assert(targetOffsets.size() == 0 &&
           "Offsets not yet supported for ReduceDimensionOp");

    // The values for the outer indices. These are fixed
    auto preReducedDimStart = reduceOp.getPreReductionDimensionStart();
    auto preReducedDimEnd = reduceOp.getPreReductionDimensionEnd();
    auto postReducedDimStart = reduceOp.getPostReductionDimensionStart();
    auto postReducedDimEnd = reduceOp.getPostReductionDimensionEnd();
    auto mappedDimensions = reduceOp.getMappedDimensions();
    auto mappedDimSet = getMappedDimensionAsInteger(mappedDimensions);

    auto reductionDim = reduceOp.getReductionDimension();
    auto reductionDimVal = GetConstantIntValueFromMLIRValue(reductionDim);

    auto sourceMemref = reduceOp.getSourceMemref();
    auto targetMemref = reduceOp.getTargetMemref();

    int32_t vectorWidth = -1;
    auto vectorWidthAttr = reduceOp->getAttr("vectorReduce");
    if (vectorWidthAttr) {
      vectorWidth = vectorWidthAttr.cast<IntegerAttr>().getInt();
    }
    auto reductionType = reduceOp.getReductionTypeAttr().getReductionType();
    auto initialValue = decisionforest::createFloatConst(
        location, rewriter,
        sourceMemref.getType().cast<MemRefType>().getElementType(),
        reduceOp.getInitialValue().convertToDouble());

    auto result = generateSimpleReductionLoopNest(
        location, rewriter, targetMemref, sourceMemref, std::move(mappedDimSet),
        std::move(preReducedDimStart), std::move(preReducedDimEnd),
        reductionDimVal, std::move(postReducedDimStart),
        std::move(postReducedDimEnd), reductionType, vectorWidth, initialValue);

    if (failed(result)) {
      return result;
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct AtomicReduceDimensionOpLowering : public ConversionPattern {
  AtomicReduceDimensionOpLowering(MLIRContext *ctx)
      : ConversionPattern(
            mlir::decisionforest::AtomicReduceDimensionOp::getOperationName(),
            1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto reduceOp =
        AssertOpIsOfType<decisionforest::AtomicReduceDimensionOp>(op);
    auto location = reduceOp.getLoc();

    // The values for the outer indices. These are fixed
    auto preReducedDimStart = reduceOp.getPreReductionDimensionStart();
    auto preReducedDimEnd = reduceOp.getPreReductionDimensionEnd();
    auto postReducedDimStart = reduceOp.getPostReductionDimensionStart();
    auto postReducedDimEnd = reduceOp.getPostReductionDimensionEnd();
    auto mappedDimensions = reduceOp.getMappedDimensions();
    auto mappedDimSet = getMappedDimensionAsInteger(mappedDimensions);

    auto reductionDim = reduceOp.getReductionDimension();
    auto reductionDimVal = GetConstantIntValueFromMLIRValue(reductionDim);

    auto reductionDimIndex = reduceOp.getReductionDimensionIndex();

    auto sourceMemref = reduceOp.getSourceMemref();
    auto targetMemref = reduceOp.getTargetMemref();

    auto reductionType = reduceOp.getReductionTypeAttr().getReductionType();
    // TODO_Ashwin Initial value is currently ignored. It is
    // assumed that the target memref is already initialized with the
    // initial value.
    auto initialValue = decisionforest::createFloatConst(
        location, rewriter,
        sourceMemref.getType().cast<MemRefType>().getElementType(),
        reduceOp.getInitialValue().convertToDouble());

    auto result = generateAtomiceductionLoopNest(
        location, rewriter, targetMemref, sourceMemref, std::move(mappedDimSet),
        std::move(preReducedDimStart), std::move(preReducedDimEnd),
        reductionDimVal, reductionDimIndex, std::move(postReducedDimStart),
        std::move(postReducedDimEnd), reductionType, initialValue);

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
    auto reductionDimVal = GetConstantIntValueFromMLIRValue(reductionDim);

    auto sourceMemref = reduceOp.getTargetMemref();
    auto reductionType = reduceOp.getReductionTypeAttr().getReductionType();

    int32_t vectorWidth = -1;
    auto vectorWidthAttr = reduceOp->getAttr("vectorReduce");
    if (vectorWidthAttr) {
      vectorWidth = vectorWidthAttr.cast<IntegerAttr>().getInt();
    }

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
        std::move(postReducedDimEnd), reductionType, vectorWidth);

    if (failed(result)) {
      return result;
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

template <typename OpT>
std::tuple<Value, Value, int64_t>
generateLocalThreadId(Location location, ConversionPatternRewriter &rewriter,
                      OpT reduceOp) {

  auto threadId = decisionforest::GetThreadID(reduceOp);
  auto localThreadIdX = rewriter.create<arith::SubIOp>(
      location, threadId.x, reduceOp.getBlockXStart());
  auto localThreadIdY = rewriter.create<arith::SubIOp>(
      location, threadId.y, reduceOp.getBlockYStart());
  // auto localThreadIdZ = rewriter.create<arith::SubIOp>(
  //     location, threadId.z, reduceOp.getBlockZStart());

  auto startZ = GetConstantIntValueFromMLIRValue(reduceOp.getBlockZStart());
  auto endZ = GetConstantIntValueFromMLIRValue(reduceOp.getBlockZEnd());
  assert(startZ == 0 && endZ == 1 && "Only 2D thread blocks supported for now");

  auto xSize = getConstantStepBetweenValues(reduceOp.getBlockXStart(),
                                            reduceOp.getBlockXEnd());

  auto ySize = getConstantStepBetweenValues(reduceOp.getBlockYStart(),
                                            reduceOp.getBlockYEnd());

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

  return std::make_tuple(localThreadId, numThreads, xSize * ySize);
}

struct CooperativeReduceDimensionOpLowering : public ConversionPattern {

  CooperativeReduceDimensionOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::decisionforest::CooperativeReduceDimensionOp::
                              getOperationName(),
                          1 /*benefit*/, ctx) {}

  void generateCooperativeThreadReductionStrategy(
      Location location, ConversionPatternRewriter &rewriter,
      Value sourceMemref, Value targetMemref, Value localThreadId,
      Value numThreads, const std::vector<Value> &rangeStartVec,
      const std::vector<Value> &rangeEndVec, int64_t reductionDimVal,
      decisionforest::CooperativeReduceDimensionOp reduceOp,
      int64_t numThreadsVal, int64_t numRowsToProcess) const {
    assert(numRowsToProcess < numThreadsVal);
    assert(numThreadsVal % numRowsToProcess ==
           0); // Each reduction must have a fixed number of threads
    auto threadsPerReductionVal = numThreadsVal / numRowsToProcess;
    auto threadsPerReduction = rewriter.create<arith::ConstantIndexOp>(
        location, threadsPerReductionVal);

    auto reductionDimSize =
        sourceMemref.getType().cast<MemRefType>().getDimSize(reductionDimVal);

    auto rowOffsetForThread = rewriter.create<arith::FloorDivSIOp>(
        location, localThreadId, threadsPerReduction);
    auto rowIndexForThread = rewriter.create<arith::AddIOp>(
        location, rowOffsetForThread, rangeStartVec[0]);

    auto threadIndexInReductionGroup = rewriter.create<arith::RemSIOp>(
        location, localThreadId, threadsPerReduction);

    if (reductionDimSize > threadsPerReductionVal) {
      // Generate a loop to reduce reductionDimSize/threadsPerReduction elems
      // Generate a tree based reduce for the threadsPerReduction elems
      llvm::errs() << "Not implemented!\n";
    } else {
      // Generate a tree based reduce for the reductionDimSize elems
      auto numStages = std::ceil(std::log2(reductionDimSize));
      auto reductionDimSizeConst =
          rewriter.create<arith::ConstantIndexOp>(location, reductionDimSize);
      auto stageSize = 1;
      for (auto stage = 0; stage < numStages; ++stage) {
        auto stageSizeConst =
            rewriter.create<arith::ConstantIndexOp>(location, stageSize);
        auto reductionDimIndex = rewriter.create<arith::AddIOp>(
            location, threadIndexInReductionGroup, stageSizeConst);
        auto cmpOp = rewriter.create<arith::CmpIOp>(
            location, arith::CmpIPredicate::slt, reductionDimIndex,
            reductionDimSizeConst);
        auto ifOp = rewriter.create<scf::IfOp>(location, cmpOp, false);
        {
          auto ifBodyBuilder = ifOp.getThenBodyBuilder();
          std::vector<Value> memrefIndex{threadIndexInReductionGroup,
                                         rowIndexForThread},
              otherValIndex{reductionDimIndex, rowIndexForThread};
          auto loadVal = ifBodyBuilder.create<memref::LoadOp>(
              location, sourceMemref, memrefIndex);
          auto otherVal = ifBodyBuilder.create<memref::LoadOp>(
              location, sourceMemref, otherValIndex);
          auto sum =
              ifBodyBuilder.create<arith::AddFOp>(location, loadVal, otherVal);
          ifBodyBuilder.create<memref::StoreOp>(location, sum, sourceMemref,
                                                memrefIndex);
        }
        rewriter.create<gpu::BarrierOp>(location);
        stageSize *= 2;
      }
      // if thread's reduction group offset is 0, write the result
      auto zeroIndexConst =
          rewriter.create<arith::ConstantIndexOp>(location, 0);
      auto cmpOp = rewriter.create<arith::CmpIOp>(
          location, arith::CmpIPredicate::eq, threadIndexInReductionGroup,
          zeroIndexConst);
      auto ifOp = rewriter.create<scf::IfOp>(location, cmpOp, false);
      {
        auto ifBodyBuilder = ifOp.getThenBodyBuilder();
        std::vector<Value> memrefIndex{threadIndexInReductionGroup,
                                       rowIndexForThread};
        auto loadVal = ifBodyBuilder.create<memref::LoadOp>(
            location, sourceMemref, memrefIndex);
        // TODO_Ashwin need to add the initial value here
        // Get the value of the initial value attr
        auto initialValAttr = reduceOp.getInitialValue().convertToDouble();
        auto initialValConst = decisionforest::createFloatConst(
            location, ifBodyBuilder,
            sourceMemref.getType().cast<MemRefType>().getElementType(),
            initialValAttr);
        auto sum = ifBodyBuilder.create<arith::AddFOp>(location, loadVal,
                                                       initialValConst);
        // ifBodyBuilder.create<memref::StoreOp>(location, sum.getResult(),
        //                                       targetMemref,
        //                                       ValueRange{rowIndexForThread});
        auto targetMemrefOffset = reduceOp.getTargetMemrefOffsets();
        if (targetMemrefOffset.size() == 0) {
          // Write value to target memref
          std::vector<Value> resultIndex{rowIndexForThread};
          ifBodyBuilder.create<memref::StoreOp>(location, sum.getResult(),
                                                targetMemref, resultIndex);
        } else {
          // TODO_Ashwin need to handle case where there is more than one offset
          // here Is it needed though? We'll have to write all class values
          // anyway
          assert(targetMemrefOffset.size() == 1);
          // actual index = threadIndex + targetMemrefOffset[0]
          auto actualIndex = ifBodyBuilder.create<arith::AddIOp>(
              location, rowIndexForThread, targetMemrefOffset[0]);
          std::vector<Value> resultIndex{actualIndex};
          ifBodyBuilder.create<memref::StoreOp>(location, sum.getResult(),
                                                targetMemref, resultIndex);
        }
      }
    }
  }

  void generateSingleThreadReductionStrategy(
      Location location, ConversionPatternRewriter &rewriter,
      Value sourceMemref, Value targetMemref, Value localThreadId,
      Value numThreads, const std::vector<Value> &rangeStartVec,
      const std::vector<Value> &rangeEndVec, int64_t reductionDimVal,
      decisionforest::CooperativeReduceDimensionOp reduceOp) const {

    assert(rangeStartVec.size() == rangeEndVec.size());
    assert(rangeStartVec.size() == 1);

    auto elemType = sourceMemref.getType().cast<MemRefType>().getElementType();
    // for i = start[0] to end[0] step numThreads
    auto loop = rewriter.create<scf::ForOp>(location, rangeStartVec[0],
                                            rangeEndVec[0], numThreads);
    rewriter.setInsertionPointToStart(loop.getBody());
    auto loopIV = loop.getInductionVar();
    auto threadIndexAddOp =
        rewriter.create<arith::AddIOp>(location, loopIV, localThreadId);
    Value threadIndex = threadIndexAddOp.getResult();
    auto targetMemrefOffset = reduceOp.getTargetMemrefOffsets();
    if (targetMemrefOffset.size() != 0) {
      // TODO_Ashwin need to handle case where there is more than one offset
      // here Is it needed though? We'll have to write all class values anyway
      assert(targetMemrefOffset.size() == 1);
      auto privatizedBufferIndex = rewriter.create<arith::SubIOp>(
          location, threadIndex, targetMemrefOffset[0]);
      threadIndex = privatizedBufferIndex.getResult();
    }
    // // if (threadIndex < end[0])
    auto cmpOp = rewriter.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::slt, threadIndexAddOp.getResult(),
        rangeEndVec[0]);
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
      // if (targetMemrefOffset.size() == 0) {
      // Write value to target memref
      std::vector<Value> resultIndex{threadIndexAddOp.getResult()};
      ifBodyBuilder.create<memref::StoreOp>(location,
                                            reductionLoop.getResults().front(),
                                            targetMemref, resultIndex);
      // ifBodyBuilder.create<gpu::PrintfOp>(
      //     location, "Result index = %ld\n",
      //     ValueRange{threadIndexAddOp.getResult()});

      // } else {
      //   // actual index = threadIndex + targetMemrefOffset[0]
      //   std::vector<Value> resultIndex{actualIndex};
      //   // Print the result index
      //   ifBodyBuilder.create<gpu::PrintfOp>(location, "Result index = %ld\n",
      //                                       ValueRange{actualIndex});
      //   ifBodyBuilder.create<memref::StoreOp>(
      //       location, reductionLoop.getResults().front(), targetMemref,
      //       resultIndex);
      // }
    }
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto reduceOp =
        AssertOpIsOfType<decisionforest::CooperativeReduceDimensionOp>(op);
    auto location = reduceOp.getLoc();

    // auto targetMemrefOffsets = reduceOp.getTargetMemrefOffsets();

    // The values for the outer indices. These are fixed
    auto reducedDims = reduceOp.getReducedDimensions();
    assert(reducedDims.empty());

    auto reductionDim = reduceOp.getReductionDimension();
    auto reductionDimVal = GetConstantIntValueFromMLIRValue(reductionDim);
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
    // std::cout << "Range size = "
    //           << getConstantStepBetweenValues(rangeStartVec[0],
    //           rangeEndVec[0])
    //           << "\n";

    auto localThreadIdAndNumThreads =
        generateLocalThreadId(location, rewriter, reduceOp);
    auto localThreadId = std::get<0>(localThreadIdAndNumThreads);
    auto numThreads = std::get<1>(localThreadIdAndNumThreads);
    auto numThreadsVal = std::get<2>(localThreadIdAndNumThreads);

    rewriter.create<gpu::BarrierOp>(location);

    // if the number of reductions to perform is greater than
    // the number of threads, then use the single thread reduction strategy
    // (Each thread fully performs a single reduction)
    auto numRowsToProcess =
        getConstantStepBetweenValues(rangeStartVec[0], rangeEndVec[0]);

    if (true) // (numRowsToProcess >= numThreadsVal)
      generateSingleThreadReductionStrategy(
          location, rewriter, sourceMemref, targetMemref, localThreadId,
          numThreads, rangeStartVec, rangeEndVec, reductionDimVal, reduceOp);
    else
      generateCooperativeThreadReductionStrategy(
          location, rewriter, sourceMemref, targetMemref, localThreadId,
          numThreads, rangeStartVec, rangeEndVec, reductionDimVal, reduceOp,
          numThreadsVal, numRowsToProcess);

    // rewriter.setInsertionPointAfter(loop);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct CooperativeArgMaxReduceOpLowering : public ConversionPattern {

  CooperativeArgMaxReduceOpLowering(MLIRContext *ctx)
      : ConversionPattern(
            mlir::decisionforest::CooperativeReduceArgMaxOp::getOperationName(),
            1 /*benefit*/, ctx) {}

  void generateSingleThreadArgMaxReduction(
      Location location, ConversionPatternRewriter &rewriter,
      Value sourceMemref, Value targetMemref, Value localThreadId,
      Value numThreads, const std::vector<Value> &rangeStartVec,
      const std::vector<Value> &rangeEndVec, int64_t reductionDimVal,
      decisionforest::CooperativeReduceArgMaxOp reduceOp) const {

    assert(rangeStartVec.size() == rangeEndVec.size());
    assert(rangeStartVec.size() == 2);
    assert(getConstantIntValue(rangeStartVec[0]).value() == 0);
    assert(getConstantIntValue(rangeEndVec[0]).value() == 1);

    auto elemType = sourceMemref.getType().cast<MemRefType>().getElementType();
    // for i = start[0] to end[0] step numThreads
    auto loop = rewriter.create<scf::ForOp>(location, rangeStartVec[1],
                                            rangeEndVec[1], numThreads);
    rewriter.setInsertionPointToStart(loop.getBody());
    auto loopIV = loop.getInductionVar();
    auto threadIndexAddOp =
        rewriter.create<arith::AddIOp>(location, loopIV, localThreadId);
    Value threadIndex = threadIndexAddOp.getResult();
    auto targetMemrefOffset = reduceOp.getTargetMemrefOffsets();
    if (targetMemrefOffset.size() != 0) {
      // TODO_Ashwin need to handle case where there is more than one offset
      // here Is it needed though? We'll have to write all class values anyway
      assert(targetMemrefOffset.size() == 1);
      auto privatizedBufferIndex = rewriter.create<arith::SubIOp>(
          location, threadIndex, targetMemrefOffset[0]);
      threadIndex = privatizedBufferIndex.getResult();
    }
    // // if (threadIndex < end[0])
    auto cmpOp = rewriter.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::slt, threadIndexAddOp.getResult(),
        rangeEndVec[1]);
    auto ifOp = rewriter.create<scf::IfOp>(location, cmpOp,
                                           /*hasElseRegion=*/false);
    {
      auto ifBodyBuilder = ifOp.getThenBodyBuilder();

      auto reductionDimSize =
          sourceMemref.getType().cast<MemRefType>().getDimSize(reductionDimVal);
      auto reductionDimSizeConst = ifBodyBuilder.create<arith::ConstantIndexOp>(
          location, reductionDimSize);
      auto zeroIndexConst =
          ifBodyBuilder.create<arith::ConstantIndexOp>(location, 0);
      auto oneIndexConst =
          ifBodyBuilder.create<arith::ConstantIndexOp>(location, 1);

      std::vector<Value> reductionLoopArgs;
      auto minusInfConst = decisionforest::createFloatConst(
          location, ifBodyBuilder, elemType, -INFINITY);
      reductionLoopArgs.push_back(minusInfConst);
      reductionLoopArgs.push_back(zeroIndexConst.getResult());

      auto reductionLoop = ifBodyBuilder.create<scf::ForOp>(
          location, zeroIndexConst.getResult(),
          reductionDimSizeConst.getResult(), oneIndexConst.getResult(),
          reductionLoopArgs);
      ifBodyBuilder.setInsertionPointToStart(reductionLoop.getBody());
      std::vector<Value> memrefIndex{zeroIndexConst, threadIndex,
                                     reductionLoop.getInductionVar()};
      std::vector<Value> storeIndex{threadIndexAddOp.getResult()};
      generateArgMax(ifBodyBuilder, location, reductionLoop, targetMemref,
                     sourceMemref, std::move(storeIndex),
                     std::move(memrefIndex));
    }
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto reduceOp =
        AssertOpIsOfType<decisionforest::CooperativeReduceArgMaxOp>(op);
    auto location = reduceOp.getLoc();

    // The values for the outer indices. These are fixed

    auto reductionDim = reduceOp.getReductionDimension();
    auto reductionDimVal = GetConstantIntValueFromMLIRValue(reductionDim);
    assert(reductionDimVal == 2);

    auto rangeStart = reduceOp.getPreReductionDimensionStart();
    auto rangeEnd = reduceOp.getPreReductionDimensionEnd();

    auto sourceMemref = reduceOp.getSourceMemref();
    // auto sourceMemrefType = sourceMemref.getType().cast<MemRefType>();
    auto targetMemref = reduceOp.getTargetMemref();

    assert(rangeStart.size() == 2);
    assert(rangeStart.size() == rangeEnd.size());
    std::vector<Value> rangeStartVec(rangeStart.begin(), rangeStart.end());
    std::vector<Value> rangeEndVec(rangeEnd.begin(), rangeEnd.end());
    // std::cout << "Range size = "
    //           << getConstantStepBetweenValues(rangeStartVec[0],
    //           rangeEndVec[0])
    //           << "\n";

    auto localThreadIdAndNumThreads =
        generateLocalThreadId(location, rewriter, reduceOp);
    auto localThreadId = std::get<0>(localThreadIdAndNumThreads);
    auto numThreads = std::get<1>(localThreadIdAndNumThreads);

    rewriter.create<gpu::BarrierOp>(location);

    // if the number of reductions to perform is greater than
    // the number of threads, then use the single thread reduction strategy
    // (Each thread fully performs a single reduction)
    // auto numRowsToProcess =
    //     getConstantStepBetweenValues(rangeStartVec[0], rangeEndVec[0]);

    generateSingleThreadArgMaxReduction(
        location, rewriter, sourceMemref, targetMemref, localThreadId,
        numThreads, rangeStartVec, rangeEndVec, reductionDimVal, reduceOp);

    // rewriter.setInsertionPointAfter(loop);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct CooperativeInplaceReduceDimensionOpLowering : public ConversionPattern {

  CooperativeInplaceReduceDimensionOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::decisionforest::CooperativeReduceInplaceOp::
                              getOperationName(),
                          1 /*benefit*/, ctx) {}

  std::tuple<Value, Value, int64_t> generateLocalThreadId(
      Location location, ConversionPatternRewriter &rewriter,
      decisionforest::CooperativeReduceInplaceOp reduceOp) const {

    auto threadId = decisionforest::GetThreadID(reduceOp);
    auto localThreadIdX = rewriter.create<arith::SubIOp>(
        location, threadId.x, reduceOp.getBlockXStart());
    auto localThreadIdY = rewriter.create<arith::SubIOp>(
        location, threadId.y, reduceOp.getBlockYStart());
    // auto localThreadIdZ = rewriter.create<arith::SubIOp>(
    //     location, threadId.z, reduceOp.getBlockZStart());

    auto startZ = GetConstantIntValueFromMLIRValue(reduceOp.getBlockZStart());
    auto endZ = GetConstantIntValueFromMLIRValue(reduceOp.getBlockZEnd());
    assert(startZ == 0 && endZ == 1 &&
           "Only 2D thread blocks supported for now");

    auto xSize = getConstantStepBetweenValues(reduceOp.getBlockXStart(),
                                              reduceOp.getBlockXEnd());

    auto ySize = getConstantStepBetweenValues(reduceOp.getBlockYStart(),
                                              reduceOp.getBlockYEnd());

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

    return std::make_tuple(localThreadId, numThreads, xSize * ySize);
  }

  void generateSingleThreadReductionStrategy(
      Location location, ConversionPatternRewriter &rewriter,
      Value targetMemref, Value localThreadId, Value numThreads,
      const std::vector<Value> &rangeStartVec,
      const std::vector<Value> &rangeEndVec, int64_t reductionDimVal,
      decisionforest::CooperativeReduceInplaceOp reduceOp) const {

    assert(rangeStartVec.size() == rangeEndVec.size());
    assert(rangeStartVec.size() == 1 || rangeStartVec.size() == 2);

    auto elemType = targetMemref.getType().cast<MemRefType>().getElementType();
    // for i = start[0] to end[0] step numThreads
    auto loop = rewriter.create<scf::ForOp>(location, rangeStartVec[0],
                                            rangeEndVec[0], numThreads);
    rewriter.setInsertionPointToStart(loop.getBody());
    auto loopIV = loop.getInductionVar();
    auto threadIndexAddOp =
        rewriter.create<arith::AddIOp>(location, loopIV, localThreadId);
    Value threadIndex = threadIndexAddOp.getResult();
    // auto targetMemrefOffset = reduceOp.getTargetMemrefOffsets();
    // if (targetMemrefOffset.size() != 0) {
    //   // TODO_Ashwin need to handle case where there is more than one offset
    //   // here Is it needed though? We'll have to write all class values
    //   anyway assert(targetMemrefOffset.size() == 1); auto
    //   privatizedBufferIndex = rewriter.create<arith::SubIOp>(
    //       location, threadIndex, targetMemrefOffset[0]);
    //   threadIndex = privatizedBufferIndex.getResult();
    // }
    // // if (threadIndex < end[0])
    auto cmpOp = rewriter.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::slt, threadIndexAddOp.getResult(),
        rangeEndVec[0]);
    auto ifOp = rewriter.create<scf::IfOp>(location, cmpOp,
                                           /*hasElseRegion=*/false);
    {
      // if the memref has an additional dimension, add one more loop
      auto ifBodyBuilder = ifOp.getThenBodyBuilder();

      auto zeroIndexConst =
          ifBodyBuilder.create<arith::ConstantIndexOp>(location, 0);
      auto oneIndexConst =
          ifBodyBuilder.create<arith::ConstantIndexOp>(location, 1);

      std::vector<Value> memrefLoadIndex = {threadIndex},
                         memrefStoreIndex = {threadIndex};
      // create a for loop between rangeStartVec[1] and rangeEndVec[1]
      if (rangeStartVec.size() == 2) {
        auto innerLoop = ifBodyBuilder.create<scf::ForOp>(
            location, rangeStartVec[1], rangeEndVec[1],
            oneIndexConst.getResult());
        ifBodyBuilder.setInsertionPointToStart(innerLoop.getBody());

        auto innerLoopIV = innerLoop.getInductionVar();
        memrefLoadIndex.push_back(innerLoopIV);
        memrefStoreIndex.push_back(innerLoopIV);
      }
      auto initialValConst = decisionforest::createFloatConst(
          location, ifBodyBuilder, elemType, 0.0);
      auto reductionDimSize =
          targetMemref.getType().cast<MemRefType>().getDimSize(reductionDimVal);
      auto reductionDimSizeConst = ifBodyBuilder.create<arith::ConstantIndexOp>(
          location, reductionDimSize);

      auto reductionLoop = ifBodyBuilder.create<scf::ForOp>(
          location, zeroIndexConst.getResult(),
          reductionDimSizeConst.getResult(), oneIndexConst.getResult(),
          ValueRange{initialValConst});
      ifBodyBuilder.setInsertionPointToStart(reductionLoop.getBody());
      // TODO_Ashwin this cannot be just a single index for the non-reduction
      // dims. Won't work for multi-class.
      // Should we add another loop to
      memrefLoadIndex.insert(memrefLoadIndex.begin(),
                             reductionLoop.getInductionVar());
      auto loadVal = ifBodyBuilder.create<memref::LoadOp>(
          location, targetMemref, memrefLoadIndex);
      auto accumulator = reductionLoop.getBody()->getArguments()[1];
      auto addVal = ifBodyBuilder.create<arith::AddFOp>(
          location, loadVal.getResult(), accumulator);
      ifBodyBuilder.create<scf::YieldOp>(location, addVal.getResult());
      ifBodyBuilder.setInsertionPointAfter(reductionLoop);
      // if (targetMemrefOffset.size() == 0) {

      // Write value to target memref
      // TODO_Ashwin the target memref index needs to be correctly constructed
      // TODO_Ashwin this will also be wrong with non-inplace version for
      // multi-class (assumes 1D destination memref)
      memrefStoreIndex.insert(memrefStoreIndex.begin(),
                              zeroIndexConst.getResult());
      ifBodyBuilder.create<memref::StoreOp>(location,
                                            reductionLoop.getResults().front(),
                                            targetMemref, memrefStoreIndex);
      // ifBodyBuilder.create<gpu::PrintfOp>(
      //     location, "Result index = %ld\n",
      //     ValueRange{threadIndexAddOp.getResult()});

      // } else {
      //   // actual index = threadIndex + targetMemrefOffset[0]
      //   std::vector<Value> resultIndex{actualIndex};
      //   // Print the result index
      //   ifBodyBuilder.create<gpu::PrintfOp>(location, "Result index = %ld\n",
      //                                       ValueRange{actualIndex});
      //   ifBodyBuilder.create<memref::StoreOp>(
      //       location, reductionLoop.getResults().front(), targetMemref,
      //       resultIndex);
      // }
    }
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto reduceOp =
        AssertOpIsOfType<decisionforest::CooperativeReduceInplaceOp>(op);
    auto location = reduceOp.getLoc();

    // auto targetMemrefOffsets = reduceOp.getTargetMemrefOffsets();

    // The values for the outer indices. These are fixed

    auto reductionDim = reduceOp.getReductionDimension();
    auto reductionDimVal = GetConstantIntValueFromMLIRValue(reductionDim);
    assert(reductionDimVal == 0);

    auto rangeStart = reduceOp.getPostReductionDimensionStart();
    auto rangeEnd = reduceOp.getPostReductionDimensionEnd();

    // auto sourceMemref = reduceOp.getSourceMemref();
    // auto sourceMemrefType = sourceMemref.getType().cast<MemRefType>();
    auto targetMemref = reduceOp.getTargetMemref();

    // assert(rangeStart.size() == 1);
    assert(rangeStart.size() == rangeEnd.size());
    std::vector<Value> rangeStartVec(rangeStart.begin(), rangeStart.end());
    std::vector<Value> rangeEndVec(rangeEnd.begin(), rangeEnd.end());
    // std::cout << "Range size = "
    //           << getConstantStepBetweenValues(rangeStartVec[0],
    //           rangeEndVec[0])
    //           << "\n";

    auto localThreadIdAndNumThreads =
        generateLocalThreadId(location, rewriter, reduceOp);
    auto localThreadId = std::get<0>(localThreadIdAndNumThreads);
    auto numThreads = std::get<1>(localThreadIdAndNumThreads);
    // auto numThreadsVal = std::get<2>(localThreadIdAndNumThreads);

    rewriter.create<gpu::BarrierOp>(location);

    // if the number of reductions to perform is greater than
    // the number of threads, then use the single thread reduction strategy
    // (Each thread fully performs a single reduction)
    // auto numRowsToProcess =
    //     getConstantStepBetweenValues(rangeStartVec[0], rangeEndVec[0]);

    generateSingleThreadReductionStrategy(
        location, rewriter, targetMemref, localThreadId, numThreads,
        rangeStartVec, rangeEndVec, reductionDimVal, reduceOp);

    // rewriter.setInsertionPointAfter(loop);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct InitializeMemrefLowering : public ConversionPattern {
  InitializeMemrefLowering(MLIRContext *ctx)
      : ConversionPattern(
            mlir::decisionforest::InitializeMemrefOp::getOperationName(),
            1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto initOp = AssertOpIsOfType<decisionforest::InitializeMemrefOp>(op);
    auto location = initOp.getLoc();

    auto gpuLaunchOp = initOp->getParentOfType<gpu::LaunchOp>();

    // threadId = get_thread_id();
    auto threadId =
        decisionforest::GenerateLocalThreadId(rewriter, location, gpuLaunchOp);

    // compute the size of the memref
    auto memref = initOp.getTargetMemref();
    auto memrefType = memref.getType().cast<MemRefType>();
    auto memrefShape = memrefType.getShape();
    auto memrefRank = memrefType.getRank();
    assert(memrefRank >= 1);
    auto memrefSize = memrefShape[0];
    for (auto i = 1; i < memrefRank; ++i) {
      memrefSize *= memrefShape[i];
    }
    // reinterpret cast the memref to a 1D memref of size memref size
    // TODO_Ashwin can we always use an empty map?
    auto memref1DType =
        MemRefType::get({memrefSize}, memrefType.getElementType(), {},
                        memrefType.getMemorySpaceAsInt());
    auto memref1D = rewriter.create<memref::ReinterpretCastOp>(
        location, memref1DType, memref, 0 /*offset*/,
        ArrayRef<int64_t>{memrefSize} /*sizes*/,
        ArrayRef<int64_t>{1} /*strides*/);

    // for i = threadId to memrefSize step numThreads
    auto memrefSizeConst =
        rewriter.create<arith::ConstantIndexOp>(location, memrefSize /*value*/);
    auto numThreads =
        decisionforest::GetNumberOfThreadsInThreadBlock(gpuLaunchOp);
    auto numThreadsConst =
        rewriter.create<arith::ConstantIndexOp>(location, numThreads /*value*/);
    double initialValue = initOp.getInitialValue().convertToDouble();
    auto initialValConst = decisionforest::createFloatConst(
        location, rewriter, memrefType.getElementType(), initialValue);
    auto loop = rewriter.create<scf::ForOp>(location, threadId, memrefSizeConst,
                                            numThreadsConst);
    {
      decisionforest::helpers::SaveAndRestoreInsertionPoint saveIP(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      auto loopIV = loop.getInductionVar();
      rewriter.create<memref::StoreOp>(location, initialValConst, memref1D,
                                       loopIV);
    }

    rewriter.create<gpu::BarrierOp>(location);

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
                           func::FuncDialect, gpu::GPUDialect,
                           vector::VectorDialect>();

    target.addIllegalOp<decisionforest::ReduceOp,
                        decisionforest::ReduceDimensionInplaceOp,
                        decisionforest::ReduceDimensionOp,
                        decisionforest::CooperativeReduceDimensionOp,
                        decisionforest::CooperativeReduceInplaceOp,
                        decisionforest::CooperativeReduceArgMaxOp,
                        decisionforest::AtomicReduceDimensionOp,
                        decisionforest::InitializeMemrefOp>();

    RewritePatternSet patterns(&getContext());
    patterns
        .add<ReduceOpLowering, ReduceInplaceOpLowering,
             ReduceDimensionOpLowering, CooperativeReduceDimensionOpLowering,
             CooperativeInplaceReduceDimensionOpLowering,
             CooperativeArgMaxReduceOpLowering, AtomicReduceDimensionOpLowering,
             InitializeMemrefLowering>(&getContext());

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