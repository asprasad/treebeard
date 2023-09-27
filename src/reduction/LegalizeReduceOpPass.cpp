#include "Dialect.h"
#include <iostream>
#include <mutex>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

#include "LIRLoweringHelpers.h"
#include "Logger.h"
#include "OpLoweringUtils.h"

using namespace mlir;

namespace mlir {
namespace decisionforest {

// Defined in LowerToMidLevelIR.cpp
Value SumOfValues(ConversionPatternRewriter &rewriter, Location location,
                  std::list<Value> &values);

std::vector<scf::ParallelOp>
getConflictingLoops(decisionforest::ReduceOp reduceOp) {
  std::vector<scf::ParallelOp> conflictingLoops;
  auto owningLoop = reduceOp->getParentOfType<scf::ParallelOp>();
  while (owningLoop) {
    if (owningLoop->getAttr("treeLoop")) {
      conflictingLoops.push_back(owningLoop);
    }
    owningLoop = owningLoop->getParentOfType<scf::ParallelOp>();
  }
  return conflictingLoops;
}

Operation *getOutermostTreeLoopOp(decisionforest::ReduceOp reduceOp) {
  auto owningOp = reduceOp->getParentOp();
  Operation *outermostTreeLoopOp = nullptr;
  while (owningOp) {
    if (owningOp->getAttr("treeLoop")) {
      outermostTreeLoopOp = owningOp;
    }
    owningOp = owningOp->getParentOp();
  }
  assert(outermostTreeLoopOp && "Outermost tree loop not found");
  return outermostTreeLoopOp;
}

std::list<Value> getSurroundingBatchLoopIndices(Operation *op) {
  // Find all the indices of batch loops
  std::list<Value> surroundingBatchLoopIndices;
  auto parentOp = op->getParentOp();
  while (parentOp) {
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      if (forOp->getAttr("batchLoop")) {
        auto index = forOp.getInductionVar();
        surroundingBatchLoopIndices.push_back(index);
      }
    }
    if (auto parForOp = dyn_cast<scf::ParallelOp>(parentOp)) {
      if (parForOp->getAttr("batchLoop")) {
        assert(parForOp.getInductionVars().size() == 1);
        for (auto index : parForOp.getInductionVars()) {
          surroundingBatchLoopIndices.push_back(index);
        }
      }
    }
    parentOp = parentOp->getParentOp();
  }
  return surroundingBatchLoopIndices;
}

Value getImmediateParentBatchLoopStep(Operation *op) {
  auto parentOp = op->getParentOp();
  while (parentOp) {
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      if (forOp->getAttr("batchLoop")) {
        return forOp.getStep();
      }
    }
    if (auto parForOp = dyn_cast<scf::ParallelOp>(parentOp)) {
      if (parForOp->getAttr("batchLoop")) {
        assert(parForOp.getStep().size() == 1);
        return parForOp.getStep()[0];
      }
    }
    parentOp = parentOp->getParentOp();
  }
  return Value();
}

void getLoopStepAndIterationCount(scf::ParallelOp parFor, int64_t &step,
                                  int64_t &iterationCount) {
  step = GetConstantIntValueFromMLIRValue(parFor.getStep()[0]);
  auto lowerBound = GetConstantIntValueFromMLIRValue(parFor.getLowerBound()[0]);
  auto upperBound = GetConstantIntValueFromMLIRValue(parFor.getUpperBound()[0]);
  iterationCount = (upperBound - lowerBound) / step;
}

struct ReduceOpLegalizationPattern : public ConversionPattern {
  std::map<void *, Value> &m_privatizationMap;

  void addPrivatization(Value originalBuffer, Value privatizedBuffer) const {
    m_privatizationMap[originalBuffer.getAsOpaquePointer()] = privatizedBuffer;
  }

  bool shouldAtomicallyReduce(scf::ParallelOp loop) const {
    auto attr = loop->getAttr("atomicReduce");
    return static_cast<bool>(attr);
  }

  MemRefType constructPrivatizedBufferType(
      Type origBufferType, Type inputValueType,
      decisionforest::ReduceOp reduceOp,
      const std::vector<scf::ParallelOp> &conflictingLoops) const {
    std::vector<int64_t> privatizedShape;
    auto memrefType = origBufferType.cast<MemRefType>();

    for (auto loop : conflictingLoops) {
      int64_t step, iterationCount;
      getLoopStepAndIterationCount(loop, step, iterationCount);
      privatizedShape.insert(privatizedShape.begin(), iterationCount);
    }
    for (auto dim : memrefType.getShape()) {
      privatizedShape.push_back(dim);
    }
    if (reduceOp.getReductionTypeAttr().getReductionType() ==
        decisionforest::Reduction::kArgMax) {
      privatizedShape.push_back(
          reduceOp->getAttr(decisionforest::getArgMaxLengthAttributeName())
              .cast<IntegerAttr>()
              .getInt());
    }

    assert(inputValueType.isF32() || inputValueType.isF64());
    auto privatizedBufferType =
        MemRefType::get(privatizedShape, inputValueType);

    return privatizedBufferType;
  }

  void initializePrivatizedBuffer(Value allocatedMemref,
                                  MemRefType privatizedBufferType,
                                  ConversionPatternRewriter &rewriter,
                                  Location location) const {
    auto elemType = privatizedBufferType.getElementType();
    Value zero;
    if (elemType.isa<Float32Type>())
      zero = rewriter.create<arith::ConstantFloatOp>(
          location, APFloat((float)0.0), elemType.cast<FloatType>());
    else if (elemType.isa<Float64Type>())
      zero = rewriter.create<arith::ConstantFloatOp>(
          location, APFloat((double)0.0), elemType.cast<FloatType>());
    else
      assert(false && "Unsupported type for privatized buffer");

    rewriter.create<linalg::FillOp>(location, zero, allocatedMemref);
  }

  Value createOrGetPrivatizedBuffer(Value originalBuffer, Value inputValue,
                                    decisionforest::ReduceOp reduceOp,
                                    const std::vector<scf::ParallelOp> &loops,
                                    ConversionPatternRewriter &rewriter) const {
    auto iter = m_privatizationMap.find(originalBuffer.getAsOpaquePointer());
    if (iter != m_privatizationMap.end()) {
      return iter->second;
    }
    decisionforest::helpers::SaveAndRestoreInsertionPoint saveIP(rewriter);

    // Set insertion point to start of owning function
    auto owningFunc = reduceOp->getParentOfType<mlir::func::FuncOp>();
    rewriter.setInsertionPointToStart(&owningFunc.getBody().front());

    auto privatizedBufferType = constructPrivatizedBufferType(
        originalBuffer.getType(), inputValue.getType(), reduceOp, loops);
    auto privatizedBuffer = rewriter.create<memref::AllocaOp>(
        originalBuffer.getLoc(), privatizedBufferType);
    initializePrivatizedBuffer(privatizedBuffer, privatizedBufferType, rewriter,
                               originalBuffer.getLoc());
    addPrivatization(originalBuffer, privatizedBuffer);
    return privatizedBuffer;
  }

  std::tuple<Value, Value> constructPrivatizedIndicesForConflictingLoop(
      ConversionPatternRewriter &rewriter, Location location,
      scf::ParallelOp loop) const {
    auto step = loop.getStep()[0];
    auto lowerBound = loop.getLowerBound()[0];
    auto loopIndex = loop.getInductionVars()[0];

    auto loopIndexMinusLowerBound =
        rewriter.create<arith::SubIOp>(location, loopIndex, lowerBound);
    auto iterationNumber = rewriter.create<arith::DivSIOp>(
        location, loopIndexMinusLowerBound, step);
    auto iterationNumberPlusOne = rewriter.create<arith::AddIOp>(
        location, iterationNumber,
        rewriter.create<arith::ConstantIndexOp>(location, 1));
    return std::make_tuple(iterationNumber, iterationNumberPlusOne);
  }

  void constructPrivatizedIndicesForConflictingLoops(
      ConversionPatternRewriter &rewriter, Location location,
      std::vector<scf::ParallelOp> &conflictingLoops,
      std::vector<Value> &preReductionIndexStart,
      std::vector<Value> &preReductionIndexEnd) const {

    for (auto loop : conflictingLoops) {
      auto range = constructPrivatizedIndicesForConflictingLoop(rewriter,
                                                                location, loop);
      auto iterationNumber = std::get<0>(range);
      auto iterationNumberPlusOne = std::get<1>(range);
      preReductionIndexStart.insert(preReductionIndexStart.begin(),
                                    iterationNumber);
      preReductionIndexEnd.insert(preReductionIndexEnd.begin(),
                                  iterationNumberPlusOne);
    }
  }

  std::vector<Value> constructPrivatizedReductionIndex(
      ConversionPatternRewriter &rewriter,
      std::vector<scf::ParallelOp> &conflictingLoops,
      decisionforest::ReduceOp reduceOp) const {

    std::vector<Value> privatizedReductionIndex, privatizedReductionIndexEnd;
    auto location = reduceOp->getLoc();

    constructPrivatizedIndicesForConflictingLoops(
        rewriter, location, conflictingLoops, privatizedReductionIndex,
        privatizedReductionIndexEnd);

    privatizedReductionIndex.insert(privatizedReductionIndex.end(),
                                    reduceOp.getIndices().begin(),
                                    reduceOp.getIndices().end());

    return privatizedReductionIndex;
  }

  std::tuple<Value, Value>
  constructRangeStartAndRangeEndIndices(ConversionPatternRewriter &rewriter,
                                        Operation *op, Location location,
                                        MemRefType targetMemrefType) const {
    // Find all surrounding batch loops and add their indices to compute the
    // start.
    auto surroundingBatchLoopIndices = getSurroundingBatchLoopIndices(op);

    if (surroundingBatchLoopIndices.size() == 0) {
      auto startIndex = rewriter.create<arith::ConstantIndexOp>(location, 0);
      assert(targetMemrefType.getShape().size() == 1);
      auto endIndex = rewriter.create<arith::ConstantIndexOp>(
          location, targetMemrefType.getShape()[0]);
      return std::make_tuple(startIndex.getResult(), endIndex.getResult());
    }

    auto startIndex =
        SumOfValues(rewriter, location, surroundingBatchLoopIndices);

    // Find the step of the nearest surrounding batch loop and add it to
    // the start to compute the end.
    auto immediateParentStep = getImmediateParentBatchLoopStep(op);
    auto endIndex = rewriter.create<arith::AddIOp>(location, startIndex,
                                                   immediateParentStep);

    return std::make_tuple(startIndex, endIndex.getResult());
  }

  void addReductionDimensionRangesForArgMax(
      ConversionPatternRewriter &rewriter, Location location,
      std::vector<Value> &reductionDimStart,
      std::vector<Value> &reductionDimEnd, int32_t argMaxLength) const {
    reductionDimStart.push_back(
        rewriter.create<arith::ConstantIndexOp>(location, 0));
    reductionDimEnd.push_back(
        rewriter.create<arith::ConstantIndexOp>(location, argMaxLength));
  }

  void
  addReduceDimensionOpForArgMax(decisionforest::ReduceOp reduceOp,
                                Value privatizedBuffer,
                                ConversionPatternRewriter &rewriter) const {
    auto targetMemrefType =
        reduceOp.getTargetMemref().getType().cast<MemRefType>();
    auto argMaxReductionTypeAttr = decisionforest::createReductionTypeAttribute(
        reduceOp.getContext(), decisionforest::Reduction::kArgMax);
    auto argMaxLengthAttrName = decisionforest::getArgMaxLengthAttributeName();
    auto argMaxLengthAttr = reduceOp->getAttr(argMaxLengthAttrName);
    auto privateBufferMemrefType =
        privatizedBuffer.getType().cast<MemRefType>();

    auto outermostTreeLoopOp = getOutermostTreeLoopOp(reduceOp);
    rewriter.setInsertionPointAfter(outermostTreeLoopOp);

    // At this point we should have reduced all the tree walks for the
    // surrounding batch loops (all tree walks for all batches if no surrounding
    // loops exist). Hence we reduce (0,0,0, ... batchStart, batchEnd)
    auto reductionDimConst = rewriter.create<arith::ConstantIndexOp>(
        reduceOp->getLoc(), privateBufferMemrefType.getShape().size() - 1);

    // At this point, we should have reduced everything except last dimensions
    // representing the batch and class weights.
    auto numberOfReducedDims = privateBufferMemrefType.getShape().size() -
                               targetMemrefType.getShape().size() - 1;
    std::vector<Value> preReductionStart, preReductionEnd;
    for (size_t j = 0; j < numberOfReducedDims; ++j) {
      preReductionStart.push_back(
          rewriter.create<arith::ConstantIndexOp>(reduceOp->getLoc(), 0));
      preReductionEnd.push_back(
          rewriter.create<arith::ConstantIndexOp>(reduceOp->getLoc(), 1));
    }

    auto startAndEndTuple = constructRangeStartAndRangeEndIndices(
        rewriter, outermostTreeLoopOp, reduceOp->getLoc(), targetMemrefType);
    Value rangeStart = std::get<0>(startAndEndTuple);
    Value rangeEnd = std::get<1>(startAndEndTuple);
    preReductionStart.push_back(rangeStart);
    preReductionEnd.push_back(rangeEnd);

    // Mapped dimension corresponds to the batch row.
    auto mappedDimension = rewriter.create<arith::ConstantIndexOp>(
        reduceOp->getLoc(), privateBufferMemrefType.getShape().size() - 2);

    auto reduceDimOp = rewriter.create<decisionforest::ReduceDimensionOp>(
        reduceOp->getLoc(), argMaxReductionTypeAttr, reduceOp.getTargetMemref(),
        privatizedBuffer, ValueRange{mappedDimension}, preReductionStart,
        preReductionEnd, reductionDimConst, ValueRange{}, ValueRange{},
        reduceOp.getInitialValueAttr());
    reduceDimOp->setAttr(argMaxLengthAttrName, argMaxLengthAttr);
  }

  ReduceOpLegalizationPattern(MLIRContext *ctx,
                              std::map<void *, Value> &privatizationMap)
      : ConversionPattern(mlir::decisionforest::ReduceOp::getOperationName(),
                          1 /*benefit*/, ctx),
        m_privatizationMap(privatizationMap) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto reduceOp = AssertOpIsOfType<mlir::decisionforest::ReduceOp>(op);
    auto reductionTypeAttr = reduceOp.getReductionTypeAttr();
    if (!reductionTypeAttr) {
      llvm::errs() << "Reduction type attribute not found\n";
      return mlir::failure();
    }
    auto reductionType = reductionTypeAttr.getReductionType();
    if (reductionType == mlir::decisionforest::Reduction::kArgMax &&
        !op->getAttr(mlir::decisionforest::getArgMaxLengthAttributeName())) {
      llvm::errs() << "Argmax reduction requires ArgMaxLength attribute\n";
      return mlir::failure();
    }

    auto location = reduceOp->getLoc();
    auto conflictingLoops = getConflictingLoops(reduceOp);
    assert(conflictingLoops.size() != 0 ||
           reductionType == decisionforest::Reduction::kArgMax);
    auto targetMemrefType =
        reduceOp.getTargetMemref().getType().cast<MemRefType>();
    auto privatizedBuffer = createOrGetPrivatizedBuffer(
        reduceOp.getTargetMemref(), reduceOp.getValue(), reduceOp,
        conflictingLoops, rewriter);
    auto privatizedReductionIndex =
        constructPrivatizedReductionIndex(rewriter, conflictingLoops, reduceOp);

    // If we're not doing argmax, use the same reductionTypeAttribute as the one
    // used in the original reduce op. ArgMax requires the intermediate reduces
    // to be kAdd
    auto newReductionTypeAttr =
        reductionType == decisionforest::Reduction::kArgMax
            ? decisionforest::createReductionTypeAttribute(
                  op->getContext(), decisionforest::Reduction::kAdd)
            : reductionTypeAttr;

    auto legalizedReduce = rewriter.create<decisionforest::ReduceOp>(
        location, newReductionTypeAttr, privatizedBuffer,
        privatizedReductionIndex, reduceOp.getValue(),
        reduceOp.getInitialValueAttr());
    legalizedReduce->setAttr("legalizedReduce", rewriter.getUnitAttr());

    // Add partial reductions at the exit of each of the conflicting loops
    for (size_t i = 0; i < conflictingLoops.size(); ++i) {
      // TODO_Ashwin Still need to figure out if there are any outer batch
      // loops
      bool lastLoop = i == conflictingLoops.size() - 1;
      auto &loop = conflictingLoops[i];

      // if atomic, set the insertion point inside the loop
      if (shouldAtomicallyReduce(loop)) {
        auto &lastOp = loop.getBody()->back();
        if (llvm::dyn_cast<scf::YieldOp>(&lastOp))
          rewriter.setInsertionPoint(&lastOp);
        else
          rewriter.setInsertionPointToEnd(loop.getBody());
      } else
        rewriter.setInsertionPointAfter(loop);

      auto reductionDimNum = (conflictingLoops.size() - 1) - i;
      auto reductionDimConst =
          rewriter.create<arith::ConstantIndexOp>(location, reductionDimNum);

      auto rangeVals = constructRangeStartAndRangeEndIndices(
          rewriter, loop, location, targetMemrefType);
      auto rangeStart = std::get<0>(rangeVals);
      auto rangeEnd = std::get<1>(rangeVals);
      assert(rangeStart && rangeEnd && "Range start and end cannot be null");
      std::vector<Value> preReductionStart, preReductionEnd;
      std::vector<Value> postReductionStart, postReductionEnd;
      if (lastLoop || shouldAtomicallyReduce(loop)) {
        // These dimensions have already been reduced, so we pass (0, 1) for
        // each of them
        std::for_each(
            conflictingLoops.begin(), conflictingLoops.begin() + i,
            [&](auto loop) {
              postReductionStart.push_back(
                  rewriter.create<arith::ConstantIndexOp>(location, 0));
              postReductionEnd.push_back(
                  rewriter.create<arith::ConstantIndexOp>(location, 1));
            });
        postReductionStart.push_back(rangeStart);
        postReductionEnd.push_back(rangeEnd);

        // This is the set of dimensions that correspond to the
        // the dims in the original result memref. (the dimension
        // corresponding to the batch in TB)
        std::vector<Value> mappedDimensions;
        auto privatizedBufferType =
            privatizedBuffer.getType().cast<MemRefType>();
        for (size_t j = conflictingLoops.size();
             j < privatizedBufferType.getShape().size(); ++j) {
          mappedDimensions.push_back(
              rewriter.create<arith::ConstantIndexOp>(location, j));
        }
        if (shouldAtomicallyReduce(loop)) {
          std::vector<Value> preReductionStart, preReductionEnd;
          // These should be (j, j+1) where j is the loop index
          // of each of the surrounding conflicting loops
          std::for_each(
              conflictingLoops.begin() + i + 1, conflictingLoops.end(),
              [&](scf::ParallelOp loop) {
                auto indexVar = loop.getInductionVars()[0];
                postReductionStart.push_back(indexVar);
                auto oneIndexConst =
                    rewriter.create<arith::ConstantIndexOp>(location, 1);
                postReductionEnd.push_back(rewriter.create<arith::AddIOp>(
                    location, indexVar, oneIndexConst));
              });
          auto range = constructPrivatizedIndicesForConflictingLoop(
              rewriter, location, loop);
          // Create an atomic reduce op within the loop
          rewriter.create<decisionforest::AtomicReduceDimensionOp>(
              location, newReductionTypeAttr, reduceOp.getTargetMemref(),
              privatizedBuffer, ValueRange{mappedDimensions},
              ValueRange{preReductionStart}, ValueRange{preReductionEnd},
              reductionDimConst, std::get<0>(range),
              ValueRange{postReductionStart}, ValueRange{postReductionEnd},
              reduceOp.getInitialValueAttr());
          break;
        } else {
          rewriter.create<decisionforest::ReduceDimensionOp>(
              location, newReductionTypeAttr, reduceOp.getTargetMemref(),
              privatizedBuffer, mappedDimensions, ValueRange{}, ValueRange{},
              reductionDimConst, ValueRange{postReductionStart},
              ValueRange{postReductionEnd}, reduceOp.getInitialValueAttr());
        }
      } else {
        std::vector<scf::ParallelOp> surroundingConflictingLoops(
            conflictingLoops.begin() + i + 1, conflictingLoops.end());
        constructPrivatizedIndicesForConflictingLoops(
            rewriter, location, surroundingConflictingLoops, preReductionStart,
            preReductionEnd);

        postReductionStart.push_back(rangeStart);
        postReductionEnd.push_back(rangeEnd);

        if (reductionType == mlir::decisionforest::Reduction::kArgMax) {
          auto argMaxLength =
              op->getAttr(mlir::decisionforest::getArgMaxLengthAttributeName())
                  .cast<IntegerAttr>()
                  .getInt();
          addReductionDimensionRangesForArgMax(rewriter, location,
                                               postReductionStart,
                                               postReductionEnd, argMaxLength);
        }

        rewriter.create<decisionforest::ReduceDimensionInplaceOp>(
            location, newReductionTypeAttr, privatizedBuffer,
            ValueRange{preReductionStart}, ValueRange{preReductionEnd},
            reductionDimConst, ValueRange{postReductionStart},
            ValueRange{postReductionEnd});
      }
    }

    if (reductionType == mlir::decisionforest::Reduction::kArgMax) {
      addReduceDimensionOpForArgMax(reduceOp, privatizedBuffer, rewriter);
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct LegalizeReductions
    : public PassWrapper<LegalizeReductions,
                         OperationPass<mlir::func::FuncOp>> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, tensor::TensorDialect,
                    scf::SCFDialect, decisionforest::DecisionForestDialect,
                    arith::ArithDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() final {
    ConversionTarget target(getContext());
    std::map<void *, Value> privatizationMap;

    target.addLegalDialect<AffineDialect, memref::MemRefDialect,
                           tensor::TensorDialect, scf::SCFDialect,
                           decisionforest::DecisionForestDialect,
                           math::MathDialect, arith::ArithDialect,
                           func::FuncDialect, linalg::LinalgDialect>();

    // target.addIllegalOp<decisionforest::ReduceOp>();
    target.addDynamicallyLegalOp<decisionforest::ReduceOp>(
        [&](decisionforest::ReduceOp op) {
          if (op->getAttr("legalizedReduce"))
            return true;
          // We still need to legalize reduction of argmax even if there aren't
          // any conflicting loops.
          return getConflictingLoops(op).size() == 0 &&
                 op.getReductionTypeAttr().getReductionType() ==
                     decisionforest::Reduction::kAdd;
        });

    RewritePatternSet patterns(&getContext());
    patterns.add<ReduceOpLegalizationPattern>(&getContext(), privatizationMap);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

void legalizeReductions(mlir::MLIRContext &context, mlir::ModuleOp module) {
  // llvm::DebugFlag = true;
  // Lower from high-level IR to mid-level IR
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(std::make_unique<LegalizeReductions>());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Legalizing reductions failed.\n";
  }
  // llvm::DebugFlag = false;
}

void lowerLinalgToLoops(mlir::MLIRContext &context, mlir::ModuleOp module) {
  // llvm::DebugFlag = true;
  // Lower from high-level IR to mid-level IR
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(createConvertLinalgToLoopsPass());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering linalg to loops failed.\n";
  }
  // llvm::DebugFlag = false;
}

} // namespace decisionforest
} // namespace mlir