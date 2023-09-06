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

// Defined in GPURepresentations.cpp
int64_t GetConstantIntValueFromMLIRValue(Value val);
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

std::list<Value> getSurroundingBatchLoopIndices(scf::ParallelOp parFor) {
  // Find all the indices of batch parfors
  std::list<Value> surroundingBatchLoopIndices;
  auto parentOp = parFor->getParentOp();
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

Value getImmediateParentBatchLoopStep(scf::ParallelOp parFor) {
  auto parentOp = parFor->getParentOp();
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

  MemRefType constructPrivatizedBufferType(
      Type origBufferType,
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

    auto privatizedBufferType =
        MemRefType::get(privatizedShape, memrefType.getElementType());

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

  Value createOrGetPrivatizedBuffer(Value originalBuffer,
                                    const std::vector<scf::ParallelOp> &loops,
                                    ConversionPatternRewriter &rewriter) const {
    auto iter = m_privatizationMap.find(originalBuffer.getAsOpaquePointer());
    if (iter != m_privatizationMap.end()) {
      return iter->second;
    }
    decisionforest::helpers::SaveAndRestoreInsertionPoint saveIP(rewriter);

    // Set insertion point to start of owning function
    auto owningFunc = loops[0]->getParentOfType<mlir::func::FuncOp>();
    rewriter.setInsertionPointToStart(&owningFunc.getBody().front());

    auto privatizedBufferType =
        constructPrivatizedBufferType(originalBuffer.getType(), loops);
    auto privatizedBuffer = rewriter.create<memref::AllocaOp>(
        originalBuffer.getLoc(), privatizedBufferType);
    initializePrivatizedBuffer(privatizedBuffer, privatizedBufferType, rewriter,
                               originalBuffer.getLoc());
    addPrivatization(originalBuffer, privatizedBuffer);
    return privatizedBuffer;
  }

  void constructPrivatizedIndicesForConflictingLoops(
      ConversionPatternRewriter &rewriter, Location location,
      std::vector<scf::ParallelOp> &conflictingLoops,
      std::vector<Value> &privatizedReductionIndex) const {

    for (auto loop : conflictingLoops) {
      auto step = loop.getStep()[0];
      auto lowerBound = loop.getLowerBound()[0];
      auto loopIndex = loop.getInductionVars()[0];

      auto loopIndexMinusLowerBound =
          rewriter.create<arith::SubIOp>(location, loopIndex, lowerBound);
      auto iterationNumber = rewriter.create<arith::DivSIOp>(
          location, loopIndexMinusLowerBound, step);
      privatizedReductionIndex.insert(privatizedReductionIndex.begin(),
                                      iterationNumber);
    }
  }

  std::vector<Value> constructPrivatizedReductionIndex(
      ConversionPatternRewriter &rewriter,
      std::vector<scf::ParallelOp> &conflictingLoops,
      decisionforest::ReduceOp reduceOp) const {

    std::vector<Value> privatizedReductionIndex;
    auto location = reduceOp->getLoc();

    constructPrivatizedIndicesForConflictingLoops(
        rewriter, location, conflictingLoops, privatizedReductionIndex);

    privatizedReductionIndex.insert(privatizedReductionIndex.end(),
                                    reduceOp.getIndices().begin(),
                                    reduceOp.getIndices().end());

    return privatizedReductionIndex;
  }

  std::tuple<Value, Value> constructRangeStartAndRangeEndIndices(
      ConversionPatternRewriter &rewriter, scf::ParallelOp conflictingLoop,
      Location location, MemRefType targetMemrefType) const {
    // Find all surrounding batch loops and add their indices to compute the
    // start.
    auto surroundingBatchLoopIndices =
        getSurroundingBatchLoopIndices(conflictingLoop);

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
    auto immediateParentStep = getImmediateParentBatchLoopStep(conflictingLoop);
    auto endIndex = rewriter.create<arith::AddIOp>(location, startIndex,
                                                   immediateParentStep);

    return std::make_tuple(startIndex, endIndex.getResult());
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
    auto location = reduceOp->getLoc();
    auto conflictingLoops = getConflictingLoops(reduceOp);
    assert(conflictingLoops.size() != 0);
    auto targetMemrefType =
        reduceOp.getTargetMemref().getType().cast<MemRefType>();
    auto privatizedBuffer = createOrGetPrivatizedBuffer(
        reduceOp.getTargetMemref(), conflictingLoops, rewriter);
    auto privatizedReductionIndex =
        constructPrivatizedReductionIndex(rewriter, conflictingLoops, reduceOp);
    auto legalizedReduce = rewriter.create<decisionforest::ReduceOp>(
        location, privatizedBuffer, privatizedReductionIndex,
        reduceOp.getValue(), reduceOp.getInitialValueAttr());
    legalizedReduce->setAttr("legalizedReduce", rewriter.getUnitAttr());

    // Add partial reductions at the exit of each of the conflicting loops
    for (size_t i = 0; i < conflictingLoops.size(); ++i) {
      // TODO_Ashwin Still need to figure out if there are any outer batch loops
      bool lastLoop = i == conflictingLoops.size() - 1;
      auto &loop = conflictingLoops[i];

      rewriter.setInsertionPointAfter(loop);
      auto reductionDimNum = (conflictingLoops.size() - 1) - i;
      auto reductionDimConst =
          rewriter.create<arith::ConstantIndexOp>(location, reductionDimNum);

      auto rangeVals = constructRangeStartAndRangeEndIndices(
          rewriter, loop, location, targetMemrefType);
      auto rangeStart = std::get<0>(rangeVals);
      auto rangeEnd = std::get<1>(rangeVals);
      assert(rangeStart && rangeEnd && "Range start and end cannot be null");
      if (lastLoop) {
        std::vector<Value> reducedDimensions;
        for (size_t j = 1; j < conflictingLoops.size(); ++j) {
          reducedDimensions.push_back(
              rewriter.create<arith::ConstantIndexOp>(location, j));
        }

        rewriter.create<decisionforest::ReduceDimensionOp>(
            location, reduceOp.getTargetMemref(), privatizedBuffer,
            reductionDimConst, reducedDimensions, ValueRange{rangeStart},
            ValueRange{rangeEnd}, reduceOp.getInitialValueAttr());

      } else {
        std::vector<Value> indices;
        std::vector<Value> rangeStartIndices, rangeEndIndices;
        for (size_t j = i; j < i; ++j) {
          rangeStartIndices.push_back(
              rewriter.create<arith::ConstantIndexOp>(location, 0));
          rangeEndIndices.push_back(
              rewriter.create<arith::ConstantIndexOp>(location, 1));
        }
        std::vector<scf::ParallelOp> surroundingConflictingLoops(
            conflictingLoops.begin() + i + 1, conflictingLoops.end());
        constructPrivatizedIndicesForConflictingLoops(
            rewriter, location, surroundingConflictingLoops, indices);

        rangeStartIndices.push_back(rangeStart);
        rangeEndIndices.push_back(rangeEnd);

        rewriter.create<decisionforest::ReduceDimensionInplaceOp>(
            location, privatizedBuffer, reductionDimConst, indices,
            rangeStartIndices, rangeEndIndices);
      }
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
          return getConflictingLoops(op).size() == 0;
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