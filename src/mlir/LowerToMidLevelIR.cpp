#include "Dialect.h"
// #include "Passes.h"
#include "OpLoweringUtils.h"

#include "ReductionTypeAttribute.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Target/TargetMachine.h"

using namespace mlir;

namespace mlir {
namespace decisionforest {

Value SumOfValues(ConversionPatternRewriter &rewriter, Location location,
                  std::list<Value> &values) {
  assert(!values.empty());
  if (values.size() == 1) {
    return values.front();
  }
  auto iter = values.begin();
  auto valueType = iter->getType();
  auto lhs = *iter;
  ++iter;
  auto rhs = *iter;
  ++iter;
  auto accumulator =
      rewriter.create<arith::AddIOp>(location, valueType, lhs, rhs);
  for (; iter != values.end(); ++iter) {
    accumulator = rewriter.create<arith::AddIOp>(
        location, valueType, static_cast<Value>(accumulator), *iter);
  }
  return accumulator;
}

} // namespace decisionforest
} // namespace mlir

namespace {

static MemRefType getRowTypeFromArgumentType(MemRefType type) {
  assert(type.hasRank() && "expected only rank shapes");
  return MemRefType::get({type.getShape()[1]}, type.getElementType());
}

void InsertPrintMemrefOp(ConversionPatternRewriter &rewriter, Location location,
                         int32_t kind, int32_t bitWidth, int64_t tileSize,
                         Value memref, Type elementType) {

  /* Search for 'default argument promotions in C' to see why.
   Making all this 64-bit to avoid unecessary noise. */
  if (bitWidth < 64) {
    bitWidth = 64;
  }

  auto tileSizeConst = rewriter.create<arith::ConstantIntOp>(
      location, tileSize, rewriter.getI32Type());
  auto kindConst = rewriter.create<arith::ConstantIntOp>(location, kind,
                                                         rewriter.getI32Type());
  auto bitWidthConst = rewriter.create<arith::ConstantIntOp>(
      location, bitWidth, rewriter.getI32Type());

  std::vector<Value> vectorValues;
  for (int32_t i = 0; i < tileSize; ++i) {
    auto index = rewriter.create<arith::ConstantIndexOp>(location, i);
    auto ithValue = rewriter.create<memref::LoadOp>(
        location, elementType, memref, static_cast<Value>(index));
    if (kind == 0) { // Integer
      auto i64Value = rewriter.create<arith::ExtSIOp>(
          location, rewriter.getI64Type(), static_cast<Value>(ithValue));
      vectorValues.push_back(i64Value);
    } else {
      auto doubleValue = rewriter.create<arith::ExtFOp>(
          location, rewriter.getF64Type(), static_cast<Value>(ithValue));
      vectorValues.push_back(doubleValue);
    }
  }
  rewriter.create<decisionforest::PrintVectorOp>(location, kindConst,
                                                 bitWidthConst, tileSizeConst,
                                                 ValueRange(vectorValues));
}

void InsertPrintElementOp(ConversionPatternRewriter &rewriter,
                          Location location, int32_t kind, int32_t bitWidth,
                          Value value) {
  auto tileSizeConst =
      rewriter.create<arith::ConstantIntOp>(location, 1, rewriter.getI32Type());
  auto kindConst = rewriter.create<arith::ConstantIntOp>(location, kind,
                                                         rewriter.getI32Type());
  auto bitWidthConst = rewriter.create<arith::ConstantIntOp>(
      location, bitWidth, rewriter.getI32Type());

  rewriter.create<decisionforest::PrintVectorOp>(
      location, kindConst, bitWidthConst, tileSizeConst, ValueRange{value});
}

typedef struct {
  bool isMultiClass;

  // Memrefs and Types
  Value treeClassesMemref;
  MemRefType treeClassesMemrefType;

  Value resultMemref;
  MemRefType resultMemrefType;

  Value data;
  MemRefType dataMemrefType;
  Value inputIndexOffset;

  // Indices
  arith::ConstantIndexOp numClassesConst;
  arith::ConstantIndexOp oneIndexConst;
  arith::ConstantIndexOp zeroIndexConst;
  arith::ConstantIndexOp batchSizeConst;
  Value initialValueConst;

  // Decision Forest and Tree Stuff.
  mlir::decisionforest::TreeType treeType;
  mlir::Value forestConst;
  mlir::arith::CmpFPredicateAttr cmpPredicate;
} PredictOpLoweringState;

template <typename LoopType> struct LoopConstructor {
  LoopType m_loop;
  LoopType m_innerLoop;

  ConversionPatternRewriter &m_rewriter;
  PredictOpLoweringState &m_loweringState;
  Value m_oldForestValue;

  Value m_oldDataValue;
  MemRefType m_oldDataMemrefType;
  Value m_oldInputIndexOffset;

  void InsertCacheRowsOpIfNeeded(const decisionforest::IndexVariable &indexVar,
                                 PredictOpLoweringState &loweringState,
                                 Location location,
                                 ConversionPatternRewriter &rewriter,
                                 Value loopIndex, Value step,
                                 std::list<Value> &batchIndexVars) {
    m_oldDataValue = loweringState.data;
    m_oldDataMemrefType = loweringState.dataMemrefType;
    if (indexVar.GetType() !=
        decisionforest::IndexVariable::IndexVariableType::kBatch)
      return;
    if (!indexVar.Cache())
      return;

    batchIndexVars.push_back(loopIndex);

    assert(loopIndex.getType().isa<IndexType>());
    assert(step.getType().isa<IndexType>());

    auto startIndex =
        decisionforest::SumOfValues(rewriter, location, batchIndexVars);
    auto endIndex = rewriter.create<arith::AddIOp>(location, startIndex, step);
    auto inputType = loweringState.dataMemrefType;
    auto numberOfRows = getConstantIntValue(step).value();
    auto cachedType = MemRefType::get(
        llvm::ArrayRef<int64_t>{numberOfRows, inputType.getShape()[1]},
        inputType.getElementType(), {}, // Affine map
        3);                             // Memory space ID

    auto cacheRows = rewriter.create<decisionforest::CacheInputRowsOp>(
        location, cachedType, loweringState.data, startIndex,
        static_cast<Value>(endIndex));
    m_oldInputIndexOffset = loweringState.inputIndexOffset;

    loweringState.inputIndexOffset = startIndex;
    loweringState.data = cacheRows;
    loweringState.dataMemrefType = cachedType;
  }

  void InsertCacheTreeOpIfNeeded(const decisionforest::IndexVariable &indexVar,
                                 PredictOpLoweringState &loweringState,
                                 Location location,
                                 ConversionPatternRewriter &rewriter,
                                 Value loopIndex, Value step,
                                 std::list<Value> &treeIndexVars) {
    m_oldForestValue = loweringState.forestConst;
    if (indexVar.GetType() !=
        decisionforest::IndexVariable::IndexVariableType::kTree)
      return;
    if (!indexVar.Cache())
      return;

    treeIndexVars.push_back(loopIndex);

    assert(loopIndex.getType().isa<IndexType>());
    assert(step.getType().isa<IndexType>());

    auto startIndex =
        decisionforest::SumOfValues(rewriter, location, treeIndexVars);
    auto endIndex = rewriter.create<arith::AddIOp>(location, startIndex, step);
    auto ensembleType = loweringState.forestConst.getType()
                            .cast<decisionforest::TreeEnsembleType>();
    assert(ensembleType.doAllTreesHaveSameType());
    auto cachedEnsembleType = decisionforest::TreeEnsembleType::get(
        ensembleType.getResultType(),
        ensembleType.getNumberOfTrees(), // TODO_Ashwin This needs to be the
                                         // right number
        ensembleType.getRowType(), ensembleType.getReductionType(),
        ensembleType.getTreeType(
            0), // TODO_Ashwin All trees should have the same type!
        true);
    auto cacheTrees = rewriter.create<decisionforest::CacheTreesFromEnsembleOp>(
        location, cachedEnsembleType, loweringState.forestConst, startIndex,
        static_cast<Value>(endIndex));
    loweringState.forestConst = cacheTrees;
  }

  void AddCustomLoopAttributes(LoopType loop,
                               const decisionforest::IndexVariable &indexVar) {
    if (indexVar.GetType() ==
        decisionforest::IndexVariable::IndexVariableType::kBatch) {
      loop->setAttr("batchLoop", m_rewriter.getUnitAttr());
    } else if (indexVar.GetType() ==
               decisionforest::IndexVariable::IndexVariableType::kTree) {
      loop->setAttr("treeLoop", m_rewriter.getUnitAttr());
    } else {
      llvm_unreachable("Unknown index variable type");
    }
  }

  LoopConstructor(
      const std::list<const decisionforest::IndexVariable *> &indexVars,
      PredictOpLoweringState &loweringState, Location location,
      ConversionPatternRewriter &rewriter, ValueRange start, ValueRange end,
      ValueRange steps, std::list<Value> &batchIndexVars,
      std::list<Value> &treeIndexVars)
      : m_rewriter(rewriter), m_loweringState(loweringState) {
    m_loop = rewriter.create<LoopType>(location, start, end, steps);
    AddCustomLoopAttributes(m_loop, *indexVars.front());
    rewriter.setInsertionPointToStart(m_loop.getBody());

    // This function should only be called with a single index variable.
    // For the GPU case (where there are multiple index variables), the
    // constructor below should be used.
    assert(indexVars.size() == 1);
    auto numInductionVars = indexVars.size();

    auto batchLoopIndices = batchIndexVars;
    auto treeLoopIndices = treeIndexVars;
    // AddLoopIndexVariableToValueList(m_loop.getInductionVars()[numInductionVars
    // - 1],
    //                                 *indexVars.back(),
    //                                 batchLoopIndices,
    //                                 treeLoopIndices);
    InsertCacheTreeOpIfNeeded(*indexVars.back(), loweringState, location,
                              rewriter,
                              m_loop.getInductionVars()[numInductionVars - 1],
                              steps[numInductionVars - 1], treeLoopIndices);
    InsertCacheRowsOpIfNeeded(*indexVars.back(), loweringState, location,
                              rewriter,
                              m_loop.getInductionVars()[numInductionVars - 1],
                              steps[numInductionVars - 1], batchLoopIndices);
  }

  LoopConstructor(
      const std::list<const decisionforest::IndexVariable *> &indexVars,
      const std::list<const decisionforest::IndexVariable *> &innerIndexVars,
      PredictOpLoweringState &loweringState, Location location,
      ConversionPatternRewriter &rewriter, ValueRange gridStart,
      ValueRange gridEnd, ValueRange gridSteps, ValueRange blockStart,
      ValueRange blockEnd, ValueRange blockSteps,
      std::list<Value> &batchIndexVars, std::list<Value> &treeIndexVars)

      : m_rewriter(rewriter), m_loweringState(loweringState) {
    // TODO_Ashwin How do we add the treeLoop or batchLoop attributes to
    // multi-index loops?
    auto oldDataValue = loweringState.data;
    auto oldDataMemrefType = loweringState.dataMemrefType;
    auto oldForestValue = loweringState.forestConst;

    auto batchLoopIndices = batchIndexVars;
    auto treeLoopIndices = treeIndexVars;

    m_loop = rewriter.create<LoopType>(location, gridStart, gridEnd, gridSteps);
    rewriter.setInsertionPointToStart(m_loop.getBody());
    // AddLoopIndexVariableToValueList(m_loop.getInductionVars(), indexVars,
    // batchLoopIndices, treeLoopIndices);

    m_innerLoop =
        rewriter.create<LoopType>(location, blockStart, blockEnd, blockSteps);
    rewriter.setInsertionPointToStart(m_innerLoop.getBody());
    // AddLoopIndexVariableToValueList(m_innerLoop.getInductionVars(),
    // innerIndexVars, batchLoopIndices, treeLoopIndices);

    auto outerIndexIter = indexVars.begin();
    for (size_t i = 0; i < indexVars.size(); ++i, ++outerIndexIter) {
      InsertCacheTreeOpIfNeeded(**outerIndexIter, loweringState, location,
                                rewriter, m_loop.getInductionVars()[i],
                                gridSteps[i], treeLoopIndices);
      InsertCacheRowsOpIfNeeded(**outerIndexIter, loweringState, location,
                                rewriter, m_loop.getInductionVars()[i],
                                gridSteps[i], batchLoopIndices);
    }

    auto innerIndexIter = innerIndexVars.begin();
    for (size_t i = 0; i < innerIndexVars.size(); ++i) {
      InsertCacheTreeOpIfNeeded(**innerIndexIter, loweringState, location,
                                rewriter, m_innerLoop.getInductionVars()[i],
                                blockSteps[i], treeLoopIndices);
      InsertCacheRowsOpIfNeeded(**innerIndexIter, loweringState, location,
                                rewriter, m_innerLoop.getInductionVars()[i],
                                blockSteps[i], batchLoopIndices);
    }

    m_oldDataValue = oldDataValue;
    m_oldDataMemrefType = oldDataMemrefType;
    m_oldForestValue = oldForestValue;
  }

  LoopConstructor(const decisionforest::IndexVariable &index,
                  PredictOpLoweringState &loweringState, Location location,
                  ConversionPatternRewriter &rewriter, Value start, Value end,
                  Value step, ValueRange loopArgs,
                  std::list<Value> &batchIndexVars,
                  std::list<Value> &treeIndexVars)
      : m_rewriter(rewriter), m_loweringState(loweringState) {
    m_loop = rewriter.create<LoopType>(location, start, end, step, loopArgs);
    AddCustomLoopAttributes(m_loop, index);
    rewriter.setInsertionPointToStart(m_loop.getBody());

    auto batchLoopIndices = batchIndexVars;
    auto treeLoopIndices = treeIndexVars;
    // AddLoopIndexVariableToValueList(m_loop.getInductionVar(),
    //                                 index,
    //                                 batchLoopIndices,
    //                                 treeLoopIndices);

    InsertCacheTreeOpIfNeeded(index, loweringState, location, rewriter,
                              m_loop.getInductionVar(), step, batchLoopIndices);
    InsertCacheRowsOpIfNeeded(index, loweringState, location, rewriter,
                              m_loop.getInductionVar(), step, treeLoopIndices);
  }

  LoopConstructor(const decisionforest::IndexVariable &index,
                  PredictOpLoweringState &loweringState, Location location,
                  ConversionPatternRewriter &rewriter, Value start, Value end,
                  Value step, std::list<Value> &batchIndexVars,
                  std::list<Value> &treeIndexVars)
      : m_rewriter(rewriter), m_loweringState(loweringState) {
    m_loop = rewriter.create<LoopType>(location, start, end, step);
    AddCustomLoopAttributes(m_loop, index);
    rewriter.setInsertionPointToStart(m_loop.getBody());

    auto batchLoopIndices = batchIndexVars;
    auto treeLoopIndices = treeIndexVars;
    // AddLoopIndexVariableToValueList(m_loop.getInductionVar(),
    //                                 index,
    //                                 batchLoopIndices,
    //                                 treeLoopIndices);

    InsertCacheTreeOpIfNeeded(index, loweringState, location, rewriter,
                              m_loop.getInductionVar(), step, treeLoopIndices);
    InsertCacheRowsOpIfNeeded(index, loweringState, location, rewriter,
                              m_loop.getInductionVar(), step, batchLoopIndices);
  }

  ~LoopConstructor() {
    m_loweringState.forestConst = m_oldForestValue;
    m_loweringState.data = m_oldDataValue;
    m_loweringState.dataMemrefType = m_oldDataMemrefType;
    m_loweringState.inputIndexOffset = m_oldInputIndexOffset;
    m_rewriter.setInsertionPointAfter(m_loop);
  }

  LoopType GetLoop() { return m_loop; }
  LoopType GetInnerLoop() { return m_innerLoop; }
};

void GenerateResultReduction(ConversionPatternRewriter &rewriter,
                             Location location, PredictOpLoweringState &state,
                             Value accumulatedValue, Value rowIndex) {
  auto reductionAttrType =
      decisionforest::ReductionAttrType::get(state.treeType.getContext());
  auto reductionTypeAttribute = decisionforest::ReductionTypeAttribute::get(
      reductionAttrType, decisionforest::Reduction::kAdd);
  rewriter.create<decisionforest::ReduceOp>(
      location, reductionTypeAttribute, state.resultMemref,
      ValueRange{rowIndex}, accumulatedValue);
}

struct PredictForestOpLowering : public ConversionPattern {
  PredictForestOpLowering(MLIRContext *ctx)
      : ConversionPattern(
            mlir::decisionforest::PredictForestOp::getOperationName(),
            1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    mlir::decisionforest::PredictForestOp forestOp =
        llvm::dyn_cast<mlir::decisionforest::PredictForestOp>(op);
    assert(forestOp);
    assert(operands.size() == 2);
    if (!forestOp)
      return mlir::failure();

    auto inputArgument = operands[0];
    int64_t batchSize = 1;
    if (inputArgument.getType().isa<mlir::MemRefType>()) {
      auto memrefType = inputArgument.getType().cast<mlir::MemRefType>();
      if (memrefType.getShape().size() !=
          2) // We can currently only deal with 2D memrefs as inputs
        return mlir::failure();
      batchSize = memrefType.getShape()[0]; // The number of rows in our input
                                            // memref is the batchsize
      return LowerPredictForestOp_Schedule(op, forestOp, operands, rewriter,
                                           memrefType, batchSize);
    } else {
      assert(false && "Lowering for non-tensor argument not implemented");
      return mlir::failure();
    }
  }
  Value GetRow(ConversionPatternRewriter &rewriter, Location location,
               Value data, Value rowIndex, MemRefType dataMemrefType) const {
    auto rowType = getRowTypeFromArgumentType(dataMemrefType);
    auto zeroIndexAttr = rewriter.getIndexAttr(0);
    auto oneIndexAttr = rewriter.getIndexAttr(1);
    auto rowSizeAttr = rewriter.getIndexAttr(rowType.getShape()[0]);
    // rewriter.create<gpu::PrintfOp>(location, "Reading row %d\n",
    // ValueRange{rowIndex});
    auto row = rewriter.create<memref::SubViewOp>(
        location, static_cast<Value>(data),
        ArrayRef<OpFoldResult>({rowIndex, zeroIndexAttr}),
        ArrayRef<OpFoldResult>({oneIndexAttr, rowSizeAttr}),
        ArrayRef<OpFoldResult>({oneIndexAttr, oneIndexAttr}));
    // std::vector<Value> subviewElems;
    // auto zeroIndex =  rewriter.create<arith::ConstantIndexOp>(location, 0);
    // for (int i=0 ; i<5; ++i) {
    //     auto constIndex = rewriter.create<arith::ConstantIndexOp>(location,
    //     i); auto val = rewriter.create<memref::LoadOp>(location,
    //     row.getResult(), ValueRange{zeroIndex.getResult(),
    //     constIndex.getResult()}); subviewElems.push_back(val);
    // }
    // rewriter.create<gpu::PrintfOp>(location, "Row %d: %lf, %lf, %lf, %lf,
    // %lf\n", ValueRange{rowIndex, subviewElems[0], subviewElems[1],
    // subviewElems[2], subviewElems[3], subviewElems[4]});

    if (decisionforest::InsertDebugHelpers) {
      rewriter.create<decisionforest::PrintInputRowOp>(location, row, rowIndex);
    }
    return row;
  }

  void InitPredictOpLoweringState(
      ConversionPatternRewriter &rewriter, Location location,
      PredictOpLoweringState &state,
      mlir::decisionforest::PredictForestOp forestOp, ArrayRef<Value> operands,
      MemRefType dataMemrefType, int64_t batchSize) const {

    auto resultType = forestOp->getResults()[0].getType();
    state.resultMemrefType = resultType.cast<mlir::MemRefType>();
    assert(state.resultMemrefType);
    state.resultMemref = operands[1];

    // Create the decision tree constant
    auto forestAttribute = forestOp.getEnsemble(); // Get the ensemble attribute
    auto forestType = forestAttribute.getType()
                          .cast<mlir::decisionforest::TreeEnsembleType>();
    state.forestConst =
        rewriter.create<mlir::decisionforest::EnsembleConstantOp>(
            location, forestType, forestAttribute);

    state.isMultiClass =
        forestAttribute.GetDecisionForest().IsMultiClassClassifier();
    state.treeType =
        forestType.getTreeType(0).cast<mlir::decisionforest::TreeType>();

    // Initialize constants
    state.batchSizeConst =
        rewriter.create<arith::ConstantIndexOp>(location, batchSize);
    state.zeroIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
    state.oneIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 1);
    state.numClassesConst = rewriter.create<arith::ConstantIndexOp>(
        location, forestAttribute.GetDecisionForest().GetNumClasses());
    auto initialValue = forestAttribute.GetDecisionForest().GetInitialOffset();
    state.initialValueConst = CreateFPConstant(
        rewriter, location, dataMemrefType.getElementType(), initialValue);

    // Initialize members for multi-class classification
    if (state.isMultiClass) {
      state.treeClassesMemrefType = MemRefType::get(
          {batchSize,
           (int64_t)forestAttribute.GetDecisionForest().GetNumClasses()},
          dataMemrefType.getElementType());

      state.treeClassesMemref = rewriter.create<memref::AllocaOp>(
          location, state.treeClassesMemrefType);
    }

    state.data = operands[0];
    state.dataMemrefType = dataMemrefType;
    state.cmpPredicate = forestOp.getPredicateAttr();
  }

  Value GenSigmoid(ConversionPatternRewriter &rewriter, Value operand,
                   Location location) const {
    assert(operand.getType().isIntOrFloat());
    auto negate = rewriter.create<mlir::arith::NegFOp>(
        location, operand.getType(), operand);
    auto exponential = rewriter.create<mlir::math::ExpOp>(
        location, operand.getType(), static_cast<Value>(negate));

    Value oneConst;
    if (operand.getType().isa<mlir::Float64Type>())
      oneConst = rewriter.create<arith::ConstantFloatOp>(
          location, llvm::APFloat(1.0), operand.getType().cast<FloatType>());
    else if (operand.getType().isa<mlir::Float32Type>())
      oneConst = rewriter.create<arith::ConstantFloatOp>(
          location, llvm::APFloat((float)1.0),
          operand.getType().cast<FloatType>());
    else
      assert(false && "Unsupported floating point type");

    auto onePlusExp = rewriter.create<arith::AddFOp>(
        location, operand.getType(), oneConst, exponential);
    auto result = rewriter.create<arith::DivFOp>(location, operand.getType(),
                                                 oneConst, onePlusExp);
    return result;
  }

  Value GenArgMax(ConversionPatternRewriter &rewriter, Location location,
                  PredictOpLoweringState &state, Value index) const {
    auto treeClassesMemref = GetRow(rewriter, location, state.treeClassesMemref,
                                    index, state.treeClassesMemrefType);
    auto treeClassElementType =
        treeClassesMemref.getType().cast<MemRefType>().getElementType();

    if (decisionforest::PrintVectors) {
      InsertPrintMemrefOp(rewriter, location, 0,         /*int kind*/
                          sizeof(float) * 8,             /*bit width*/
                          state.numClassesConst.value(), /* nelts in memref */
                          treeClassesMemref, treeClassElementType);
    }

    auto firstValue = rewriter.create<memref::LoadOp>(
        location, treeClassElementType, static_cast<Value>(treeClassesMemref),
        ValueRange(
            llvm::ArrayRef<Value>{state.zeroIndexConst, state.zeroIndexConst}));

    auto maxClassLoop = rewriter.create<scf::ForOp>(
        location, state.oneIndexConst, state.numClassesConst,
        state.oneIndexConst,
        ValueRange({static_cast<Value>(firstValue),
                    static_cast<Value>(state.zeroIndexConst)}));

    rewriter.setInsertionPointToStart(maxClassLoop.getBody());

    auto k = maxClassLoop.getInductionVar();
    auto currentValue = rewriter.create<memref::LoadOp>(
        location, treeClassElementType, static_cast<Value>(treeClassesMemref),
        ValueRange(llvm::ArrayRef<Value>{state.zeroIndexConst, k}));

    auto maxValue = maxClassLoop.getLoopBody().getArgument(1);
    auto maxIndex = maxClassLoop.getLoopBody().getArgument(2);

    auto compareResult = rewriter.create<arith::CmpFOp>(
        location, mlir::arith::CmpFPredicate::OGT, maxValue, currentValue);

    auto ifElse = rewriter.create<scf::IfOp>(
        location, TypeRange({treeClassElementType, k.getType()}), compareResult,
        true);
    {
      auto thenBodyBuilder = ifElse.getThenBodyBuilder();
      thenBodyBuilder.create<scf::YieldOp>(location,
                                           ValueRange({maxValue, maxIndex}));
    }
    {
      auto elseBodyBuilder = ifElse.getElseBodyBuilder();
      elseBodyBuilder.create<scf::YieldOp>(
          location, ValueRange({static_cast<Value>(currentValue),
                                static_cast<Value>(k)}));
    }

    rewriter.create<scf::YieldOp>(location, ifElse.getResults());

    rewriter.setInsertionPointAfter(maxClassLoop);

    if (decisionforest::PrintVectors) {
      InsertPrintElementOp(rewriter, location, 1, sizeof(int64_t) * 8,
                           maxClassLoop.getResult(1));
    }

    return rewriter.create<arith::IndexCastOp>(
        location, state.treeType.getResultType(),
        static_cast<Value>(maxClassLoop.getResult(1)));
  }

  Value CreateFPConstant(ConversionPatternRewriter &rewriter, Location location,
                         Type type, double value) const {
    Value constValue;
    if (type.isa<mlir::Float64Type>())
      constValue = rewriter.create<arith::ConstantFloatOp>(
          location, llvm::APFloat(value), type.cast<FloatType>());
    else if (type.isa<mlir::Float32Type>())
      constValue = rewriter.create<arith::ConstantFloatOp>(
          location, llvm::APFloat((float)value), type.cast<FloatType>());
    else if (type.isa<mlir::IntegerType>())
      constValue = rewriter.create<arith::ConstantIntOp>(location, 0, type);
    else
      assert(false && "Unsupported floating point type");
    return constValue;
  }

  void InitializeTreeClassWeightsMemref(ConversionPatternRewriter &rewriter,
                                        Location location,
                                        PredictOpLoweringState &state) const {
    if (state.isMultiClass) {
      auto outerLoop = rewriter.create<scf::ForOp>(
          location, state.zeroIndexConst, state.batchSizeConst,
          state.oneIndexConst);
      rewriter.setInsertionPointToStart(outerLoop.getBody());
      {
        auto i = outerLoop.getInductionVar();
        auto row = GetRow(rewriter, location, state.treeClassesMemref, i,
                          state.treeClassesMemrefType);
        auto innerLoop = rewriter.create<scf::ForOp>(
            location, state.zeroIndexConst, state.numClassesConst,
            state.oneIndexConst);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        {
          auto j = innerLoop.getInductionVar();

          rewriter.create<memref::StoreOp>(
              location, static_cast<Value>(state.initialValueConst),
              static_cast<Value>(row),
              ValueRange(llvm::ArrayRef<Value>{state.zeroIndexConst, j}));
          rewriter.setInsertionPointAfter(innerLoop);
        }
      }
      rewriter.setInsertionPointAfter(outerLoop);
    }
  }

  void InitializeResultMemref(ConversionPatternRewriter &rewriter,
                              Location location,
                              PredictOpLoweringState &state) const {

    // We don't accumulate into result memref in case of multi-class.
    if (state.isMultiClass)
      return;

    // Create a for loop over the outputs
    auto batchLoop =
        rewriter.create<scf::ForOp>(location, state.zeroIndexConst,
                                    state.batchSizeConst, state.oneIndexConst);

    rewriter.setInsertionPointToStart(batchLoop.getBody());
    auto i = batchLoop.getInductionVar();
    rewriter.create<memref::StoreOp>(location, state.initialValueConst,
                                     state.resultMemref, i);
    rewriter.setInsertionPointAfter(batchLoop);
  }

  void
  TransformResultMemref(ConversionPatternRewriter &rewriter, Location location,
                        decisionforest::PredictionTransformation predTransform,
                        PredictOpLoweringState &state) const {

    if (predTransform == decisionforest::PredictionTransformation::kIdentity)
      return;

    // assert (resultMemrefType.getElementType().isa<mlir::FloatType>());
    // assert (predTransform ==
    // decisionforest::PredictionTransformation::kSigmoid);

    auto batchLoop =
        rewriter.create<scf::ForOp>(location, state.zeroIndexConst,
                                    state.batchSizeConst, state.oneIndexConst);

    rewriter.setInsertionPointToStart(batchLoop.getBody());
    auto i = batchLoop.getInductionVar();

    auto memrefElem =
        rewriter.create<memref::LoadOp>(location, state.resultMemref, i);
    Value transformedValue;
    if (predTransform == decisionforest::PredictionTransformation::kSigmoid)
      transformedValue =
          GenSigmoid(rewriter, static_cast<Value>(memrefElem), location);
    else if (predTransform ==
             decisionforest::PredictionTransformation::kSoftMax)
      transformedValue = GenArgMax(rewriter, location, state, i);
    else
      assert(false && "Unsupported prediction transformation.");

    rewriter.create<memref::StoreOp>(location, transformedValue,
                                     state.resultMemref, i);
    rewriter.setInsertionPointAfter(batchLoop);
  }

  void GenerateMultiClassAccumulate(ConversionPatternRewriter &rewriter,
                                    Location location, Value result,
                                    Value rowIndex, Value index,
                                    PredictOpLoweringState &state) const {
    if (state.isMultiClass) {
      auto classId = rewriter.create<decisionforest::GetTreeClassIdOp>(
          location, state.treeType.getResultType(), state.forestConst, index);
      auto classIdIndex = rewriter.create<arith::IndexCastOp>(
          location, rewriter.getIndexType(), static_cast<Value>(classId));

      auto reductionAttrType =
          decisionforest::ReductionAttrType::get(state.treeType.getContext());
      auto reductionTypeAttribute = decisionforest::ReductionTypeAttribute::get(
          reductionAttrType, decisionforest::Reduction::kSoftMax);

      rewriter.create<decisionforest::ReduceOp>(
          location, reductionTypeAttribute, state.treeClassesMemref,
          ValueRange{rowIndex, classIdIndex.getResult()}, result);
    }
  }

  Value GenerateTreeIndexLeafLoopBody(
      ConversionPatternRewriter &rewriter, Location location,
      const decisionforest::IndexVariable &indexVar,
      std::list<Value> treeIndices, PredictOpLoweringState &state, Value row,
      Value rowIndex, Value prevAccumulatorValue) const {

    Value treeIndex =
        decisionforest::SumOfValues(rewriter, location, treeIndices);

    // Get the current tree
    auto forestType =
        state.forestConst.getType().cast<decisionforest::TreeEnsembleType>();
    assert(forestType
               .doAllTreesHaveSameTileSize()); // TODO how do we check which
                                               // type of tree we'll get here?
    auto treeType =
        forestType.getTreeType(0).cast<mlir::decisionforest::TreeType>();
    auto tree = rewriter.create<decisionforest::GetTreeFromEnsembleOp>(
        location, treeType, state.forestConst, treeIndex);

    // Walk the tree
    Value walkOp;
    if (indexVar.PeelWalk()) {
      auto peelItersAttrib =
          rewriter.getI32IntegerAttr(indexVar.IterationsToPeel());
      walkOp = rewriter.create<decisionforest::WalkDecisionTreePeeledOp>(
          location, treeType.getThresholdType(), state.cmpPredicate, tree, row,
          peelItersAttrib);
    } else {
      // auto indexConst = rewriter.create<arith::ConstantIndexOp>(location,
      // (int64_t)1); walkOp = rewriter.create<memref::LoadOp>()
      auto walkUnrollAttr =
          rewriter.getI64IntegerAttr(indexVar.GetTreeWalkUnrollFactor());
      walkOp = rewriter.create<decisionforest::WalkDecisionTreeOp>(
          location, treeType.getThresholdType(), state.cmpPredicate,
          walkUnrollAttr, tree, row);
      // auto printResult = rewriter.create<gpu::PrintfOp>(location, "Result
      // [%d]: %lf\t", ValueRange{rowIndex, static_cast<Value>(walkOp)});
    }
    GenerateMultiClassAccumulate(rewriter, location, static_cast<Value>(walkOp),
                                 rowIndex, treeIndex, state);

    if (state.isMultiClass)
      return prevAccumulatorValue;

    // Accumulate the tree prediction
    assert(forestType.getReductionType() ==
           decisionforest::ReductionType::kAdd);
    auto accumulatedValue = rewriter.create<arith::AddFOp>(
        location, state.resultMemrefType.getElementType(), prevAccumulatorValue,
        walkOp);

    if (mlir::decisionforest::InsertDebugHelpers) {
      Value treePred = walkOp;
      if (!treePred.getType().isF64())
        treePred = rewriter.create<arith::ExtFOp>(
            location, rewriter.getF64Type(), walkOp);
      rewriter.create<decisionforest::PrintTreePredictionOp>(location, treePred,
                                                             treeIndex);
    }
    // auto updatedResultTensor = rewriter.create<tensor::InsertOp>(location,
    // resultMemrefType, accumulatedValue,
    // treeLoop.getBody()->getArguments()[1], i);
    return accumulatedValue;
  }

  Value GeneratePipelinedTreeIndexLeafLoopBody(
      ConversionPatternRewriter &rewriter, Location location, int32_t stepSize,
      std::list<Value> treeIndices, PredictOpLoweringState &state, Value row,
      Value rowIndex, Value prevAccumulatorValue) const {

    // Get the current tree
    auto forestType =
        state.forestConst.getType().cast<decisionforest::TreeEnsembleType>();
    assert(forestType
               .doAllTreesHaveSameTileSize()); // TODO how do we check which
                                               // type of tree we'll get here?
    auto treeType =
        forestType.getTreeType(0).cast<mlir::decisionforest::TreeType>();

    std::vector<Value> finalTreeIndices;
    std::vector<Value> rows;
    std::vector<Value> trees;
    std::vector<Type> treeResultTypes;
    for (int32_t i = 0; i < stepSize; i++) {
      treeIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(location, i));

      Value treeIndex =
          decisionforest::SumOfValues(rewriter, location, treeIndices);
      auto tree = rewriter.create<decisionforest::GetTreeFromEnsembleOp>(
          location, treeType, state.forestConst, treeIndex);
      finalTreeIndices.push_back(treeIndex);
      trees.push_back(tree);
      treeResultTypes.push_back(treeType.getThresholdType());
      rows.push_back(row);

      treeIndices.pop_back();
    }

    // Walk the tree.
    auto unrollLoopAttr =
        decisionforest::UnrollLoopAttribute::get(treeType, -1);
    auto walkOp = rewriter.create<decisionforest::PipelinedWalkDecisionTreeOp>(
        location, treeResultTypes, unrollLoopAttr, state.cmpPredicate, trees,
        rows);

    for (size_t i = 0; i < trees.size(); i++) {
      if (state.isMultiClass) {
        GenerateMultiClassAccumulate(rewriter, location, walkOp.getResult(i),
                                     rowIndex, finalTreeIndices[i], state);
      } else {
        // Accumulate the tree prediction
        assert(forestType.getReductionType() ==
               decisionforest::ReductionType::kAdd);
        prevAccumulatorValue = rewriter.create<arith::AddFOp>(
            location, state.resultMemrefType.getElementType(),
            prevAccumulatorValue, walkOp.getResult(i));
      }

      if (mlir::decisionforest::InsertDebugHelpers) {
        rewriter.create<decisionforest::PrintTreePredictionOp>(
            location, walkOp.getResult(i), finalTreeIndices[i]);
      }
      // auto updatedResultTensor = rewriter.create<tensor::InsertOp>(location,
      // resultMemrefType, accumulatedValue,
      // treeLoop.getBody()->getArguments()[1], i);
    }

    return prevAccumulatorValue;
  }

  void GenerateLeafLoopForTreeIndex(
      ConversionPatternRewriter &rewriter, Location location,
      const decisionforest::IndexVariable &indexVar,
      std::list<Value> batchIndices, std::list<Value> treeIndices,
      PredictOpLoweringState &state) const {

    assert(indexVar.GetType() ==
           decisionforest::IndexVariable::IndexVariableType::kTree);

    Value rowIndex =
        decisionforest::SumOfValues(rewriter, location, batchIndices);
    Value rowIndexForRowRead = rowIndex;
    if (state.inputIndexOffset)
      rowIndexForRowRead = rewriter.create<arith::SubIOp>(
          location, rowIndex, state.inputIndexOffset);

    // Get the current row
    Value row = GetRow(rewriter, location, state.data, rowIndexForRowRead,
                       state.dataMemrefType);

    if (indexVar.Unroll()) {
      auto range = indexVar.GetRange();
      auto zeroConst = CreateFPConstant(
          rewriter, location, state.dataMemrefType.getElementType(), 0.0);
      Value accumulatedValue = zeroConst;
      for (int32_t i = range.m_start; i < range.m_stop; i += range.m_step) {
        auto treeIndex = rewriter.create<arith::ConstantIndexOp>(location, i);
        treeIndices.push_back(treeIndex);
        accumulatedValue = GenerateTreeIndexLeafLoopBody(
            rewriter, location, indexVar, treeIndices, state, row, rowIndex,
            accumulatedValue);
      }

      // Don't accumulate into memref in case of multiclass.
      if (state.isMultiClass)
        return;

      // Generate the store back in to the result memref
      GenerateResultReduction(rewriter, location, state, accumulatedValue,
                              rowIndex);
    } else if (indexVar.Pipelined()) {
      auto range = indexVar.GetRange();
      int32_t peeledLoopStart =
          range.m_stop - (range.m_stop - range.m_start) % range.m_step;
      int32_t peeledLoopStep = range.m_stop - peeledLoopStart;

      auto stopConst =
          rewriter.create<arith::ConstantIndexOp>(location, peeledLoopStart);
      auto startConst =
          rewriter.create<arith::ConstantIndexOp>(location, range.m_start);
      auto stepConst =
          rewriter.create<arith::ConstantIndexOp>(location, range.m_step);

      auto zeroConst = CreateFPConstant(
          rewriter, location, state.dataMemrefType.getElementType(), 0.0);

      scf::ForOp loop;
      {
        LoopConstructor<scf::ForOp> loopConstructor(
            indexVar, state, location, rewriter, startConst, stopConst,
            stepConst, ValueRange{zeroConst}, batchIndices, treeIndices);
        loop = loopConstructor.GetLoop();
        treeIndices.push_back(loop.getInductionVar());
        auto accumulatedValue = GeneratePipelinedTreeIndexLeafLoopBody(
            rewriter, location, range.m_step, treeIndices, state, row, rowIndex,
            loop.getBody()->getArguments()[1]);
        rewriter.create<scf::YieldOp>(location,
                                      static_cast<Value>(accumulatedValue));
      }

      // Don't accumulate into memref in case of multiclass.
      if (!state.isMultiClass) {
        // Generate the store back in to the result memref
        auto accumulatedValue = loop.getResults()[0];
        GenerateResultReduction(rewriter, location, state, accumulatedValue,
                                rowIndex);
      }

      if (peeledLoopStart < range.m_stop) {
        stopConst =
            rewriter.create<arith::ConstantIndexOp>(location, range.m_stop);
        startConst =
            rewriter.create<arith::ConstantIndexOp>(location, peeledLoopStart);
        stepConst =
            rewriter.create<arith::ConstantIndexOp>(location, peeledLoopStep);

        scf::ForOp peeledLoop;
        {
          LoopConstructor<scf::ForOp> peeledLoopConstructor(
              indexVar, state, location, rewriter, startConst, stopConst,
              stepConst, ValueRange{zeroConst}, batchIndices, treeIndices);
          peeledLoop = peeledLoopConstructor.GetLoop();

          treeIndices.pop_back();
          treeIndices.push_back(peeledLoop.getInductionVar());

          auto accumulatedValue = GeneratePipelinedTreeIndexLeafLoopBody(
              rewriter, location, peeledLoopStep, treeIndices, state, row,
              rowIndex, peeledLoop.getBody()->getArguments()[1]);
          rewriter.create<scf::YieldOp>(location,
                                        static_cast<Value>(accumulatedValue));
        }

        if (!state.isMultiClass) {
          Value accumulatedValue = peeledLoop.getResults()[0];
          GenerateResultReduction(rewriter, location, state, accumulatedValue,
                                  rowIndex);
        }
      }
    } else {
      // Generate leaf loop for tree index var
      auto range = indexVar.GetRange();
      auto stopConst =
          rewriter.create<arith::ConstantIndexOp>(location, range.m_stop);
      auto startConst =
          rewriter.create<arith::ConstantIndexOp>(location, range.m_start);
      auto stepConst =
          rewriter.create<arith::ConstantIndexOp>(location, range.m_step);

      auto zeroConst = CreateFPConstant(
          rewriter, location, state.dataMemrefType.getElementType(), 0.0);

      Value loopResult;
      {
        LoopConstructor<scf::ForOp> loopConstructor(
            indexVar, state, location, rewriter, startConst, stopConst,
            stepConst, ValueRange{zeroConst}, batchIndices, treeIndices);
        scf::ForOp loop = loopConstructor.GetLoop();
        rewriter.setInsertionPointToStart(loop.getBody());
        treeIndices.push_back(loop.getInductionVar());
        auto accumulatedValue = GenerateTreeIndexLeafLoopBody(
            rewriter, location, indexVar, treeIndices, state, row, rowIndex,
            loop.getBody()->getArguments()[1]);
        rewriter.create<scf::YieldOp>(location,
                                      static_cast<Value>(accumulatedValue));
        // rewriter.setInsertionPointAfter(loop);
        loopResult = loop.getResult(0);
      }

      // Don't accumulate into memref in case of multiclass.
      if (state.isMultiClass)
        return;

      // Generate the store back in to the result memref
      GenerateResultReduction(rewriter, location, state, loopResult, rowIndex);

      // rewriter.create<gpu::PrintfOp>(location, "Writing result[%d] = %lf +
      // %lf: %lf\n", ValueRange{rowIndex, currentMemrefElem,
      // loop.getResults()[0], newMemrefElem});
    }
  }

  void GeneratePipelinedBatchIndexLeafLoopBody(
      ConversionPatternRewriter &rewriter, Location location,
      const decisionforest::IndexVariable &indexVar, int32_t stepSize,
      std::list<Value> batchIndices, decisionforest::TreeType treeType,
      Value tree, Value treeIndex, PredictOpLoweringState &state) const {

    std::vector<Value> rowIndices;
    std::vector<Value> rows;
    std::vector<Value> trees;
    std::vector<Type> treeResultTypes;
    for (int32_t i = 0; i < stepSize; i++) {
      batchIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(location, i));
      auto rowIndex =
          decisionforest::SumOfValues(rewriter, location, batchIndices);
      Value rowIndexForRowRead = rowIndex;
      if (state.inputIndexOffset)
        rowIndexForRowRead = rewriter.create<arith::SubIOp>(
            location, rowIndex, state.inputIndexOffset);

      rows.push_back(GetRow(rewriter, location, state.data, rowIndexForRowRead,
                            state.dataMemrefType));
      rowIndices.push_back(rowIndex);
      trees.push_back(tree);
      treeResultTypes.push_back(treeType.getThresholdType());

      batchIndices.pop_back();
    }

    // Walk the tree
    auto unrollLoopAttr = decisionforest::UnrollLoopAttribute::get(
        treeType, indexVar.GetContainingLoop()->GetTreeWalkUnrollFactor());
    auto walkOp = rewriter.create<decisionforest::PipelinedWalkDecisionTreeOp>(
        location, treeResultTypes, unrollLoopAttr, state.cmpPredicate, trees,
        rows);
    for (size_t i = 0; i < rowIndices.size(); i++) {
      // Don't accumulate into memref in case of multiclass.
      if (state.isMultiClass) {
        GenerateMultiClassAccumulate(rewriter, location, walkOp.getResult(i),
                                     rowIndices[i], treeIndex, state);
      } else {
        // Accumulate the tree prediction and generate the store back in to the
        // result memref
        GenerateResultReduction(rewriter, location, state, walkOp.getResult(i),
                                rowIndices[i]);
        if (mlir::decisionforest::InsertDebugHelpers) {
          rewriter.create<decisionforest::PrintTreePredictionOp>(
              location, walkOp.getResult(i), treeIndex);
        }
      }
    }
  }

  void GenerateBatchIndexLeafLoopBody(
      ConversionPatternRewriter &rewriter, Location location,
      const decisionforest::IndexVariable &indexVar,
      std::list<Value> batchIndices, decisionforest::TreeType treeType,
      Value tree, Value treeIndex, PredictOpLoweringState &state) const {
    Value rowIndex =
        decisionforest::SumOfValues(rewriter, location, batchIndices);
    Value rowIndexForRowRead = rowIndex;
    if (state.inputIndexOffset)
      rowIndexForRowRead = rewriter.create<arith::SubIOp>(
          location, rowIndex, state.inputIndexOffset);

    // Get the current row
    Value row = GetRow(rewriter, location, state.data, rowIndexForRowRead,
                       state.dataMemrefType);

    // Walk the tree
    Value walkOp;
    if (indexVar.PeelWalk()) {
      auto peelItersAttrib =
          rewriter.getI32IntegerAttr(indexVar.IterationsToPeel());
      walkOp = rewriter.create<decisionforest::WalkDecisionTreePeeledOp>(
          location, treeType.getThresholdType(), state.cmpPredicate, tree, row,
          peelItersAttrib);
    } else {
      auto walkUnrollAttr =
          rewriter.getI64IntegerAttr(indexVar.GetTreeWalkUnrollFactor());
      walkOp = rewriter.create<decisionforest::WalkDecisionTreeOp>(
          location, treeType.getThresholdType(), state.cmpPredicate,
          walkUnrollAttr, tree, row);
      // walkOp = rewriter.create<arith::ConstantFloatOp>(location,
      // APFloat((double)0), treeType.getThresholdType().cast<FloatType>());
    }

    GenerateMultiClassAccumulate(rewriter, location, static_cast<Value>(walkOp),
                                 rowIndex, treeIndex, state);

    // Don't accumulate into memref in case of multiclass.
    if (state.isMultiClass)
      return;

    // Accumulate the tree prediction and generate the store back in to the
    // result memref
    GenerateResultReduction(rewriter, location, state, walkOp, rowIndex);
    if (mlir::decisionforest::InsertDebugHelpers) {
      Value treePred = walkOp;
      if (!treePred.getType().isF64())
        treePred = rewriter.create<arith::ExtFOp>(
            location, rewriter.getF64Type(), walkOp);
      rewriter.create<decisionforest::PrintTreePredictionOp>(location, treePred,
                                                             treeIndex);
    }
  }

  void GenerateLeafLoopForBatchIndex(
      ConversionPatternRewriter &rewriter, Location location,
      const decisionforest::IndexVariable &indexVar,
      std::list<Value> batchIndices, std::list<Value> treeIndices,
      PredictOpLoweringState &state) const {

    assert(indexVar.GetType() ==
           decisionforest::IndexVariable::IndexVariableType::kBatch);

    Value treeIndex =
        decisionforest::SumOfValues(rewriter, location, treeIndices);

    // Get the current tree
    auto forestType =
        state.forestConst.getType().cast<decisionforest::TreeEnsembleType>();
    assert(forestType
               .doAllTreesHaveSameTileSize()); // TODO how do we check which
                                               // type of tree we'll get here?
    assert(forestType.getReductionType() ==
           decisionforest::ReductionType::kAdd);
    auto treeType =
        forestType.getTreeType(0).cast<mlir::decisionforest::TreeType>();
    auto tree = rewriter.create<decisionforest::GetTreeFromEnsembleOp>(
        location, treeType, state.forestConst, treeIndex);

    if (indexVar.Unroll()) {
      auto range = indexVar.GetRange();
      for (int32_t i = range.m_start; i < range.m_stop; i += range.m_step) {
        auto batchIndex = rewriter.create<arith::ConstantIndexOp>(location, i);
        batchIndices.push_back(batchIndex);
        GenerateBatchIndexLeafLoopBody(rewriter, location, indexVar,
                                       batchIndices, treeType, tree, treeIndex,
                                       state);
      }
    } else if (indexVar.Pipelined()) {
      // Currently supports only single variable.
      auto range = indexVar.GetRange();

      int32_t peeledLoopStart =
          range.m_stop - (range.m_stop - range.m_start) % range.m_step;
      int32_t peeledLoopStep = range.m_stop - peeledLoopStart;

      auto stopConst =
          rewriter.create<arith::ConstantIndexOp>(location, peeledLoopStart);
      auto startConst =
          rewriter.create<arith::ConstantIndexOp>(location, range.m_start);
      auto stepConst =
          rewriter.create<arith::ConstantIndexOp>(location, range.m_step);

      {
        LoopConstructor<scf::ForOp> loopConstructor(
            indexVar, state, location, rewriter, startConst, stopConst,
            stepConst, batchIndices, treeIndices);
        auto loop = loopConstructor.GetLoop();
        batchIndices.push_back(loop.getInductionVar());
        GeneratePipelinedBatchIndexLeafLoopBody(
            rewriter, location, indexVar, range.m_step, batchIndices, treeType,
            tree, treeIndex, state);
      }

      if (peeledLoopStart < range.m_stop) {
        stopConst =
            rewriter.create<arith::ConstantIndexOp>(location, range.m_stop);
        startConst =
            rewriter.create<arith::ConstantIndexOp>(location, peeledLoopStart);
        stepConst =
            rewriter.create<arith::ConstantIndexOp>(location, peeledLoopStep);

        LoopConstructor<scf::ForOp> peeledLoopConstructor(
            indexVar, state, location, rewriter, startConst, stopConst,
            stepConst, batchIndices, treeIndices);
        auto peeledLoop = peeledLoopConstructor.GetLoop();

        batchIndices.pop_back();
        batchIndices.push_back(peeledLoop.getInductionVar());

        GeneratePipelinedBatchIndexLeafLoopBody(
            rewriter, location, indexVar, peeledLoopStep, batchIndices,
            treeType, tree, treeIndex, state);
      }
    } else {
      // Generate leaf loop for tree index var
      auto range = indexVar.GetRange();
      auto stopConst =
          rewriter.create<arith::ConstantIndexOp>(location, range.m_stop);
      auto startConst =
          rewriter.create<arith::ConstantIndexOp>(location, range.m_start);
      auto stepConst =
          rewriter.create<arith::ConstantIndexOp>(location, range.m_step);

      LoopConstructor<scf::ForOp> loopConstructor(
          indexVar, state, location, rewriter, startConst, stopConst, stepConst,
          batchIndices, treeIndices);
      scf::ForOp loop = loopConstructor.GetLoop();
      rewriter.setInsertionPointToStart(loop.getBody());
      batchIndices.push_back(loop.getInductionVar());
      GenerateBatchIndexLeafLoopBody(rewriter, location, indexVar, batchIndices,
                                     treeType, tree, treeIndex, state);
      rewriter.setInsertionPointAfter(loop);
    }
  }

  void GenerateLeafLoop(ConversionPatternRewriter &rewriter, Location location,
                        const decisionforest::IndexVariable &indexVar,
                        std::list<Value> batchIndices,
                        std::list<Value> treeIndices,
                        PredictOpLoweringState &state) const {
    if (indexVar.GetType() ==
        decisionforest::IndexVariable::IndexVariableType::kTree) {
      GenerateLeafLoopForTreeIndex(rewriter, location, indexVar, batchIndices,
                                   treeIndices, state);
    } else {
      GenerateLeafLoopForBatchIndex(rewriter, location, indexVar, batchIndices,
                                    treeIndices, state);
    }
  }

  void GenerateUnrolledLoop(ConversionPatternRewriter &rewriter,
                            Location location,
                            const decisionforest::IndexVariable &indexVar,
                            std::list<Value> batchIndices,
                            std::list<Value> treeIndices,
                            PredictOpLoweringState &state) const {
    auto range = indexVar.GetRange();

    for (int32_t i = range.m_start; i < range.m_stop; ++i) {
      auto indexVal = rewriter.create<arith::ConstantIndexOp>(location, i);

      if (indexVar.GetType() ==
          decisionforest::IndexVariable::IndexVariableType::kBatch)
        batchIndices.push_back(indexVal);
      else if (indexVar.GetType() ==
               decisionforest::IndexVariable::IndexVariableType::kTree)
        treeIndices.push_back(indexVal);
      else
        assert(false && "Unknown index variable type!");

      for (auto nestedIndexVar : indexVar.GetContainedLoops()) {
        GenerateLoop(rewriter, location, *nestedIndexVar, batchIndices,
                     treeIndices, state);
      }

      if (indexVar.GetType() ==
          decisionforest::IndexVariable::IndexVariableType::kBatch)
        batchIndices.pop_back();
      else if (indexVar.GetType() ==
               decisionforest::IndexVariable::IndexVariableType::kTree)
        treeIndices.pop_back();
      else
        assert(false && "Unknown index variable type!");
    }
  }

  bool isGPULoop(const decisionforest::IndexVariable &indexVar) const {
    return indexVar.GetGPUDimension().construct !=
           decisionforest::IndexVariable::GPUConstruct::None;
  }

  gpu::Processor getGPUProcessorFromIndexVar(
      const decisionforest::IndexVariable &indexVar) const {
    auto gpuDim = indexVar.GetGPUDimension();
    if (gpuDim.construct == decisionforest::IndexVariable::GPUConstruct::Grid) {
      if (gpuDim.dimension == decisionforest::IndexVariable::Dimension::X)
        return gpu::Processor::BlockX;
      else if (gpuDim.dimension == decisionforest::IndexVariable::Dimension::Y)
        return gpu::Processor::BlockY;
      else if (gpuDim.dimension == decisionforest::IndexVariable::Dimension::Z)
        return gpu::Processor::BlockZ;
    } else if (gpuDim.construct ==
               decisionforest::IndexVariable::GPUConstruct::ThreadBlock) {
      if (gpuDim.dimension == decisionforest::IndexVariable::Dimension::X)
        return gpu::Processor::ThreadX;
      else if (gpuDim.dimension == decisionforest::IndexVariable::Dimension::Y)
        return gpu::Processor::ThreadY;
      else if (gpuDim.dimension == decisionforest::IndexVariable::Dimension::Z)
        return gpu::Processor::ThreadZ;
    }
    assert(false && "Trying to get GPU processor from non-GPU index variable");
    return gpu::Processor::Sequential;
  }

  void addGPUMappingAttributeToParallelLoop(
      ConversionPatternRewriter &rewriter, Location location,
      scf::ParallelOp parallelLoop,
      const decisionforest::IndexVariable &indexVar) const {

    if (!isGPULoop(indexVar))
      return;
    auto gpuProcessor = getGPUProcessorFromIndexVar(indexVar);
    auto attr = rewriter.getAttr<gpu::ParallelLoopDimMappingAttr>(
        gpuProcessor, rewriter.getDimIdentityMap(),
        rewriter.getDimIdentityMap());
    std::vector<Attribute> attributes{attr};
    parallelLoop->setAttr(
        getMappingAttrName(),
        ArrayAttr::get(parallelLoop.getContext(), attributes));
  }

  void GenerateSingleLoop(ConversionPatternRewriter &rewriter,
                          Location location,
                          const decisionforest::IndexVariable &indexVar,
                          std::list<Value> batchIndices,
                          std::list<Value> treeIndices,
                          PredictOpLoweringState &state) const {
    auto range = indexVar.GetRange();
    auto stopConst =
        rewriter.create<arith::ConstantIndexOp>(location, range.m_stop);
    auto startConst =
        rewriter.create<arith::ConstantIndexOp>(location, range.m_start);
    auto stepConst =
        rewriter.create<arith::ConstantIndexOp>(location, range.m_step);

    if (indexVar.Parallel() || isGPULoop(indexVar)) {
      LoopConstructor<scf::ParallelOp> loopConstructor(
          std::list<const decisionforest::IndexVariable *>{&indexVar}, state,
          location, rewriter, ValueRange{startConst}, ValueRange{stopConst},
          ValueRange{stepConst}, batchIndices, treeIndices);
      auto parallelLoop = loopConstructor.GetLoop();

      std::list<const decisionforest::IndexVariable *> indexVarList{&indexVar};
      AddLoopInductionVariablesToIndexVariableLists(parallelLoop, batchIndices,
                                                    treeIndices, indexVarList);
      addGPUMappingAttributeToParallelLoop(rewriter, location, parallelLoop,
                                           indexVar);
      for (auto nestedIndexVar : indexVar.GetContainedLoops()) {
        GenerateLoop(rewriter, location, *nestedIndexVar, batchIndices,
                     treeIndices, state);
      }
    } else {
      LoopConstructor<scf::ForOp> loopConstructor(
          indexVar, state, location, rewriter, startConst, stopConst, stepConst,
          batchIndices, treeIndices);
      auto i = loopConstructor.GetLoop().getInductionVar();

      if (indexVar.GetType() ==
          decisionforest::IndexVariable::IndexVariableType::kBatch)
        batchIndices.push_back(i);
      else if (indexVar.GetType() ==
               decisionforest::IndexVariable::IndexVariableType::kTree)
        treeIndices.push_back(i);
      else
        assert(false && "Unknown index variable type!");
      for (auto nestedIndexVar : indexVar.GetContainedLoops()) {
        GenerateLoop(rewriter, location, *nestedIndexVar, batchIndices,
                     treeIndices, state);
      }
    }
  }

  void AddLoopInductionVariablesToIndexVariableLists(
      scf::ParallelOp parallelLoop, std::list<Value> &batchIndices,
      std::list<Value> &treeIndices,
      std::list<const decisionforest::IndexVariable *> &scheduleIndexVariables)
      const {
    auto iter = scheduleIndexVariables.begin();
    auto parallelLoopInductionVars = parallelLoop.getInductionVars();
    for (size_t i = 0; i < scheduleIndexVariables.size(); ++i) {
      auto &indexVar = **iter;
      if (indexVar.GetType() ==
          decisionforest::IndexVariable::IndexVariableType::kBatch)
        batchIndices.push_back(parallelLoopInductionVars[i]);
      else if (indexVar.GetType() ==
               decisionforest::IndexVariable::IndexVariableType::kTree)
        treeIndices.push_back(parallelLoopInductionVars[i]);
      else
        assert(false && "Unknown index variable type!");
    }
  }

  StringRef getMappingAttrName() const { return "mapping"; }

  void GenerateLoop(ConversionPatternRewriter &rewriter, Location location,
                    const decisionforest::IndexVariable &indexVar,
                    std::list<Value> batchIndices, std::list<Value> treeIndices,
                    PredictOpLoweringState &state) const {
    // This assert should be removed once we start supporting code generation
    // for tiled loops assert (indexVar.GetParentModifier() == nullptr); Any
    // index in the actual loop nest should not have indices derived from it
    assert(indexVar.GetIndexModifier() == nullptr);

    // Generate all the nested loops
    if (indexVar.GetContainedLoops().empty()) {
      // GPU loops should always contain some nested loops
      assert(indexVar.GetGPUDimension().construct ==
             decisionforest::IndexVariable::GPUConstruct::None);
      GenerateLeafLoop(rewriter, location, indexVar, batchIndices, treeIndices,
                       state);
    } else {
      if (indexVar.Unroll()) {
        GenerateUnrolledLoop(rewriter, location, indexVar, batchIndices,
                             treeIndices, state);
      } else {
        GenerateSingleLoop(rewriter, location, indexVar, batchIndices,
                           treeIndices, state);
      }
    }
  }

  LogicalResult LowerPredictForestOp_Schedule(
      Operation *op, mlir::decisionforest::PredictForestOp forestOp,
      ArrayRef<Value> operands, ConversionPatternRewriter &rewriter,
      mlir::MemRefType dataMemrefType, int64_t batchSize) const {
    PredictOpLoweringState state;

    auto location = op->getLoc();
    assert(dataMemrefType.getElementType().isa<mlir::FloatType>());

    // First initialize the result memref to zeros (This is not always needed,
    // for example for the default schedule, but leaving that optimization out
    // for now)
    InitPredictOpLoweringState(rewriter, location, state, forestOp, operands,
                               dataMemrefType, batchSize);
    InitializeResultMemref(rewriter, location, state);
    InitializeTreeClassWeightsMemref(rewriter, location, state);

    auto scheduleAttribute = forestOp.getSchedule();
    auto &schedule = *scheduleAttribute.GetSchedule();

    // Generate the loop nest
    auto rootIndex = schedule.GetRootIndex();
    assert(rootIndex);
    for (auto index : rootIndex->GetContainedLoops())
      GenerateLoop(rewriter, location, *index, std::list<Value>{},
                   std::list<Value>{}, state);

    // Generate the transformations to compute final prediction (sigmoid etc)
    TransformResultMemref(rewriter, location,
                          forestOp.getEnsemble()
                              .GetDecisionForest()
                              .GetPredictionTransformation(),
                          state);
    rewriter.replaceOp(op, static_cast<Value>(state.resultMemref));
    return mlir::success();
  }
};

struct HighLevelIRToMidLevelIRLoweringPass
    : public PassWrapper<HighLevelIRToMidLevelIRLoweringPass,
                         OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnOperation() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<memref::MemRefDialect, scf::SCFDialect,
                           decisionforest::DecisionForestDialect,
                           math::MathDialect, arith::ArithDialect,
                           func::FuncDialect, gpu::GPUDialect>();

    target.addIllegalOp<decisionforest::PredictForestOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<PredictForestOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace decisionforest {

void AddWalkDecisionTreeOpLoweringPass(mlir::OpPassManager &optPM);

void LowerFromHighLevelToMidLevelIR(mlir::MLIRContext &context,
                                    mlir::ModuleOp module) {
  // llvm::DebugFlag = true;
  // Lower from high-level IR to mid-level IR
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(std::make_unique<HighLevelIRToMidLevelIRLoweringPass>());
  AddWalkDecisionTreeOpLoweringPass(optPM);

  mlir::GreedyRewriteConfig config;
  std::vector<std::string> disabledPatterns = {
      "(anonymous namespace)::MergeNestedParallelLoops"};
  pm.addPass(createCanonicalizerPass(config, disabledPatterns));

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to mid level IR failed.\n";
  }
  // llvm::DebugFlag = false;
}

} // namespace decisionforest
} // namespace mlir
