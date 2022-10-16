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

using namespace mlir;

namespace {

static MemRefType getRowTypeFromArgumentType(MemRefType type) {
  assert (type.hasRank() && "expected only rank shapes");
  return MemRefType::get({type.getShape()[1]}, type.getElementType());
}

void InsertPrintMemrefOp(
  ConversionPatternRewriter &rewriter,
  Location location,
  int32_t kind,
  int32_t bitWidth,
  int64_t tileSize,
  Value memref,
  Type elementType) {
  
  /* Search for 'default argument promotions in C' to see why.
   Making all this 64-bit to avoid unecessary noise. */
  if (bitWidth < 64) {
    bitWidth = 64;
  }

  auto tileSizeConst = rewriter.create<arith::ConstantIntOp>(location, tileSize, rewriter.getI32Type());
  auto kindConst = rewriter.create<arith::ConstantIntOp>(location, kind, rewriter.getI32Type());
  auto bitWidthConst = rewriter.create<arith::ConstantIntOp>(location, bitWidth, rewriter.getI32Type());
  
  std::vector<Value> vectorValues;
  for (int32_t i=0; i<tileSize ; ++i) {
    auto index = rewriter.create<arith::ConstantIndexOp>(location, i);
    auto ithValue = rewriter.create<memref::LoadOp>(location, elementType, memref, static_cast<Value>(index));
    if (kind == 0)  { // Integer
      auto i64Value = rewriter.create<arith::ExtSIOp>(location, rewriter.getI64Type(), static_cast<Value>(ithValue));
      vectorValues.push_back(i64Value);
    }
    else {
      auto doubleValue = rewriter.create<arith::ExtFOp>(location, rewriter.getF64Type(), static_cast<Value>(ithValue));
      vectorValues.push_back(doubleValue);
    }
  }
  rewriter.create<decisionforest::PrintVectorOp>(location, kindConst, bitWidthConst, tileSizeConst, ValueRange(vectorValues));
}

void InsertPrintElementOp(
  ConversionPatternRewriter &rewriter,
  Location location,
  int32_t kind,
  int32_t bitWidth,
  Value value) {
    auto tileSizeConst = rewriter.create<arith::ConstantIntOp>(location, 1, rewriter.getI32Type());
    auto kindConst = rewriter.create<arith::ConstantIntOp>(location, kind, rewriter.getI32Type());
    auto bitWidthConst = rewriter.create<arith::ConstantIntOp>(location, bitWidth, rewriter.getI32Type());

    rewriter.create<decisionforest::PrintVectorOp>(location, kindConst, bitWidthConst, tileSizeConst, ValueRange{value});
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
  
  // Indices
  arith::ConstantIndexOp numClassesConst;
  arith::ConstantIndexOp oneIndexConst;
  arith::ConstantIndexOp zeroIndexConst;
  arith::ConstantIndexOp batchSizeConst;
  Value initialValueConst;
  
  // Decision Forest and Tree Stuff.
  mlir::decisionforest::TreeType treeType;
  mlir::decisionforest::EnsembleConstantOp forestConst;
} PredictOpLoweringState;

struct PredictForestOpLowering: public ConversionPattern {
  PredictForestOpLowering(MLIRContext *ctx) : ConversionPattern(mlir::decisionforest::PredictForestOp::getOperationName(), 1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    mlir::decisionforest::PredictForestOp forestOp = llvm::dyn_cast<mlir::decisionforest::PredictForestOp>(op);
    assert(forestOp);
    assert(operands.size() == 2);
    if (!forestOp)
        return mlir::failure();

    auto inputArgument = operands[0];
    int64_t batchSize = 1;
    if (inputArgument.getType().isa<mlir::MemRefType>())
    {
        auto memrefType = inputArgument.getType().cast<mlir::MemRefType>();
        if (memrefType.getShape().size() != 2) // We can currently only deal with 2D memrefs as inputs
            return mlir::failure();
        batchSize = memrefType.getShape()[0]; // The number of rows in our input memref is the batchsize
        // return LowerPredictForestOp_Batch(op, forestOp, operands, rewriter, memrefType, batchSize);
        return LowerPredictForestOp_Schedule(op, forestOp, operands, rewriter, memrefType, batchSize);  
    }
    else
    {
        assert(false && "Lowering for non-tensor argument not implemented");
        return mlir::failure();
    }
  }
  
  Value GetRow(ConversionPatternRewriter &rewriter, Location location, Value data, Value rowIndex, MemRefType dataMemrefType) const {
    auto rowType = getRowTypeFromArgumentType(dataMemrefType);
    auto zeroIndexAttr = rewriter.getIndexAttr(0);
    auto oneIndexAttr = rewriter.getIndexAttr(1);
    auto rowSizeAttr = rewriter.getIndexAttr(rowType.getShape()[0]);
    auto row = rewriter.create<memref::SubViewOp>(location, static_cast<Value>(data), ArrayRef<OpFoldResult>({rowIndex, zeroIndexAttr}),
                                                  ArrayRef<OpFoldResult>({oneIndexAttr, rowSizeAttr}), ArrayRef<OpFoldResult>({oneIndexAttr, oneIndexAttr}));
    if (decisionforest::InsertDebugHelpers) {
      rewriter.create<decisionforest::PrintInputRowOp>(location, row, rowIndex);
    }
    return row;
  }

  void InitPredictOpLoweringState(
    ConversionPatternRewriter &rewriter,
    Location location,
    PredictOpLoweringState& state,
    mlir::decisionforest::PredictForestOp forestOp,
    ArrayRef<Value> operands,
    MemRefType dataMemrefType,
    int64_t batchSize) const {

    auto resultType = forestOp->getResults()[0].getType();
    state.resultMemrefType = resultType.cast<mlir::MemRefType>();
    assert (state.resultMemrefType);
    state.resultMemref = operands[1];

    // Create the decision tree constant
    auto forestAttribute = forestOp.ensemble(); // Get the ensemble attribute
    auto forestType = forestAttribute.getType().cast<mlir::decisionforest::TreeEnsembleType>();
    state.forestConst = rewriter.create<mlir::decisionforest::EnsembleConstantOp>(location, forestType, forestAttribute);

    state.isMultiClass = forestAttribute.GetDecisionForest().IsMultiClassClassifier();
    state.treeType = forestType.getTreeType(0).cast<mlir::decisionforest::TreeType>();

    // Initialize constants
    state.batchSizeConst = rewriter.create<arith::ConstantIndexOp>(location, batchSize); 
    state.zeroIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
    state.oneIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 1); 
    state.numClassesConst = rewriter.create<arith::ConstantIndexOp>(location, forestAttribute.GetDecisionForest().GetNumClasses());
    auto initialValue = forestAttribute.GetDecisionForest().GetInitialOffset();
    state.initialValueConst = CreateFPConstant(rewriter, location, dataMemrefType.getElementType(), initialValue);

    // Initialize members for multi-class classification
    if (state.isMultiClass) {
      state.treeClassesMemrefType = MemRefType::get(
          {batchSize, (int64_t)forestAttribute.GetDecisionForest().GetNumClasses()},
          dataMemrefType.getElementType());

      state.treeClassesMemref = rewriter.create<memref::AllocaOp>(location, state.treeClassesMemrefType);
    }

    state.data = operands[0];
    state.dataMemrefType = dataMemrefType;
  }

  Value GenSigmoid(ConversionPatternRewriter& rewriter, Value operand, Location location) const {
    assert (operand.getType().isIntOrFloat());
    auto negate = rewriter.create<mlir::arith::NegFOp>(location, operand.getType(), operand);
    auto exponential = rewriter.create<mlir::math::ExpOp>(location, operand.getType(), static_cast<Value>(negate));
    
    Value oneConst;
    if (operand.getType().isa<mlir::Float64Type>())
      oneConst = rewriter.create<arith::ConstantFloatOp>(location, llvm::APFloat(1.0), operand.getType().cast<FloatType>());
    else if(operand.getType().isa<mlir::Float32Type>())
      oneConst = rewriter.create<arith::ConstantFloatOp>(location, llvm::APFloat((float)1.0), operand.getType().cast<FloatType>());
    else
      assert(false && "Unsupported floating point type");

    auto onePlusExp = rewriter.create<arith::AddFOp>(location, operand.getType(), oneConst, exponential);
    auto result = rewriter.create<arith::DivFOp>(location, operand.getType(), oneConst, onePlusExp);
    return result;
  }

  Value GenArgMax(ConversionPatternRewriter& rewriter, Location location, PredictOpLoweringState& state, Value index) const {
      auto treeClassesMemref = GetRow(rewriter, location, state.treeClassesMemref, index, state.treeClassesMemrefType);
      auto treeClassElementType = treeClassesMemref.getType().cast<MemRefType>().getElementType();

      if (decisionforest::PrintVectors) {
        InsertPrintMemrefOp(
          rewriter,
          location,
          0, /*int kind*/
          sizeof(float) * 8, /*bit width*/
          state.numClassesConst.value(), /* nelts in memref */
          treeClassesMemref,
          treeClassElementType);
      }

      auto firstValue = rewriter.create<memref::LoadOp>(
        location,
        treeClassElementType,
        static_cast<Value>(treeClassesMemref),
        ValueRange({state.zeroIndexConst, state.zeroIndexConst}));

      auto maxClassLoop = rewriter.create<scf::ForOp>(
        location,
        state.oneIndexConst,
        state.numClassesConst,
        state.oneIndexConst,
        ValueRange({static_cast<Value>(firstValue), static_cast<Value>(state.zeroIndexConst)}));
      
      rewriter.setInsertionPointToStart(maxClassLoop.getBody());

      auto k = maxClassLoop.getInductionVar();
      auto currentValue = rewriter.create<memref::LoadOp>(
        location,
        treeClassElementType,
        static_cast<Value>(treeClassesMemref),
        ValueRange({state.zeroIndexConst, k}));
      
      auto maxValue = maxClassLoop.getLoopBody().getArgument(1);
      auto maxIndex = maxClassLoop.getLoopBody().getArgument(2);

      auto compareResult = rewriter.create<arith::CmpFOp>(
        location,
        mlir::arith::CmpFPredicate::OGT,
        maxValue,
        currentValue);
      
      auto ifElse = rewriter.create<scf::IfOp>(
        location,
        TypeRange ( {treeClassElementType, k.getType()}),
        compareResult,
        true);
      {
        auto thenBodyBuilder = ifElse.getThenBodyBuilder();
        thenBodyBuilder.create<scf::YieldOp>(
          location,
          ValueRange({maxValue, maxIndex}));
      }
      {
        auto elseBodyBuilder = ifElse.getElseBodyBuilder();
        elseBodyBuilder.create<scf::YieldOp>(
          location,
          ValueRange({static_cast<Value>(currentValue), static_cast<Value>(k)}));
      }

      rewriter.create<scf::YieldOp>(
        location,
        ifElse.getResults());
      
      rewriter.setInsertionPointAfter(maxClassLoop);

      if (decisionforest::PrintVectors) {
        InsertPrintElementOp(rewriter, location, 1, sizeof(int64_t) * 8, maxClassLoop.getResult(1));
      }

      return rewriter.create<arith::IndexCastOp>(location, state.treeType.getResultType(), static_cast<Value>(maxClassLoop.getResult(1)));
  }

  LogicalResult
  LowerPredictForestOp_Batch(Operation *op, mlir::decisionforest::PredictForestOp forestOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter, 
                            mlir::MemRefType dataMemrefType, int64_t batchSize) const
  {
        auto location = op->getLoc();
        assert (dataMemrefType.getElementType().isa<mlir::FloatType>());

        PredictOpLoweringState state;
        InitPredictOpLoweringState(rewriter, location, state, forestOp, operands, dataMemrefType, batchSize);
        InitializeTreeClassWeightsMemref(rewriter, location, state);
        // auto resultMemrefType = convertTensorToMemRef(resultTensorType);
        // auto allocOp = rewriter.create<memref::AllocOp>(location, resultMemrefType);

        auto dataElementType = state.dataMemrefType.getElementType();
        // auto floatConstZero = rewriter.create<ConstantFloatOp>(location, llvm::APFloat(0.0), resultMemrefType.getElementType().cast<mlir::FloatType>());
        // std::vector<Value> values(batchSize, static_cast<Value>(floatConstZero));

        // Create the decision tree constant
        auto forestAttribute = forestOp.ensemble(); // Get the ensemble attribute
        auto forestType = forestAttribute.getType().cast<mlir::decisionforest::TreeEnsembleType>();
        
        auto batchLoop = rewriter.create<scf::ForOp>(location, state.zeroIndexConst, state.batchSizeConst, state.oneIndexConst/*, static_cast<Value>(memrefResult)*/);
        
        rewriter.setInsertionPointToStart(batchLoop.getBody());
        auto i = batchLoop.getInductionVar();

        // Extract the slice of the input tensor for the current iteration -- row i
        auto rowType = getRowTypeFromArgumentType(dataMemrefType);
        auto zeroIndexAttr = rewriter.getIndexAttr(0);
        auto oneIndexAttr = rewriter.getIndexAttr(1);
        auto rowSizeAttr = rewriter.getIndexAttr(rowType.getShape()[0]);
        auto row = rewriter.create<memref::SubViewOp>(location, static_cast<Value>(state.data), ArrayRef<OpFoldResult>({i, zeroIndexAttr}),
                                                      ArrayRef<OpFoldResult>({oneIndexAttr, rowSizeAttr}), ArrayRef<OpFoldResult>({oneIndexAttr, oneIndexAttr}));
        if (decisionforest::InsertDebugHelpers) {
          rewriter.create<decisionforest::PrintInputRowOp>(location, row, i);
        }

        // Create a for loop over the trees
        auto resultElementType = state.resultMemrefType.getElementType();
        int64_t numTrees = static_cast<int64_t>(forestType.getNumberOfTrees());
        auto ensembleSizeConst = rewriter.create<arith::ConstantIndexOp>(location, numTrees); 
        
        Value initialValueConst;
        double_t initialValueValue = forestAttribute.GetDecisionForest().GetInitialOffset();

        if (dataElementType.isa<mlir::Float64Type>())
          initialValueConst = rewriter.create<arith::ConstantFloatOp>(location, llvm::APFloat(initialValueValue), dataElementType.cast<FloatType>());
        else if(dataElementType.isa<mlir::Float32Type>())
          initialValueConst = rewriter.create<arith::ConstantFloatOp>(location, llvm::APFloat((float)initialValueValue), dataElementType.cast<FloatType>());
        else
          assert(false && "Unsupported floating point type");
        // auto oneIndexConst2 = rewriter.create<ConstantIndexOp>(location, 1);
        auto treeLoopArgs = state.isMultiClass ? ValueRange({}) : ValueRange({initialValueConst});
      
        auto treeLoop = rewriter.create<scf::ForOp>(
          location,
          state.zeroIndexConst,
          ensembleSizeConst,
          state.oneIndexConst,
          treeLoopArgs);
        
        rewriter.setInsertionPointToStart(treeLoop.getBody());
        auto j = treeLoop.getInductionVar();

        // TODO The forest shouldn't contain the tree type because each tree may have a different type, 
        // but the tree type should have more information -- tiling for example. We need to construct 
        // the default tiling here.
        assert (forestType.doAllTreesHaveSameTileSize());
        auto tree = rewriter.create<decisionforest::GetTreeFromEnsembleOp>(location, state.treeType, state.forestConst, j);

        
        // Create the while loop to walk the tree
        auto walkOp = rewriter.create<decisionforest::WalkDecisionTreeOp>(location, state.treeType.getThresholdType(), tree, row);
        // rewriter.create<memref::StoreOp>(location, TypeRange({ }), static_cast<Value>(treePrediction), memrefResult, j);
        // result[i]
        
        // auto readResultOfi = rewriter.create<memref::LoadOp>(location, resultElementType, memrefResult, i);
        // Accumulate the tree prediction
        
        if (state.isMultiClass) {
          GenerateMultiClassAccumulate(rewriter, location, static_cast<Value>(walkOp), i, j, state);
            // rewriter.create<scf::YieldOp>(location, static_cast<Value>(accumulatedValue));
        }
        else {
            assert(forestType.getReductionType() == decisionforest::ReductionType::kAdd);
            auto accumulatedValue = rewriter.create<arith::AddFOp>(location, resultElementType, treeLoop.getBody()->getArguments()[1], walkOp);

            if (mlir::decisionforest::InsertDebugHelpers) {
              Value treePred = walkOp;
              if (!walkOp.getResult().getType().isF64())
                treePred = rewriter.create<arith::ExtFOp>(location, rewriter.getF64Type(), walkOp.getResult());
              rewriter.create<decisionforest::PrintTreePredictionOp>(location, treePred, j);
            }
            // auto updatedResultTensor = rewriter.create<tensor::InsertOp>(location, resultMemrefType, accumulatedValue, treeLoop.getBody()->getArguments()[1], i);
            rewriter.create<scf::YieldOp>(location, static_cast<Value>(accumulatedValue));
        }

        rewriter.setInsertionPointAfter(treeLoop);
        // result[i] = Accumulated value
        auto predTransform = forestAttribute.GetDecisionForest().GetPredictionTransformation();
        Value transformedValue;
        if (predTransform == decisionforest::PredictionTransformation::kIdentity)
          transformedValue = static_cast<Value>(treeLoop.getResults()[0]);
        else if (predTransform == decisionforest::PredictionTransformation::kSigmoid) {
          transformedValue = GenSigmoid(rewriter, static_cast<Value>(treeLoop.getResults()[0]), location);
        }
        else if (predTransform == decisionforest::PredictionTransformation::kSoftMax) {
          assert(state.isMultiClass);
          transformedValue = GenArgMax(rewriter, location, state, i);
        }
        else {
          assert (false && "Unsupported prediction transformation type");
        }
        rewriter.create<memref::StoreOp>(location, TypeRange({ }), transformedValue, state.resultMemref, i);
        // rewriter.create<scf::YieldOp>(location); //, static_cast<Value>(treeLoop.results()[0]));S

        rewriter.replaceOp(op, static_cast<Value>(state.resultMemref));
        return mlir::success();
  }

  Value CreateFPConstant(ConversionPatternRewriter &rewriter, Location location, Type type, double value) const {
    Value constValue;
    if (type.isa<mlir::Float64Type>())
      constValue = rewriter.create<arith::ConstantFloatOp>(location, llvm::APFloat(value), type.cast<FloatType>());
    else if(type.isa<mlir::Float32Type>())
      constValue = rewriter.create<arith::ConstantFloatOp>(location, llvm::APFloat((float)value), type.cast<FloatType>());
    else if (type.isa<mlir::IntegerType>())
      constValue = rewriter.create<arith::ConstantIntOp>(location, 0, type);
    else
      assert(false && "Unsupported floating point type");
    return constValue;
  }

  void InitializeTreeClassWeightsMemref(ConversionPatternRewriter &rewriter, Location location, PredictOpLoweringState& state) const {
    if (state.isMultiClass) {
      auto outerLoop = rewriter.create<scf::ForOp>(location, state.zeroIndexConst, state.batchSizeConst, state.oneIndexConst);
      rewriter.setInsertionPointToStart(outerLoop.getBody());
      {
        auto i = outerLoop.getInductionVar();
        auto row = GetRow(rewriter, location, state.treeClassesMemref, i, state.treeClassesMemrefType);
        auto innerLoop = rewriter.create<scf::ForOp>(location, state.zeroIndexConst, state.numClassesConst, state.oneIndexConst);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        {
          auto j = innerLoop.getInductionVar();
        
          rewriter.create<memref::StoreOp>(
            location,
            static_cast<Value>(state.initialValueConst),
            static_cast<Value>(row),
            ValueRange({state.zeroIndexConst, j}));
          rewriter.setInsertionPointAfter(innerLoop);
        }
      }
      rewriter.setInsertionPointAfter(outerLoop);
    }
  }

  void InitializeResultMemref(ConversionPatternRewriter &rewriter, Location location, PredictOpLoweringState& state) const {

    // We don't accumulate into result memref in case of multi-class.
    if (state.isMultiClass) return;

    // Create a for loop over the outputs
    auto batchLoop = rewriter.create<scf::ForOp>(location, state.zeroIndexConst, state.batchSizeConst, state.oneIndexConst);
    
    rewriter.setInsertionPointToStart(batchLoop.getBody());
    auto i = batchLoop.getInductionVar();
    rewriter.create<memref::StoreOp>(location, state.initialValueConst, state.resultMemref, i);
    rewriter.setInsertionPointAfter(batchLoop);
  }

  void TransformResultMemref(
    ConversionPatternRewriter &rewriter, Location location, decisionforest::PredictionTransformation predTransform, PredictOpLoweringState& state) const {
    
    if (predTransform == decisionforest::PredictionTransformation::kIdentity)
      return;

    // assert (resultMemrefType.getElementType().isa<mlir::FloatType>());
    // assert (predTransform == decisionforest::PredictionTransformation::kSigmoid);

    auto batchLoop = rewriter.create<scf::ForOp>(location, state.zeroIndexConst, state.batchSizeConst, state.oneIndexConst);
    
    rewriter.setInsertionPointToStart(batchLoop.getBody());
    auto i = batchLoop.getInductionVar();
    
    auto memrefElem = rewriter.create<memref::LoadOp>(location, state.resultMemref, i);
    Value transformedValue;
    if (predTransform == decisionforest::PredictionTransformation::kSigmoid)
      transformedValue = GenSigmoid(rewriter, static_cast<Value>(memrefElem), location);
    else if (predTransform == decisionforest::PredictionTransformation::kSoftMax)
      transformedValue = GenArgMax(rewriter, location, state, i);
    else
      assert(false && "Unsupported prediction transformation.");

    rewriter.create<memref::StoreOp>(location, transformedValue, state.resultMemref, i);
    rewriter.setInsertionPointAfter(batchLoop);
  }

  Value SumOfValues(ConversionPatternRewriter &rewriter, Location location, std::list<Value>& values) const {
    assert (!values.empty());
    if (values.size() == 1) {
      return values.front();
    }
    auto iter = values.begin();
    auto valueType = iter->getType();
    auto lhs = *iter;
    ++iter;
    auto rhs = *iter;
    ++iter;
    auto accumulator = rewriter.create<arith::AddIOp>(location, valueType, lhs, rhs);
    for (; iter!=values.end() ; ++iter) {
      accumulator = rewriter.create<arith::AddIOp>(location, valueType, static_cast<Value>(accumulator), *iter);
    }
    return accumulator;
  }
  
  void GenerateMultiClassAccumulate(ConversionPatternRewriter& rewriter, Location location, Value result, Value rowIndex, Value index, PredictOpLoweringState& state) const {
    if (state.isMultiClass) {
      auto batchTreeClassMemref = GetRow(rewriter, location, state.treeClassesMemref, rowIndex, state.treeClassesMemrefType);
      auto classId = rewriter.create<decisionforest::GetTreeClassIdOp>(location, state.treeType.getResultType(), state.forestConst, index);
      auto classIdIndex = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(classId));
      auto currentValue = rewriter.create<memref::LoadOp>(
        location,
        state.treeClassesMemrefType.getElementType(),
        batchTreeClassMemref,
        ValueRange({state.zeroIndexConst, classIdIndex}));
      
      auto accumulatedValue = rewriter.create<arith::AddFOp>(location, state.dataMemrefType.getElementType(), currentValue, result);
      rewriter.create<memref::StoreOp>(location, static_cast<Value>(accumulatedValue), batchTreeClassMemref, ValueRange({state.zeroIndexConst, classIdIndex}));
    }
  }

  Value GenerateTreeIndexLeafLoopBody(ConversionPatternRewriter &rewriter, Location location, const decisionforest::IndexVariable& indexVar, 
                        std::list<Value> treeIndices, PredictOpLoweringState& state, Value row, Value rowIndex, Value prevAccumulatorValue) const {
   
    Value treeIndex = SumOfValues(rewriter, location, treeIndices);

    // Get the current tree
    auto forestType = state.forestConst.getType().cast<decisionforest::TreeEnsembleType>();
    assert (forestType.doAllTreesHaveSameTileSize()); // TODO how do we check which type of tree we'll get here?
    auto treeType = forestType.getTreeType(0).cast<mlir::decisionforest::TreeType>();
    auto tree = rewriter.create<decisionforest::GetTreeFromEnsembleOp>(location, treeType, state.forestConst, treeIndex);

    // Walk the tree
    Value walkOp;
    if (indexVar.PeelWalk()) {
      auto peelItersAttrib = rewriter.getI32IntegerAttr(indexVar.IterationsToPeel());
      walkOp = rewriter.create<decisionforest::WalkDecisionTreePeeledOp>(location, treeType.getThresholdType(), tree, row, peelItersAttrib);
    }
    else {
      walkOp = rewriter.create<decisionforest::WalkDecisionTreeOp>(location, treeType.getThresholdType(), tree, row);
    }
    GenerateMultiClassAccumulate(rewriter, location, static_cast<Value>(walkOp), rowIndex, treeIndex, state);

    if (state.isMultiClass) return prevAccumulatorValue;

    // Accumulate the tree prediction
    assert(forestType.getReductionType() == decisionforest::ReductionType::kAdd);
    auto accumulatedValue = rewriter.create<arith::AddFOp>(location, state.resultMemrefType.getElementType(), prevAccumulatorValue, walkOp);

    if (mlir::decisionforest::InsertDebugHelpers) {
      Value treePred = walkOp;
      if (!treePred.getType().isF64())
        treePred = rewriter.create<arith::ExtFOp>(location, rewriter.getF64Type(), walkOp);
      rewriter.create<decisionforest::PrintTreePredictionOp>(location, treePred, treeIndex);
    }
    // auto updatedResultTensor = rewriter.create<tensor::InsertOp>(location, resultMemrefType, accumulatedValue, treeLoop.getBody()->getArguments()[1], i);
    return accumulatedValue;
  }

  Value GeneratePipelinedTreeIndexLeafLoopBody(
    ConversionPatternRewriter &rewriter,
    Location location,
    int32_t stepSize,
    std::list<Value> treeIndices,
    PredictOpLoweringState& state,
    Value row,
    Value rowIndex,
    Value prevAccumulatorValue) const {
    
    // Get the current tree
    auto forestType = state.forestConst.getType().cast<decisionforest::TreeEnsembleType>();
    assert (forestType.doAllTreesHaveSameTileSize()); // TODO how do we check which type of tree we'll get here?
    auto treeType = forestType.getTreeType(0).cast<mlir::decisionforest::TreeType>();

    std::vector <Value> finalTreeIndices;
    std::vector <Value> rows;
    std::vector <Value> trees;
    std::vector <Type> treeResultTypes;
    for (int32_t i = 0; i < stepSize; i++) {
      treeIndices.push_back(rewriter.create<arith::ConstantIndexOp>(location, i));
      
      Value treeIndex = SumOfValues(rewriter, location, treeIndices);
      auto tree = rewriter.create<decisionforest::GetTreeFromEnsembleOp>(location, treeType, state.forestConst, treeIndex);
      finalTreeIndices.push_back(treeIndex);
      trees.push_back(tree);
      treeResultTypes.push_back(treeType.getThresholdType());
      rows.push_back(row);

      treeIndices.pop_back();
    }

    // Walk the tree.
    auto unrollLoopAttr = decisionforest::UnrollLoopAttribute::get(treeType, -1);
    auto walkOp = rewriter.create<decisionforest::PipelinedWalkDecisionTreeOp>(location, treeResultTypes, unrollLoopAttr, trees, rows);
    
    for (size_t i = 0; i < trees.size(); i++) {
      if (state.isMultiClass) {
        GenerateMultiClassAccumulate(rewriter, location, walkOp.getResult(i), rowIndex, finalTreeIndices[i], state);
      }
      else {
          // Accumulate the tree prediction
        assert(forestType.getReductionType() == decisionforest::ReductionType::kAdd);
        prevAccumulatorValue = rewriter.create<arith::AddFOp>(location, state.resultMemrefType.getElementType(), prevAccumulatorValue, walkOp.getResult(i));
      }

      if (mlir::decisionforest::InsertDebugHelpers) {
        rewriter.create<decisionforest::PrintTreePredictionOp>(location, walkOp.getResult(i), finalTreeIndices[i]);
      }
      // auto updatedResultTensor = rewriter.create<tensor::InsertOp>(location, resultMemrefType, accumulatedValue, treeLoop.getBody()->getArguments()[1], i);
    }
    
    return prevAccumulatorValue;
  }

  void GenerateLeafLoopForTreeIndex(ConversionPatternRewriter &rewriter, Location location, const decisionforest::IndexVariable& indexVar, 
                        std::list<Value> batchIndices, std::list<Value> treeIndices, PredictOpLoweringState& state) const {
    
    assert (indexVar.GetType() == decisionforest::IndexVariable::IndexVariableType::kTree);
    
    Value rowIndex = SumOfValues(rewriter, location, batchIndices);
    
    // Get the current row
    Value row = GetRow(rewriter, location, state.data, rowIndex, state.dataMemrefType);

    if(indexVar.Unroll()) {
      auto range = indexVar.GetRange();
      auto zeroConst = CreateFPConstant(rewriter, location, state.dataMemrefType.getElementType(), 0.0);      
      Value accumulatedValue = zeroConst;
      for (int32_t i=range.m_start ; i<range.m_stop ; i+=range.m_step) {
        auto treeIndex = rewriter.create<arith::ConstantIndexOp>(location, i);
        treeIndices.push_back(treeIndex);  
        accumulatedValue = GenerateTreeIndexLeafLoopBody(rewriter, location, indexVar, treeIndices, state, row, rowIndex, accumulatedValue);
      }

      // Don't accumulate into memref in case of multiclass.
      if (state.isMultiClass) return;

      // Generate the store back in to the result memref
      auto currentMemrefElem = rewriter.create<memref::LoadOp>(location, state.resultMemref, ValueRange{rowIndex});
      auto newMemrefElem = rewriter.create<arith::AddFOp>(location, state.resultMemrefType.getElementType(), accumulatedValue, currentMemrefElem);
      rewriter.create<memref::StoreOp>(location, newMemrefElem, state.resultMemref, ValueRange{rowIndex});
    }
    else if (indexVar.Pipelined()) {
      auto range = indexVar.GetRange();
      int32_t peeledLoopStart = range.m_stop - (range.m_stop - range.m_start) % range.m_step;
      int32_t peeledLoopStep = range.m_stop - peeledLoopStart;

      auto stopConst = rewriter.create<arith::ConstantIndexOp>(location, peeledLoopStart); 
      auto startConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_start);
      auto stepConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_step);

      auto zeroConst = CreateFPConstant(rewriter, location, state.dataMemrefType.getElementType(), 0.0);      

      scf::ForOp loop = rewriter.create<scf::ForOp>(location, startConst, stopConst, stepConst, ValueRange{ zeroConst });
      rewriter.setInsertionPointToStart(loop.getBody());
      treeIndices.push_back(loop.getInductionVar());
      auto accumulatedValue = GeneratePipelinedTreeIndexLeafLoopBody(rewriter, location, range.m_step, treeIndices, state, row, rowIndex, loop.getBody()->getArguments()[1]);
      rewriter.create<scf::YieldOp>(location, static_cast<Value>(accumulatedValue));
      rewriter.setInsertionPointAfter(loop);

      // Don't accumulate into memref in case of multiclass.
      if (!state.isMultiClass) {
        // Generate the store back in to the result memref
        auto currentMemrefElem = rewriter.create<memref::LoadOp>(location, state.resultMemref, ValueRange{rowIndex});
        auto newMemrefElem = rewriter.create<arith::AddFOp>(location, state.resultMemrefType.getElementType(), loop.getResults()[0], currentMemrefElem);
        rewriter.create<memref::StoreOp>(location, newMemrefElem, state.resultMemref, ValueRange{rowIndex});
      }

      if (peeledLoopStart < range.m_stop) {
        stopConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_stop); 
        startConst = rewriter.create<arith::ConstantIndexOp>(location, peeledLoopStart);
        stepConst = rewriter.create<arith::ConstantIndexOp>(location, peeledLoopStep);      

        scf::ForOp peeledLoop = rewriter.create<scf::ForOp>(location, startConst, stopConst, stepConst, ValueRange{ zeroConst });
        rewriter.setInsertionPointToStart(peeledLoop.getBody());
        
        treeIndices.pop_back();
        treeIndices.push_back(peeledLoop.getInductionVar());
        
        auto accumulatedValue = GeneratePipelinedTreeIndexLeafLoopBody(rewriter, location, peeledLoopStep, treeIndices, state, row, rowIndex, peeledLoop.getBody()->getArguments()[1]);
        rewriter.create<scf::YieldOp>(location, static_cast<Value>(accumulatedValue));
        rewriter.setInsertionPointAfter(peeledLoop);

        if (!state.isMultiClass) {

          auto currentMemrefElem = rewriter.create<memref::LoadOp>(location, state.resultMemref, ValueRange{rowIndex});
          auto newMemrefElem = rewriter.create<arith::AddFOp>(location, state.resultMemrefType.getElementType(), loop.getResults()[0], currentMemrefElem);
          rewriter.create<memref::StoreOp>(location, newMemrefElem, state.resultMemref, ValueRange{rowIndex});
        }
      }
    }
    else {
      // Generate leaf loop for tree index var
      auto range = indexVar.GetRange();
      auto stopConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_stop); 
      auto startConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_start);
      auto stepConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_step);

      auto zeroConst = CreateFPConstant(rewriter, location, state.dataMemrefType.getElementType(), 0.0);      

      scf::ForOp loop = rewriter.create<scf::ForOp>(location, startConst, stopConst, stepConst, ValueRange{ zeroConst });
      rewriter.setInsertionPointToStart(loop.getBody());
      treeIndices.push_back(loop.getInductionVar());
      auto accumulatedValue = GenerateTreeIndexLeafLoopBody(rewriter, location, indexVar, treeIndices, state, row, rowIndex, loop.getBody()->getArguments()[1]);
      rewriter.create<scf::YieldOp>(location, static_cast<Value>(accumulatedValue));
      rewriter.setInsertionPointAfter(loop);
      
      // Don't accumulate into memref in case of multiclass.
      if (state.isMultiClass) return;
      
      // Generate the store back in to the result memref
      auto currentMemrefElem = rewriter.create<memref::LoadOp>(location, state.resultMemref, ValueRange{rowIndex});
      auto newMemrefElem = rewriter.create<arith::AddFOp>(location, state.resultMemrefType.getElementType(), loop.getResults()[0], currentMemrefElem);
      rewriter.create<memref::StoreOp>(location, newMemrefElem, state.resultMemref, ValueRange{rowIndex});
    }
  }

  void GeneratePipelinedBatchIndexLeafLoopBody(
    ConversionPatternRewriter &rewriter,
    Location location,
    const decisionforest::IndexVariable& indexVar,
    int32_t stepSize,
    std::list<Value> batchIndices,
    decisionforest::TreeType treeType,
    Value tree,
    Value treeIndex,
    PredictOpLoweringState& state) const {

    std::vector <Value> rowIndices;
    std::vector <Value> rows;
    std::vector <Value> trees;
    std::vector <Type> treeResultTypes;
    for (int32_t i = 0; i < stepSize; i++) {
      batchIndices.push_back(rewriter.create<arith::ConstantIndexOp>(location, i));
      auto rowIndex = SumOfValues(rewriter, location, batchIndices);
      
      rows.push_back(GetRow(rewriter, location, state.data, rowIndex, state.dataMemrefType));
      rowIndices.push_back(rowIndex);
      trees.push_back(tree);
      treeResultTypes.push_back(treeType.getThresholdType());
      
      batchIndices.pop_back();
    }

    // Walk the tree
    auto unrollLoopAttr = decisionforest::UnrollLoopAttribute::get(treeType, indexVar.GetContainingLoop()->GetUnrollFactor());
    auto walkOp = rewriter.create<decisionforest::PipelinedWalkDecisionTreeOp>(location, treeResultTypes, unrollLoopAttr, trees, rows);
    for (size_t i = 0; i < rowIndices.size(); i++) {
      // Don't accumulate into memref in case of multiclass.
      if (state.isMultiClass) {
        GenerateMultiClassAccumulate(rewriter, location, walkOp.getResult(i), rowIndices[i], treeIndex, state);
      }
      else {
        // Accumulate the tree prediction and generate the store back in to the result memref
        // TODO - Check of load and store value range can just store all row indices at once
        auto currentMemrefElem = rewriter.create<memref::LoadOp>(location, state.resultMemref, ValueRange{rowIndices[i]});
        auto accumulatedValue = rewriter.create<arith::AddFOp>(location, state.resultMemrefType.getElementType(), walkOp.getResult(i), currentMemrefElem);
        rewriter.create<memref::StoreOp>(location, accumulatedValue, state.resultMemref, ValueRange{rowIndices[i]});

        if (mlir::decisionforest::InsertDebugHelpers) {
          rewriter.create<decisionforest::PrintTreePredictionOp>(location, walkOp.getResult(i), treeIndex);
        }
      }
    }
  }

  void GenerateBatchIndexLeafLoopBody(ConversionPatternRewriter &rewriter, Location location, const decisionforest::IndexVariable& indexVar, 
                        std::list<Value> batchIndices, decisionforest::TreeType treeType, Value tree, Value treeIndex,
                        PredictOpLoweringState& state) const {
    Value rowIndex = SumOfValues(rewriter, location, batchIndices);
    
    // Get the current row
    Value row = GetRow(rewriter, location, state.data, rowIndex, state.dataMemrefType);

    // Walk the tree
    Value walkOp;
    if (indexVar.PeelWalk()) {
      auto peelItersAttrib = rewriter.getI32IntegerAttr(indexVar.IterationsToPeel());
      walkOp = rewriter.create<decisionforest::WalkDecisionTreePeeledOp>(location, treeType.getThresholdType(), tree, row, peelItersAttrib);
    }
    else {
      walkOp = rewriter.create<decisionforest::WalkDecisionTreeOp>(location, treeType.getThresholdType(), tree, row);
    }
    
    GenerateMultiClassAccumulate(rewriter, location, static_cast<Value>(walkOp), rowIndex, treeIndex, state);

    // Don't accumulate into memref in case of multiclass.
    if (state.isMultiClass) return;

    // Accumulate the tree prediction and generate the store back in to the result memref
    auto currentMemrefElem = rewriter.create<memref::LoadOp>(location, state.resultMemref, ValueRange{rowIndex});
    auto accumulatedValue = rewriter.create<arith::AddFOp>(location, state.resultMemrefType.getElementType(), walkOp, currentMemrefElem);
    rewriter.create<memref::StoreOp>(location, accumulatedValue, state.resultMemref, ValueRange{rowIndex});

    if (mlir::decisionforest::InsertDebugHelpers) {
      Value treePred = walkOp;
      if (!treePred.getType().isF64())
        treePred = rewriter.create<arith::ExtFOp>(location, rewriter.getF64Type(), walkOp);
      rewriter.create<decisionforest::PrintTreePredictionOp>(location, treePred, treeIndex);
    }
  }

  void GenerateLeafLoopForBatchIndex(ConversionPatternRewriter &rewriter, Location location, const decisionforest::IndexVariable& indexVar, 
                        std::list<Value> batchIndices, std::list<Value> treeIndices, PredictOpLoweringState& state) const {

    assert (indexVar.GetType() == decisionforest::IndexVariable::IndexVariableType::kBatch);

    Value treeIndex = SumOfValues(rewriter, location, treeIndices);

    // Get the current tree
    auto forestType = state.forestConst.getType().cast<decisionforest::TreeEnsembleType>();
    assert (forestType.doAllTreesHaveSameTileSize()); // TODO how do we check which type of tree we'll get here?
    assert(forestType.getReductionType() == decisionforest::ReductionType::kAdd);
    auto treeType = forestType.getTreeType(0).cast<mlir::decisionforest::TreeType>();
    auto tree = rewriter.create<decisionforest::GetTreeFromEnsembleOp>(location, treeType, state.forestConst, treeIndex);

    if (indexVar.Unroll()) {
      auto range = indexVar.GetRange();
      for (int32_t i=range.m_start ; i<range.m_stop ; i+=range.m_step) {
        auto batchIndex = rewriter.create<arith::ConstantIndexOp>(location, i);
        batchIndices.push_back(batchIndex);
        GenerateBatchIndexLeafLoopBody(rewriter, location, indexVar, batchIndices, treeType, tree, treeIndex, state);
      }
    }
    else if (indexVar.Pipelined()) {
      // Currently supports only single variable.
      auto range = indexVar.GetRange();
      
      int32_t peeledLoopStart = range.m_stop - (range.m_stop - range.m_start) % range.m_step;
      int32_t peeledLoopStep = range.m_stop - peeledLoopStart;

      auto stopConst = rewriter.create<arith::ConstantIndexOp>(location, peeledLoopStart); 
      auto startConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_start);
      auto stepConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_step);
      
      scf::ForOp loop = rewriter.create<scf::ForOp>(location, startConst, stopConst, stepConst);
      rewriter.setInsertionPointToStart(loop.getBody());
      batchIndices.push_back(loop.getInductionVar());
      GeneratePipelinedBatchIndexLeafLoopBody(rewriter, location, indexVar, range.m_step, batchIndices, treeType, tree, treeIndex, state);
      rewriter.setInsertionPointAfter(loop);

      if (peeledLoopStart < range.m_stop) {
        stopConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_stop); 
        startConst = rewriter.create<arith::ConstantIndexOp>(location, peeledLoopStart);
        stepConst = rewriter.create<arith::ConstantIndexOp>(location, peeledLoopStep);

        scf::ForOp peeledLoop = rewriter.create<scf::ForOp>(location, startConst, stopConst, stepConst);
        rewriter.setInsertionPointToStart(peeledLoop.getBody());

        batchIndices.pop_back();
        batchIndices.push_back(peeledLoop.getInductionVar());

        GeneratePipelinedBatchIndexLeafLoopBody(rewriter, location, indexVar, peeledLoopStep, batchIndices, treeType, tree, treeIndex, state);
        rewriter.setInsertionPointAfter(peeledLoop);
      }
    }
    else {
      // Generate leaf loop for tree index var
      auto range = indexVar.GetRange();
      auto stopConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_stop); 
      auto startConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_start);
      auto stepConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_step);

      scf::ForOp loop = rewriter.create<scf::ForOp>(location, startConst, stopConst, stepConst);
      rewriter.setInsertionPointToStart(loop.getBody());
      batchIndices.push_back(loop.getInductionVar());
      GenerateBatchIndexLeafLoopBody(rewriter, location, indexVar, batchIndices, treeType, tree, treeIndex, state);
      rewriter.setInsertionPointAfter(loop);
    }
  }

  void GenerateLeafLoop(ConversionPatternRewriter &rewriter, Location location, const decisionforest::IndexVariable& indexVar, 
                        std::list<Value> batchIndices, std::list<Value> treeIndices, PredictOpLoweringState& state) const {
    if (indexVar.GetType() == decisionforest::IndexVariable::IndexVariableType::kTree) {    
      GenerateLeafLoopForTreeIndex(rewriter, location, indexVar, batchIndices, treeIndices, state);
    }
    else {
      GenerateLeafLoopForBatchIndex(rewriter, location, indexVar, batchIndices, treeIndices, state);
    }
  }

  void GenerateUnrolledLoop(ConversionPatternRewriter &rewriter, Location location, const decisionforest::IndexVariable& indexVar, 
                        std::list<Value> batchIndices, std::list<Value> treeIndices, PredictOpLoweringState& state) const {
    auto range = indexVar.GetRange();

    for (int32_t i=range.m_start ; i<range.m_stop ; ++i) {
      auto indexVal = rewriter.create<arith::ConstantIndexOp>(location, i);

      if (indexVar.GetType() == decisionforest::IndexVariable::IndexVariableType::kBatch)
        batchIndices.push_back(indexVal);
      else if (indexVar.GetType() == decisionforest::IndexVariable::IndexVariableType::kTree)
        treeIndices.push_back(indexVal);
      else
        assert (false && "Unknown index variable type!");

      for (auto nestedIndexVar : indexVar.GetContainedLoops()) {
        GenerateLoop(rewriter, location, *nestedIndexVar, batchIndices, treeIndices, state);
      }

      if (indexVar.GetType() == decisionforest::IndexVariable::IndexVariableType::kBatch)
        batchIndices.pop_back();
      else if (indexVar.GetType() == decisionforest::IndexVariable::IndexVariableType::kTree)
        treeIndices.pop_back();
      else
        assert (false && "Unknown index variable type!");

    }
  }

  void GenerateSingleLoop(ConversionPatternRewriter &rewriter, Location location, const decisionforest::IndexVariable& indexVar, 
                    std::list<Value> batchIndices, std::list<Value> treeIndices, PredictOpLoweringState& state) const {
    auto range = indexVar.GetRange();
    auto stopConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_stop); 
    auto startConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_start);
    auto stepConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_step);

    Value i;
    mlir::Operation* loopOp = nullptr;
    if (indexVar.Parallel()) {
      auto loop = rewriter.create<scf::ParallelOp>(location, ValueRange{startConst}, ValueRange{stopConst}, ValueRange{stepConst});
      rewriter.setInsertionPointToStart(loop.getBody());
      loopOp = loop;
      i = loop.getInductionVars()[0];
    }
    else {
      auto loop = rewriter.create<scf::ForOp>(location, startConst, stopConst, stepConst);
      rewriter.setInsertionPointToStart(loop.getBody());
      loopOp = loop;
      i = loop.getInductionVar();
    }

    if (indexVar.GetType() == decisionforest::IndexVariable::IndexVariableType::kBatch)
      batchIndices.push_back(i);
    else if (indexVar.GetType() == decisionforest::IndexVariable::IndexVariableType::kTree)
      treeIndices.push_back(i);
    else
      assert (false && "Unknown index variable type!");

    for (auto nestedIndexVar : indexVar.GetContainedLoops()) {
      GenerateLoop(rewriter, location, *nestedIndexVar, batchIndices, treeIndices, state);
    }
    rewriter.setInsertionPointAfter(loopOp);
  }    
  
  void GenerateLoop(
    ConversionPatternRewriter &rewriter,
    Location location,
    const decisionforest::IndexVariable& indexVar,
    std::list<Value> batchIndices,
    std::list<Value> treeIndices,
    PredictOpLoweringState& state) const {
    // This assert should be removed once we start supporting code generation for tiled loops
    // assert (indexVar.GetParentModifier() == nullptr);
    // Any index in the actual loop nest should not have indices derived from it
    assert (indexVar.GetIndexModifier() == nullptr);

    // Generate all the nested loops
    if (indexVar.GetContainedLoops().empty()) {
      GenerateLeafLoop(rewriter, location, indexVar, batchIndices, treeIndices, state);
    }
    else {
      if (indexVar.Unroll()) {
        GenerateUnrolledLoop(rewriter, location, indexVar, batchIndices, treeIndices, state);
      }
      else {
        GenerateSingleLoop(rewriter, location, indexVar, batchIndices, treeIndices, state);
      }
    }
  }

  LogicalResult
  LowerPredictForestOp_Schedule(Operation *op, mlir::decisionforest::PredictForestOp forestOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter, 
                            mlir::MemRefType dataMemrefType, int64_t batchSize) const
  {
    PredictOpLoweringState state;

    auto location = op->getLoc();
    assert (dataMemrefType.getElementType().isa<mlir::FloatType>());

    // First initialize the result memref to zeros (This is not always needed, for example for the default schedule, but leaving that optimization out for now)
    InitPredictOpLoweringState(rewriter, location, state, forestOp, operands, dataMemrefType, batchSize);
    InitializeResultMemref(rewriter, location, state);
    InitializeTreeClassWeightsMemref(rewriter, location, state);

    auto scheduleAttribute = forestOp.schedule();
    auto& schedule = *scheduleAttribute.GetSchedule();

    // Generate the loop nest
    auto rootIndex = schedule.GetRootIndex();
    assert (rootIndex);
    for (auto index : rootIndex->GetContainedLoops())
      GenerateLoop(rewriter, location, *index, std::list<Value>{}, std::list<Value>{}, state);

    // Generate the transformations to compute final prediction (sigmoid etc)
    TransformResultMemref(rewriter, location, forestOp.ensemble().GetDecisionForest().GetPredictionTransformation(), state);
    rewriter.replaceOp(op, static_cast<Value>(state.resultMemref));
    return mlir::success();
  }

};

struct HighLevelIRToMidLevelIRLoweringPass: public PassWrapper<HighLevelIRToMidLevelIRLoweringPass, OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnOperation() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<memref::MemRefDialect, scf::SCFDialect, 
                           decisionforest::DecisionForestDialect, math::MathDialect,
                           arith::ArithmeticDialect, func::FuncDialect>();

    target.addIllegalOp<decisionforest::PredictForestOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<PredictForestOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir
{
namespace decisionforest
{

void AddWalkDecisionTreeOpLoweringPass(mlir::OpPassManager &optPM);

void LowerFromHighLevelToMidLevelIR(mlir::MLIRContext& context, mlir::ModuleOp module) {
  // Lower from high-level IR to mid-level IR
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(std::make_unique<HighLevelIRToMidLevelIRLoweringPass>());
  AddWalkDecisionTreeOpLoweringPass(optPM);

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to mid level IR failed.\n";
  }
}

} // decisionforest
} // mlir
