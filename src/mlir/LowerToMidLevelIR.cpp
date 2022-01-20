#include "Dialect.h"
// #include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

  LogicalResult
  LowerPredictForestOp_Batch(Operation *op, mlir::decisionforest::PredictForestOp forestOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter, 
                            mlir::MemRefType dataMemrefType, int64_t batchSize) const
  {
    auto location = op->getLoc();

    // Create the return array (TODO This needs to become a function argument)
    auto resultType = op->getResults()[0].getType();
    auto resultMemrefType = resultType.cast<mlir::MemRefType>();
    // auto resultMemrefType = convertTensorToMemRef(resultTensorType);
    // auto allocOp = rewriter.create<memref::AllocOp>(location, resultMemrefType);

    assert (dataMemrefType.getElementType().isa<mlir::FloatType>());
    // auto floatConstZero = rewriter.create<ConstantFloatOp>(location, llvm::APFloat(0.0), resultMemrefType.getElementType().cast<mlir::FloatType>());
    // std::vector<Value> values(batchSize, static_cast<Value>(floatConstZero));
    auto memrefResult = operands[1]; // rewriter.create<tensor::FromElementsOp>(location, resultMemrefType.getElementType(), values);
    // The input data on which we need to run inference
    auto data = operands[0];

    // Create the decision tree constant
    auto forestAttribute = forestOp.ensemble(); // Get the ensemble attribute
    auto forestType = forestAttribute.getType().cast<mlir::decisionforest::TreeEnsembleType>();
    auto forestConst = rewriter.create<mlir::decisionforest::EnsembleConstantOp>(location, forestType, forestAttribute);

    // Create a for loop over the inputs
    auto batchSizeConst = rewriter.create<arith::ConstantIndexOp>(location, batchSize); 
    auto zeroConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
    auto oneIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 1);
    auto batchLoop = rewriter.create<scf::ForOp>(location, zeroConst, batchSizeConst, oneIndexConst/*, static_cast<Value>(memrefResult)*/);
    
    rewriter.setInsertionPointToStart(batchLoop.getBody());
    auto i = batchLoop.getInductionVar();

    // Extract the slice of the input tensor for the current iteration -- row i
    auto rowType = getRowTypeFromArgumentType(dataMemrefType);
    auto zeroIndexAttr = rewriter.getIndexAttr(0);
    auto oneIndexAttr = rewriter.getIndexAttr(1);
    auto rowSizeAttr = rewriter.getIndexAttr(rowType.getShape()[0]);
    auto row = rewriter.create<memref::SubViewOp>(location, static_cast<Value>(data), ArrayRef<OpFoldResult>({i, zeroIndexAttr}),
                                                  ArrayRef<OpFoldResult>({oneIndexAttr, rowSizeAttr}), ArrayRef<OpFoldResult>({oneIndexAttr, oneIndexAttr}));
    if (decisionforest::InsertDebugHelpers) {
      rewriter.create<decisionforest::PrintInputRowOp>(location, row, i);
    }

    // Create a for loop over the trees
    auto resultElementType = resultMemrefType.getElementType();
    int64_t numTrees = static_cast<int64_t>(forestType.getNumberOfTrees());
    auto ensembleSizeConst = rewriter.create<arith::ConstantIndexOp>(location, numTrees); 
    
    Value initialValueConst;
    double_t initialValueValue = forestAttribute.GetDecisionForest().GetInitialOffset();

    if (resultElementType.isa<mlir::Float64Type>())
      initialValueConst = rewriter.create<arith::ConstantFloatOp>(location, llvm::APFloat(initialValueValue), resultElementType.cast<FloatType>());
    else if(resultElementType.isa<mlir::Float32Type>())
      initialValueConst = rewriter.create<arith::ConstantFloatOp>(location, llvm::APFloat((float)initialValueValue), resultElementType.cast<FloatType>());
    else
      assert(false && "Unsupported floating point type");
    // auto oneIndexConst2 = rewriter.create<ConstantIndexOp>(location, 1);
            
    auto treeLoop = rewriter.create<scf::ForOp>(
      location,
      zeroConst,
      ensembleSizeConst,
      oneIndexConst,
      static_cast<Value>(initialValueConst));
    
    rewriter.setInsertionPointToStart(treeLoop.getBody());
    auto j = treeLoop.getInductionVar();

    // TODO The forest shouldn't contain the tree type because each tree may have a different type, 
    // but the tree type should have more information -- tiling for example. We need to construct 
    // the default tiling here.
    assert (forestType.doAllTreesHaveSameTileSize());
    auto treeType = forestType.getTreeType(0).cast<mlir::decisionforest::TreeType>();
    auto tree = rewriter.create<decisionforest::GetTreeFromEnsembleOp>(location, treeType, forestConst, j);

    
    // Create the while loop to walk the tree
    auto walkOp = rewriter.create<decisionforest::WalkDecisionTreeOp>(location, treeType.getResultType(), tree, row);

    // rewriter.create<memref::StoreOp>(location, TypeRange({ }), static_cast<Value>(treePrediction), memrefResult, j);
    // result[i]
    
    // auto readResultOfi = rewriter.create<memref::LoadOp>(location, resultElementType, memrefResult, i);
    // Accumulate the tree prediction
    assert(forestType.getReductionType() == decisionforest::ReductionType::kAdd);
    auto accumulatedValue = rewriter.create<arith::AddFOp>(location, resultElementType, treeLoop.getBody()->getArguments()[1], walkOp);

    if (mlir::decisionforest::InsertDebugHelpers) {
      rewriter.create<decisionforest::PrintTreePredictionOp>(location, walkOp, j);
    }
    // auto updatedResultTensor = rewriter.create<tensor::InsertOp>(location, resultMemrefType, accumulatedValue, treeLoop.getBody()->getArguments()[1], i);
    rewriter.create<scf::YieldOp>(location, static_cast<Value>(accumulatedValue));

    rewriter.setInsertionPointAfter(treeLoop);
    // result[i] = Accumulated value
    auto predTransform = forestAttribute.GetDecisionForest().GetPredictionTransformation();
    Value transformedValue;
    if (predTransform == decisionforest::PredictionTransformation::kIdentity)
      transformedValue = static_cast<Value>(treeLoop.results()[0]);
    else if (predTransform == decisionforest::PredictionTransformation::kSigmoid) {
      transformedValue = GenSigmoid(rewriter, static_cast<Value>(treeLoop.results()[0]), location);
    }
    else {
      assert (false && "Unsupported prediction transformation type");
    }
    rewriter.create<memref::StoreOp>(location, TypeRange({ }), transformedValue, memrefResult, i);
    // rewriter.create<scf::YieldOp>(location); //, static_cast<Value>(treeLoop.results()[0]));

    rewriter.replaceOp(op, static_cast<Value>(memrefResult));
    return mlir::success();
  }

  Value CreateFPConstant(ConversionPatternRewriter &rewriter, Location location, Type type, double value) const {
    Value constValue;
    if (type.isa<mlir::Float64Type>())
      constValue = rewriter.create<arith::ConstantFloatOp>(location, llvm::APFloat(value), type.cast<FloatType>());
    else if(type.isa<mlir::Float32Type>())
      constValue = rewriter.create<arith::ConstantFloatOp>(location, llvm::APFloat((float)value), type.cast<FloatType>());
    else
      assert(false && "Unsupported floating point type");
    return constValue;
  }

  void InitializeResultMemref(Value resultMemref, MemRefType resultMemrefType, int64_t batchSize, double initialValue, 
                              ConversionPatternRewriter &rewriter, Location location) const {
    assert (resultMemrefType.getElementType().isa<mlir::FloatType>());
    
    // Create a for loop over the outputs
    auto batchSizeConst = rewriter.create<arith::ConstantIndexOp>(location, batchSize); 
    auto zeroConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
    auto oneIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 1);
    auto batchLoop = rewriter.create<scf::ForOp>(location, zeroConst, batchSizeConst, oneIndexConst);
    
    rewriter.setInsertionPointToStart(batchLoop.getBody());
    auto i = batchLoop.getInductionVar();
    
    Value initialValueConst = CreateFPConstant(rewriter, location, resultMemrefType.getElementType(), initialValue);

    rewriter.create<memref::StoreOp>(location, initialValueConst, resultMemref, i);
    rewriter.setInsertionPointAfter(batchLoop);
  }

  void TransformResultMemref(Value resultMemref, MemRefType resultMemrefType, int64_t batchSize, 
                             decisionforest::PredictionTransformation predTransform,
                             ConversionPatternRewriter &rewriter, Location location) const {
    
    if (predTransform == decisionforest::PredictionTransformation::kIdentity)
      return;

    assert (resultMemrefType.getElementType().isa<mlir::FloatType>());
    assert (predTransform == decisionforest::PredictionTransformation::kSigmoid);

    // Create a for loop over the outputs
    auto batchSizeConst = rewriter.create<arith::ConstantIndexOp>(location, batchSize); 
    auto zeroConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
    auto oneIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 1);
    auto batchLoop = rewriter.create<scf::ForOp>(location, zeroConst, batchSizeConst, oneIndexConst);
    
    rewriter.setInsertionPointToStart(batchLoop.getBody());
    auto i = batchLoop.getInductionVar();
    
    auto memrefElem = rewriter.create<memref::LoadOp>(location, resultMemref, i);
    auto transformedValue = GenSigmoid(rewriter, static_cast<Value>(memrefElem), location);
    rewriter.create<memref::StoreOp>(location, transformedValue, resultMemref, i);
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

  Value GenerateTreeIndexLeafLoopBody(ConversionPatternRewriter &rewriter, Location location, const decisionforest::IndexVariable& indexVar, 
                        std::list<Value> treeIndices, MemRefType resultMemrefType, Value forestConst, Value row, Value prevAccumulatorValue) const {
   
    Value treeIndex = SumOfValues(rewriter, location, treeIndices);

    // Get the current tree
    auto forestType = forestConst.getType().cast<decisionforest::TreeEnsembleType>();
    assert (forestType.doAllTreesHaveSameTileSize()); // TODO how do we check which type of tree we'll get here?
    auto treeType = forestType.getTreeType(0).cast<mlir::decisionforest::TreeType>();
    auto tree = rewriter.create<decisionforest::GetTreeFromEnsembleOp>(location, treeType, forestConst, treeIndex);

    // Walk the tree
    auto walkOp = rewriter.create<decisionforest::WalkDecisionTreeOp>(location, treeType.getResultType(), tree, row);

    // Accumulate the tree prediction
    assert(forestType.getReductionType() == decisionforest::ReductionType::kAdd);
    auto accumulatedValue = rewriter.create<arith::AddFOp>(location, resultMemrefType.getElementType(), prevAccumulatorValue, walkOp);

    if (mlir::decisionforest::InsertDebugHelpers) {
      rewriter.create<decisionforest::PrintTreePredictionOp>(location, walkOp, treeIndex);
    }
    // auto updatedResultTensor = rewriter.create<tensor::InsertOp>(location, resultMemrefType, accumulatedValue, treeLoop.getBody()->getArguments()[1], i);
    return accumulatedValue;
  }                          


  void GenerateLeafLoopForTreeIndex(ConversionPatternRewriter &rewriter, Location location, const decisionforest::IndexVariable& indexVar, 
                        std::list<Value> batchIndices, std::list<Value> treeIndices, Value resultMemref, MemRefType resultMemrefType,
                        Value data, MemRefType dataMemrefType, Value forestConst) const {
    
    assert (indexVar.GetType() == decisionforest::IndexVariable::IndexVariableType::kTree);
    
    Value rowIndex = SumOfValues(rewriter, location, batchIndices);
    
    // Get the current row
    Value row = GetRow(rewriter, location, data, rowIndex, dataMemrefType);

    if(indexVar.Unroll()) {
      auto range = indexVar.GetRange();
      auto zeroConst = CreateFPConstant(rewriter, location, resultMemrefType.getElementType(), 0.0);      
      Value accumulatedValue = zeroConst;
      for (int32_t i=range.m_start ; i<range.m_stop ; i+=range.m_step) {
        auto treeIndex = rewriter.create<arith::ConstantIndexOp>(location, i);
        treeIndices.push_back(treeIndex);  
        accumulatedValue = GenerateTreeIndexLeafLoopBody(rewriter, location, indexVar, treeIndices, resultMemrefType, forestConst, row, accumulatedValue);
      }
      // Generate the store back in to the result memref
      auto currentMemrefElem = rewriter.create<memref::LoadOp>(location, resultMemref, ValueRange{rowIndex});
      auto newMemrefElem = rewriter.create<arith::AddFOp>(location, resultMemrefType.getElementType(), accumulatedValue, currentMemrefElem);
      rewriter.create<memref::StoreOp>(location, newMemrefElem, resultMemref, ValueRange{rowIndex});
    }
    else {
      // Generate leaf loop for tree index var
      auto range = indexVar.GetRange();
      auto stopConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_stop); 
      auto startConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_start);
      auto stepConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_step);

      auto zeroConst = CreateFPConstant(rewriter, location, resultMemrefType.getElementType(), 0.0);      

      scf::ForOp loop = rewriter.create<scf::ForOp>(location, startConst, stopConst, stepConst, ValueRange{ zeroConst });
      rewriter.setInsertionPointToStart(loop.getBody());
      treeIndices.push_back(loop.getInductionVar());
      auto accumulatedValue = GenerateTreeIndexLeafLoopBody(rewriter, location, indexVar, treeIndices, resultMemrefType, forestConst, row, loop.getBody()->getArguments()[1]);
      rewriter.create<scf::YieldOp>(location, static_cast<Value>(accumulatedValue));
      rewriter.setInsertionPointAfter(loop);
      
      // Generate the store back in to the result memref
      auto currentMemrefElem = rewriter.create<memref::LoadOp>(location, resultMemref, ValueRange{rowIndex});
      auto newMemrefElem = rewriter.create<arith::AddFOp>(location, resultMemrefType.getElementType(), loop.getResults()[0], currentMemrefElem);
      rewriter.create<memref::StoreOp>(location, newMemrefElem, resultMemref, ValueRange{rowIndex});
    }
  }

  void GenerateBatchIndexLeafLoopBody(ConversionPatternRewriter &rewriter, Location location, const decisionforest::IndexVariable& indexVar, 
                        std::list<Value> batchIndices, Value resultMemref, MemRefType resultMemrefType,
                        Value data, MemRefType dataMemrefType, decisionforest::TreeType treeType, Value tree, Value treeIndex) const {
    Value rowIndex = SumOfValues(rewriter, location, batchIndices);
    
    // Get the current row
    Value row = GetRow(rewriter, location, data, rowIndex, dataMemrefType);

    // Walk the tree
    auto walkOp = rewriter.create<decisionforest::WalkDecisionTreeOp>(location, treeType.getResultType(), tree, row);

    // Accumulate the tree prediction and generate the store back in to the result memref
    auto currentMemrefElem = rewriter.create<memref::LoadOp>(location, resultMemref, ValueRange{rowIndex});
    auto accumulatedValue = rewriter.create<arith::AddFOp>(location, resultMemrefType.getElementType(), static_cast<Value>(walkOp), currentMemrefElem);
    rewriter.create<memref::StoreOp>(location, accumulatedValue, resultMemref, ValueRange{rowIndex});

    if (mlir::decisionforest::InsertDebugHelpers) {
      rewriter.create<decisionforest::PrintTreePredictionOp>(location, walkOp, treeIndex);
    }
  }

  void GenerateLeafLoopForBatchIndex(ConversionPatternRewriter &rewriter, Location location, const decisionforest::IndexVariable& indexVar, 
                        std::list<Value> batchIndices, std::list<Value> treeIndices, Value resultMemref, MemRefType resultMemrefType,
                        Value data, MemRefType dataMemrefType, Value forestConst) const {

    assert (indexVar.GetType() == decisionforest::IndexVariable::IndexVariableType::kBatch);

    Value treeIndex = SumOfValues(rewriter, location, treeIndices);

    // Get the current tree
    auto forestType = forestConst.getType().cast<decisionforest::TreeEnsembleType>();
    assert (forestType.doAllTreesHaveSameTileSize()); // TODO how do we check which type of tree we'll get here?
    assert(forestType.getReductionType() == decisionforest::ReductionType::kAdd);
    auto treeType = forestType.getTreeType(0).cast<mlir::decisionforest::TreeType>();
    auto tree = rewriter.create<decisionforest::GetTreeFromEnsembleOp>(location, treeType, forestConst, treeIndex);

    if (indexVar.Unroll()) {
      auto range = indexVar.GetRange();
      for (int32_t i=range.m_start ; i<range.m_stop ; i+=range.m_step) {
        auto batchIndex = rewriter.create<arith::ConstantIndexOp>(location, i);
        batchIndices.push_back(batchIndex);
        GenerateBatchIndexLeafLoopBody(rewriter, location, indexVar, batchIndices, resultMemref, resultMemrefType, data, dataMemrefType, treeType, tree, treeIndex);
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
      GenerateBatchIndexLeafLoopBody(rewriter, location, indexVar, batchIndices, resultMemref, resultMemrefType, data, dataMemrefType, treeType, tree, treeIndex);
      rewriter.setInsertionPointAfter(loop);
    }
  }

  void GenerateLeafLoop(ConversionPatternRewriter &rewriter, Location location, const decisionforest::IndexVariable& indexVar, 
                        std::list<Value> batchIndices, std::list<Value> treeIndices, Value resultMemref, MemRefType resultMemrefType,
                        Value data, MemRefType dataMemrefType, Value forestConst) const {
    if (indexVar.GetType() == decisionforest::IndexVariable::IndexVariableType::kTree) {    
      GenerateLeafLoopForTreeIndex(rewriter, location, indexVar, batchIndices, treeIndices, resultMemref, resultMemrefType, data, dataMemrefType, forestConst);
    }
    else {
      GenerateLeafLoopForBatchIndex(rewriter, location, indexVar, batchIndices, treeIndices, resultMemref, resultMemrefType, data, dataMemrefType, forestConst);
    }
  }

  void GenerateUnrolledLoop(ConversionPatternRewriter &rewriter, Location location, const decisionforest::IndexVariable& indexVar, 
                        std::list<Value> batchIndices, std::list<Value> treeIndices, Value resultMemref, MemRefType resultMemrefType,
                        Value data, MemRefType dataMemrefType, Value forestConst) const {
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
        GenerateLoop(rewriter, location, *nestedIndexVar, batchIndices, treeIndices, resultMemref, resultMemrefType, data, dataMemrefType, forestConst);
      }
    }
  }

  void GenerateSingleLoop(ConversionPatternRewriter &rewriter, Location location, const decisionforest::IndexVariable& indexVar, 
                    std::list<Value> batchIndices, std::list<Value> treeIndices, Value resultMemref, MemRefType resultMemrefType,
                    Value data, MemRefType dataMemrefType, Value forestConst) const {
    auto range = indexVar.GetRange();
    auto stopConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_stop); 
    auto startConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_start);
    auto stepConst = rewriter.create<arith::ConstantIndexOp>(location, range.m_step);

    auto loop = rewriter.create<scf::ForOp>(location, startConst, stopConst, stepConst);
  
    rewriter.setInsertionPointToStart(loop.getBody());
    auto i = loop.getInductionVar();

    if (indexVar.GetType() == decisionforest::IndexVariable::IndexVariableType::kBatch)
      batchIndices.push_back(i);
    else if (indexVar.GetType() == decisionforest::IndexVariable::IndexVariableType::kTree)
      treeIndices.push_back(i);
    else
      assert (false && "Unknown index variable type!");

    for (auto nestedIndexVar : indexVar.GetContainedLoops()) {
      GenerateLoop(rewriter, location, *nestedIndexVar, batchIndices, treeIndices, resultMemref, resultMemrefType, data, dataMemrefType, forestConst);
    }
    rewriter.setInsertionPointAfter(loop);
  }    
  
  void GenerateLoop(ConversionPatternRewriter &rewriter, Location location, const decisionforest::IndexVariable& indexVar, 
                    std::list<Value> batchIndices, std::list<Value> treeIndices, Value resultMemref, MemRefType resultMemrefType,
                    Value data, MemRefType dataMemrefType, Value forestConst) const {
    // This assert should be removed once we start supporting code generation for tiled loops
    // assert (indexVar.GetParentModifier() == nullptr);
    // Any index in the actual loop nest should not have indices derived from it
    assert (indexVar.GetIndexModifier() == nullptr);

    // Generate all the nested loops
    if (indexVar.GetContainedLoops().empty()) {
      GenerateLeafLoop(rewriter, location, indexVar, batchIndices, treeIndices, resultMemref, resultMemrefType, data, dataMemrefType, forestConst);
    }
    else {
      if (indexVar.Unroll()) {
        GenerateUnrolledLoop(rewriter, location, indexVar, batchIndices, treeIndices, resultMemref, resultMemrefType, data, dataMemrefType, forestConst);
      }
      else {
        GenerateSingleLoop(rewriter, location, indexVar, batchIndices, treeIndices, resultMemref, resultMemrefType, data, dataMemrefType, forestConst);
      }
    }
  }

  LogicalResult
  LowerPredictForestOp_Schedule(Operation *op, mlir::decisionforest::PredictForestOp forestOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter, 
                            mlir::MemRefType dataMemrefType, int64_t batchSize) const
  {
    auto location = op->getLoc();

    auto resultType = op->getResults()[0].getType();
    auto resultMemrefType = resultType.cast<mlir::MemRefType>();
    assert (resultMemrefType);
    auto resultMemref = operands[1]; 

    assert (dataMemrefType.getElementType().isa<mlir::FloatType>());
    
    // Create the decision tree constant
    auto forestAttribute = forestOp.ensemble(); // Get the ensemble attribute
    auto forestType = forestAttribute.getType().cast<mlir::decisionforest::TreeEnsembleType>();
    auto forestConst = rewriter.create<mlir::decisionforest::EnsembleConstantOp>(location, forestType, forestAttribute);

    // First initialize the result memref to zeros (This is not always needed, for example for the default schedule, but leaving that optimization out for now)
    InitializeResultMemref(resultMemref, resultMemrefType, batchSize, forestAttribute.GetDecisionForest().GetInitialOffset(),  rewriter, location);

    // The input data on which we need to run inference
    auto data = operands[0];
    auto scheduleAttribute = forestOp.schedule();
    auto& schedule = *scheduleAttribute.GetSchedule();

    // Generate the loop nest
    auto rootIndex = schedule.GetRootIndex();
    assert (rootIndex);
    for (auto index : rootIndex->GetContainedLoops())
      GenerateLoop(rewriter, location, *index, std::list<Value>{}, std::list<Value>{}, resultMemref, resultMemrefType, data, dataMemrefType, forestConst);

    // Generate the transformations to compute final prediction (sigmoid etc)
    TransformResultMemref(resultMemref, resultMemrefType, batchSize, forestAttribute.GetDecisionForest().GetPredictionTransformation(), 
                          rewriter, location);
    rewriter.replaceOp(op, static_cast<Value>(resultMemref));
    return mlir::success();
  }

};

struct HighLevelIRToMidLevelIRLoweringPass: public PassWrapper<HighLevelIRToMidLevelIRLoweringPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, StandardOpsDialect, scf::SCFDialect>();
  }
  void runOnFunction() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<memref::MemRefDialect, StandardOpsDialect, scf::SCFDialect, 
                           decisionforest::DecisionForestDialect, math::MathDialect, arith::ArithmeticDialect>();

    target.addIllegalOp<decisionforest::PredictForestOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<PredictForestOpLowering>(&getContext());

    if (failed(applyPartialConversion(getFunction(), target, std::move(patterns))))
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
  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  optPM.addPass(std::make_unique<HighLevelIRToMidLevelIRLoweringPass>());
  AddWalkDecisionTreeOpLoweringPass(optPM);

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to mid level IR failed.\n";
  }
}

} // decisionforest
} // mlir