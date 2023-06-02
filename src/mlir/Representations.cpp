#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "Dialect.h"
#include "../gpu/GPURepresentations.h"
#include "LIRLoweringHelpers.h"
#include "Logger.h"
#include "OpLoweringUtils.h"
#include "Representations.h"

using namespace mlir;
using namespace mlir::decisionforest::helpers;

namespace
{
const int32_t kAlignedPointerIndexInMemrefStruct = 1;
const int32_t kOffsetIndexInMemrefStruct = 2;
const int32_t kThresholdElementNumberInTile = 0;
const int32_t kFeatureIndexElementNumberInTile = 1;
const int32_t kTileShapeElementNumberInTile = 2;
const int32_t kChildIndexElementNumberInTile = 3;

Type generateGetElementPtr(Operation *op, ArrayRef<Value> operands,
                           ConversionPatternRewriter &rewriter,
                           Type elementMLIRType, int64_t elementNumber,
                           TypeConverter *typeConverter, Value &elementPtr) {
  const int32_t kTreeMemrefOperandNum = 0;
  const int32_t kIndexOperandNum = 1;
  auto location = op->getLoc();
  
  auto memrefType = operands[kTreeMemrefOperandNum].getType();
  auto memrefStructType = memrefType.cast<LLVM::LLVMStructType>();
  auto alignedPtrType = memrefStructType.getBody()[kAlignedPointerIndexInMemrefStruct].cast<LLVM::LLVMPointerType>();
  auto tileType = alignedPtrType.getElementType().cast<LLVM::LLVMStructType>();
  
  auto indexVal = operands[kIndexOperandNum];
  auto indexType = indexVal.getType();
  assert (indexType.isa<IntegerType>());
  
  auto elementType = typeConverter->convertType(elementMLIRType);

  // Extract the memref's aligned pointer
  auto extractMemrefBufferPointer = rewriter.create<LLVM::ExtractValueOp>(location, alignedPtrType, operands[kTreeMemrefOperandNum],
                                                                          rewriter.getDenseI64ArrayAttr(kAlignedPointerIndexInMemrefStruct));

  auto extractMemrefOffset = rewriter.create<LLVM::ExtractValueOp>(location, indexType, operands[kTreeMemrefOperandNum],
                                                                   rewriter.getDenseI64ArrayAttr(kOffsetIndexInMemrefStruct));

  auto actualIndex = rewriter.create<LLVM::AddOp>(location, indexType, static_cast<Value>(extractMemrefOffset), static_cast<Value>(indexVal));

  // Get a pointer to i'th tile's threshold
  auto elementPtrType = LLVM::LLVMPointerType::get(elementType, alignedPtrType.getAddressSpace());
  assert(elementType == tileType.getBody()[elementNumber] && "The result type should be the same as the element type in the struct.");
  auto elemIndexConst = rewriter.create<LLVM::ConstantOp>(location, rewriter.getI32Type(), rewriter.getIntegerAttr(rewriter.getI32Type(), elementNumber));
  elementPtr = rewriter.create<LLVM::GEPOp>(location, elementPtrType, static_cast<Value>(extractMemrefBufferPointer), 
                                            ValueRange({static_cast<Value>(actualIndex), static_cast<Value>(elemIndexConst)}));

  // Insert call to print pointers if debug helpers is on
  // if (decisionforest::InsertDebugHelpers)
  //   decisionforest::InsertPrintElementAddressIfNeeded(rewriter, location, op->getParentOfType<ModuleOp>(), 
  //                                                     extractMemrefBufferPointer, indexVal, actualIndex, elemIndexConst, elementPtr);
  
  return elementType;
}

void generateLoadStructElement(Operation *op, ArrayRef<Value> operands,
                               ConversionPatternRewriter &rewriter,
                               int64_t elementNumber,
                               TypeConverter *typeConverter) {

  auto location = op->getLoc();
  Value elementPtr;
  auto elementType =
      generateGetElementPtr(op, operands, rewriter, op->getResult(0).getType(),
                            elementNumber, typeConverter, elementPtr);

  // Load the element
  auto elementVal = rewriter.create<LLVM::LoadOp>(location, elementType, static_cast<Value>(elementPtr));
  
  rewriter.replaceOp(op, static_cast<Value>(elementVal));
}

void generateStoreStructElement(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter, Type elementMLIRType,
                               int64_t elementNumber, TypeConverter* typeConverter, Value elementVal) {
  
  auto location = op->getLoc();
  Value elementPtr;
  generateGetElementPtr(op, operands, rewriter, elementMLIRType, elementNumber,
                        typeConverter, elementPtr);

  // Store the element
  rewriter.create<LLVM::StoreOp>(location, elementVal, elementPtr);
}

struct LoadTileThresholdOpLowering: public ConversionPattern {
  LoadTileThresholdOpLowering(LLVMTypeConverter& typeConverter)
  : ConversionPattern(typeConverter, mlir::decisionforest::LoadTileThresholdsOp::getOperationName(), 1 /*benefit*/, &typeConverter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    assert (operands.size() == 3);
    generateLoadStructElement(op, operands, rewriter,
                              kThresholdElementNumberInTile,
                              getTypeConverter());
    return mlir::success();
  }
};

struct LoadTileFeatureIndicesOpLowering: public ConversionPattern {
  LoadTileFeatureIndicesOpLowering(LLVMTypeConverter& typeConverter) 
  : ConversionPattern(typeConverter, mlir::decisionforest::LoadTileFeatureIndicesOp::getOperationName(), 1 /*benefit*/, &typeConverter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 3);
    generateLoadStructElement(op, operands, rewriter,
                              kFeatureIndexElementNumberInTile,
                              getTypeConverter());
    return mlir::success();
  }
};

struct LoadTileShapeOpLowering : public ConversionPattern {
  LoadTileShapeOpLowering(LLVMTypeConverter& typeConverter) 
  : ConversionPattern(typeConverter, mlir::decisionforest::LoadTileShapeOp::getOperationName(), 1 /*benefit*/, &typeConverter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 3);
    generateLoadStructElement(op, operands, rewriter,
                              kTileShapeElementNumberInTile,
                              getTypeConverter());
    return mlir::success();
  }
};

struct LoadChildIndexOpLowering : public ConversionPattern {
  LoadChildIndexOpLowering(LLVMTypeConverter& typeConverter) 
  : ConversionPattern(typeConverter, mlir::decisionforest::LoadChildIndexOp::getOperationName(), 1 /*benefit*/, &typeConverter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 2);
    generateLoadStructElement(op, operands, rewriter,
                              kChildIndexElementNumberInTile,
                              getTypeConverter());
    return mlir::success();
  }
};

struct InitTileOpLowering : public ConversionPattern {
  InitTileOpLowering(LLVMTypeConverter& typeConverter) 
  : ConversionPattern(typeConverter, mlir::decisionforest::InitTileOp::getOperationName(), 1 /*benefit*/, &typeConverter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 5);
    decisionforest::InitTileOpAdaptor tileOpAdaptor(operands);
    generateStoreStructElement(op, operands, rewriter, tileOpAdaptor.getThresholds().getType(), 0, getTypeConverter(), tileOpAdaptor.getThresholds());
    generateStoreStructElement(op, operands, rewriter, tileOpAdaptor.getFeatureIndices().getType(), 1, getTypeConverter(), tileOpAdaptor.getFeatureIndices());
    auto modelMemrefType = op->getOperand(0).getType().cast<MemRefType>();
    auto tileType = modelMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
    if (tileType.getTileSize() > 1)
      generateStoreStructElement(op, operands, rewriter, tileOpAdaptor.getTileShapeID().getType(), 2, getTypeConverter(), tileOpAdaptor.getTileShapeID());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct InitSparseTileOpLowering : public ConversionPattern {
  InitSparseTileOpLowering(LLVMTypeConverter& typeConverter) 
  : ConversionPattern(typeConverter, mlir::decisionforest::InitSparseTileOp::getOperationName(), 1 /*benefit*/, &typeConverter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 6);
    decisionforest::InitSparseTileOpAdaptor tileOpAdaptor(operands);
    generateStoreStructElement(op, operands, rewriter, tileOpAdaptor.getThresholds().getType(), 0, getTypeConverter(), tileOpAdaptor.getThresholds());
    generateStoreStructElement(op, operands, rewriter, tileOpAdaptor.getFeatureIndices().getType(), 1, getTypeConverter(), tileOpAdaptor.getFeatureIndices());
    auto modelMemrefType = op->getOperand(0).getType().cast<MemRefType>();
    auto tileType = modelMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
    if (tileType.getTileSize() > 1)
      generateStoreStructElement(op, operands, rewriter, tileOpAdaptor.getTileShapeID().getType(), 2, getTypeConverter(), tileOpAdaptor.getTileShapeID());
    generateStoreStructElement(op, operands, rewriter, tileOpAdaptor.getChildIndex().getType(), 3, getTypeConverter(), tileOpAdaptor.getChildIndex());

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct GetModelMemrefSizeOpLowering : public ConversionPattern {
  GetModelMemrefSizeOpLowering(LLVMTypeConverter& typeConverter) 
  : ConversionPattern(typeConverter, mlir::decisionforest::GetModelMemrefSizeOp::getOperationName(), 1 /*benefit*/, &typeConverter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    const int32_t kTreeMemrefOperandNum = 0;
    const int32_t kLengthOperandNum = 1;
    auto location = op->getLoc();
    
    auto memrefType = operands[kTreeMemrefOperandNum].getType();
    auto memrefStructType = memrefType.cast<LLVM::LLVMStructType>();
    auto alignedPtrType = memrefStructType.getBody()[kAlignedPointerIndexInMemrefStruct].cast<LLVM::LLVMPointerType>();
    
    auto indexVal = operands[kLengthOperandNum];
    auto indexType = indexVal.getType();
    assert (indexType.isa<IntegerType>());
    
    // Extract the memref's aligned pointer
    auto extractMemrefBufferPointer = rewriter.create<LLVM::ExtractValueOp>(location, alignedPtrType, operands[kTreeMemrefOperandNum],
                                                                            rewriter.getDenseI64ArrayAttr(kAlignedPointerIndexInMemrefStruct));

    auto extractMemrefOffset = rewriter.create<LLVM::ExtractValueOp>(location, indexType, operands[kTreeMemrefOperandNum],
                                                                    rewriter.getDenseI64ArrayAttr(kOffsetIndexInMemrefStruct));

    auto actualIndex = rewriter.create<LLVM::AddOp>(location, indexType, static_cast<Value>(extractMemrefOffset), static_cast<Value>(indexVal));

    auto endPtr = rewriter.create<LLVM::GEPOp>(location, alignedPtrType, static_cast<Value>(extractMemrefBufferPointer), 
                                               ValueRange({static_cast<Value>(actualIndex)}));

    auto buffPtrInt = rewriter.create<LLVM::PtrToIntOp>(location, indexType, extractMemrefBufferPointer);
    auto endPtrInt = rewriter.create<LLVM::PtrToIntOp>(location, indexType, endPtr);
    auto sizeInBytes = rewriter.create<LLVM::SubOp>(location, indexType, endPtrInt, buffPtrInt);
    auto sizeInBytesI32 = rewriter.create<LLVM::TruncOp>(location, rewriter.getI32Type(), sizeInBytes);
    rewriter.replaceOp(op, static_cast<Value>(sizeInBytesI32));
    return mlir::success();
  }
};

// TODO_Ashwin This is just a hacky caching implementation. 
// We can't just use a memref.subview since that would lead 
// to the types of the cacheInputRowsOp and the replacement 
// subview op not matching. Therefore, I'm allocating a new
// buffer with the right type and copying the subview into 
// it and replacing the cache op with the new allocation.
void LowerCacheRowsOpToCPU(ConversionPatternRewriter &rewriter,
                           mlir::Operation *op,
                           ArrayRef<Value> operands) {
  auto location = op->getLoc();
  auto cacheInputOp = AssertOpIsOfType<decisionforest::CacheInputRowsOp>(op);
  decisionforest::CacheInputRowsOpAdaptor cacheInputOpAdaptor(operands);
  auto resultType = cacheInputOp.getResult().getType();
  auto resultMemrefType = resultType.cast<MemRefType>();

  auto zeroConst = rewriter.getIndexAttr(0);
  auto oneConst = rewriter.getIndexAttr(1);
  auto numRows = rewriter.getIndexAttr(resultMemrefType.getShape()[0]);
  auto numCols = rewriter.getIndexAttr(resultMemrefType.getShape()[1]);
  auto cacheSubview = rewriter.create<memref::SubViewOp>(location, 
          cacheInputOpAdaptor.getData(),
          ArrayRef<OpFoldResult>({cacheInputOpAdaptor.getStartIndex(), zeroConst}), // offsets
          ArrayRef<OpFoldResult>({numRows, numCols}), // sizes
          ArrayRef<OpFoldResult>({oneConst, oneConst}) // strides
          );
  // auto prefetchOp = rewriter.create<memref::PrefetchOp>(location, 
  //         cacheSubview.getResult(),
  //         ValueRange{zeroConst.getResult(), zeroConst.getResult()},
  //         false, //isWrite
  //         (uint32_t)3, // locality hint
  //         true); // data cache
  auto allocCache = rewriter.create<memref::AllocaOp>(location, resultMemrefType);
  rewriter.create<memref::CopyOp>(location, cacheSubview.getResult(), allocCache.getResult());
  rewriter.replaceOp(op, allocCache.getResult());
}


} // anonymous namespace

namespace mlir
{
namespace decisionforest
{
// ===---------------------------------------------------=== //
// Array based representation
// ===---------------------------------------------------=== //

void ArrayBasedRepresentation::InitRepresentation() {
  ensembleConstantToMemrefsMap.clear();
  getTreeOperationMap.clear();
}

mlir::LogicalResult ArrayBasedRepresentation::GenerateModelGlobals(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter,
                                                    std::shared_ptr<decisionforest::IModelSerializer> serializer) {

    mlir::decisionforest::EnsembleConstantOp ensembleConstOp = llvm::dyn_cast<mlir::decisionforest::EnsembleConstantOp>(op);
    assert(ensembleConstOp);
    assert(operands.size() == 0);
    if (!ensembleConstOp)
        return mlir::failure();
    
    auto location = op->getLoc();
    auto owningModule = op->getParentOfType<mlir::ModuleOp>();
    assert (owningModule);
    
    // TODO the names of the model and offset global should be generated so they're unique for each ensemble constant
    // TODO the getter function names need to be persisted with the actual tree values in the JSON so the runtime can call them. 
    std::string modelMemrefName = "model";
    std::string offsetMemrefName = "offsets";
    std::string lengthMemrefName = "lengths";
    std::string classInfoMemrefName = "treeClassInfo";


    auto memrefTypes = AddGlobalMemrefs(
      owningModule,
      ensembleConstOp,
      rewriter,
      location,
      modelMemrefName,
      offsetMemrefName,
      lengthMemrefName,
      classInfoMemrefName,
      serializer);


    AddModelMemrefInitFunction(owningModule, modelMemrefName, memrefTypes.model.cast<MemRefType>(), rewriter, location);
    auto getModelGlobal = rewriter.create<memref::GetGlobalOp>(location, memrefTypes.model, modelMemrefName);
    auto getOffsetGlobal = rewriter.create<memref::GetGlobalOp>(location, memrefTypes.offset, offsetMemrefName);
    auto getLengthGlobal = rewriter.create<memref::GetGlobalOp>(location, memrefTypes.offset, lengthMemrefName);
    auto classInfoGlobal = rewriter.create<memref::GetGlobalOp>(location, memrefTypes.classInfo, classInfoMemrefName);

    EnsembleConstantLoweringInfo info 
    {
      static_cast<Value>(getModelGlobal),
      static_cast<Value>(getOffsetGlobal),
      static_cast<Value>(getLengthGlobal),
      static_cast<Value>(classInfoGlobal),
      memrefTypes.model,
      memrefTypes.offset,
      memrefTypes.offset,
      memrefTypes.classInfo,
    };
    ensembleConstantToMemrefsMap[op] = info;
    return mlir::success();
}

ArrayBasedRepresentation::GlobalMemrefTypes ArrayBasedRepresentation::AddGlobalMemrefs(
  mlir::ModuleOp module,
  mlir::decisionforest::EnsembleConstantOp& ensembleConstOp,
  ConversionPatternRewriter &rewriter,
  Location location,
  const std::string& modelMemrefName,
  const std::string& offsetMemrefName,
  const std::string& lengthMemrefName,
  const std::string& treeInfo,
  std::shared_ptr<decisionforest::IModelSerializer> serializer) {
  mlir::decisionforest::DecisionForestAttribute forestAttribute = ensembleConstOp.getForest();
  mlir::decisionforest::DecisionForest& forest = forestAttribute.GetDecisionForest();

  SaveAndRestoreInsertionPoint saveAndRestoreInsertPoint(rewriter);
  rewriter.setInsertionPoint(&module.front());

  auto forestType = ensembleConstOp.getResult().getType().cast<decisionforest::TreeEnsembleType>();
  assert (forestType.doAllTreesHaveSameTileSize()); // There is still an assumption here that all trees have the same tile size
  auto treeType = forestType.getTreeType(0).cast<decisionforest::TreeType>();

  m_thresholdType = treeType.getThresholdType();
  m_featureIndexType = treeType.getFeatureIndexType(); 
  auto tileSize = treeType.getTileSize();
  m_tileShapeType = treeType.getTileShapeType();
  // assert (tileSize == 1);
  Type memrefElementType = decisionforest::TiledNumericalNodeType::get(m_thresholdType, m_featureIndexType, m_tileShapeType, tileSize);
  
  m_tileSize = tileSize;
  serializer->Persist(forest, forestType);
  
  auto modelMemrefSize = decisionforest::GetTotalNumberOfTiles();
  auto modelMemrefType = MemRefType::get({modelMemrefSize}, memrefElementType);
  rewriter.create<memref::GlobalOp>(location, modelMemrefName,
                                    /*sym_visibility=*/rewriter.getStringAttr("private"),
                                    /*type=*/modelMemrefType,
                                    /*initial_value=*/rewriter.getUnitAttr(),
                                    /*constant=*/false, IntegerAttr());
  AddGlobalMemrefGetter(module, modelMemrefName, modelMemrefType, rewriter, location);
  
  auto offsetSize = (int32_t)forest.NumTrees();
  auto offsetMemrefType = MemRefType::get({offsetSize}, rewriter.getIndexType());
  rewriter.create<memref::GlobalOp>(location, offsetMemrefName, rewriter.getStringAttr("private"),
                                    offsetMemrefType, rewriter.getUnitAttr(), false, IntegerAttr());
  AddGlobalMemrefGetter(module, offsetMemrefName, offsetMemrefType, rewriter, location);
  
  rewriter.create<memref::GlobalOp>(location, lengthMemrefName, rewriter.getStringAttr("private"),
                                    offsetMemrefType, rewriter.getUnitAttr(), false, IntegerAttr());
  AddGlobalMemrefGetter(module, lengthMemrefName, offsetMemrefType, rewriter, location);

  auto classInfoSize = forest.IsMultiClassClassifier() ? offsetSize : 0;
  auto classInfoMemrefType = MemRefType::get({classInfoSize}, treeType.getResultType());
  rewriter.create<memref::GlobalOp>(
    location,
    treeInfo,
    rewriter.getStringAttr("public"),
    classInfoMemrefType,
    rewriter.getUnitAttr(),
    false,
    IntegerAttr());
  AddGlobalMemrefGetter(module, treeInfo, classInfoMemrefType, rewriter, location);

  return GlobalMemrefTypes { modelMemrefType, offsetMemrefType, classInfoMemrefType };
}

void ArrayBasedRepresentation::GenModelMemrefInitFunctionBody(MemRefType memrefType, Value getGlobalMemref,
                                                              mlir::OpBuilder &builder, Location location, Value tileIndex,
                                                              Value thresholdMemref, Value indexMemref, Value tileShapeIdMemref) {
  auto modelMemrefElementType = memrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
  int32_t tileSize = modelMemrefElementType.getTileSize();

  // index = tileSize * tileIndex
  auto tileSizeConst = builder.create<arith::ConstantIndexOp>(location, tileSize);
  auto tileSizeTimesi = builder.create<arith::MulIOp>(location, tileIndex, tileSizeConst);
  
  if (tileSize > 1) {
    auto thresholdVec = CreateZeroVectorFPConst(builder, location, modelMemrefElementType.getThresholdElementType(), tileSize);
    auto indexVec = CreateZeroVectorIntConst(builder, location, modelMemrefElementType.getIndexElementType(), tileSize);

    // Load from index to index + (tileSize - 1) into a vector
    for (int32_t j = 0 ; j<tileSize ; ++j) {
      auto offset = builder.create<arith::ConstantIndexOp>(location, j);
      auto index =  builder.create<arith::AddIOp>(location, tileSizeTimesi, offset);
      auto thresholdVal = builder.create<memref::LoadOp>(location, thresholdMemref, static_cast<Value>(index));
      auto jConst = builder.create<arith::ConstantIntOp>(location, j, builder.getI32Type());
      thresholdVec = builder.create<vector::InsertElementOp>(location, thresholdVal, thresholdVec, jConst);
      auto indexVal = builder.create<memref::LoadOp>(location, indexMemref, static_cast<Value>(index));
      indexVec = builder.create<vector::InsertElementOp>(location, indexVal, indexVec, jConst);
    }
    auto tileShapeID = builder.create<memref::LoadOp>(location, tileShapeIdMemref, tileIndex);
    builder.create<decisionforest::InitTileOp>(location, getGlobalMemref, tileIndex, thresholdVec, indexVec, tileShapeID);
  }
  else {
    // Load from index to index + (tileSize - 1) into a vector
    auto thresholdVal = builder.create<memref::LoadOp>(location, thresholdMemref, static_cast<Value>(tileIndex));
    auto indexVal = builder.create<memref::LoadOp>(location, indexMemref, static_cast<Value>(tileIndex));
    // TODO check how tileShapeID vector is created when tileSize = 1
    auto tileShapeID = builder.create<arith::ConstantIntOp>(location, 0, builder.getI32Type());
    builder.create<decisionforest::InitTileOp>(location, getGlobalMemref, tileIndex, thresholdVal, indexVal, tileShapeID);
  }
}

void ArrayBasedRepresentation::AddModelMemrefInitFunction(mlir::ModuleOp module, std::string globalName, MemRefType memrefType, 
                                                          ConversionPatternRewriter &rewriter, Location location) {
  assert (memrefType.getShape().size() == 1);
  SaveAndRestoreInsertionPoint saveAndRestoreEntryPoint(rewriter);
  auto modelMemrefElementType = memrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
  int32_t tileSize = modelMemrefElementType.getTileSize();
  auto thresholdArgType = MemRefType::get({ memrefType.getShape()[0] * tileSize }, modelMemrefElementType.getThresholdElementType());
  auto indexArgType = MemRefType::get({ memrefType.getShape()[0] * tileSize }, modelMemrefElementType.getIndexElementType());
  auto tileShapeIDArgType = MemRefType::get(memrefType.getShape(), modelMemrefElementType.getTileShapeType());
  auto getMemrefFuncType = rewriter.getFunctionType(TypeRange{thresholdArgType, indexArgType, tileShapeIDArgType}, rewriter.getI32Type());
  std::string funcName = "Init_" + globalName;
  NamedAttribute visibilityAttribute{module.getSymVisibilityAttrName(), rewriter.getStringAttr("public")};
  auto initModelMemrefFunc = mlir::func::FuncOp::create(location, funcName, getMemrefFuncType, ArrayRef<NamedAttribute>(visibilityAttribute));
  auto &entryBlock = *initModelMemrefFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(&entryBlock);

  // for tileIndex = 0 : len
  auto getGlobalMemref = rewriter.create<memref::GetGlobalOp>(location, memrefType, globalName);

  auto zeroIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
  auto oneIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 1);
  auto lenIndexConst = rewriter.create<arith::ConstantIndexOp>(location, memrefType.getShape()[0]);
  auto forLoop = rewriter.create<scf::ForOp>(location, zeroIndexConst, lenIndexConst, oneIndexConst);
  auto tileIndex = forLoop.getInductionVar();
  rewriter.setInsertionPointToStart(forLoop.getBody());

  GenModelMemrefInitFunctionBody(memrefType, getGlobalMemref, rewriter, location, tileIndex, 
                                entryBlock.getArgument(0), entryBlock.getArgument(1), entryBlock.getArgument(2));

  rewriter.setInsertionPointAfter(forLoop);
  
  auto modelSize = rewriter.create<decisionforest::GetModelMemrefSizeOp>(location, rewriter.getI32Type(), getGlobalMemref, lenIndexConst);
  rewriter.create<mlir::func::ReturnOp>(location, static_cast<Value>(modelSize));

  module.push_back(initModelMemrefFunc);
}

mlir::Value ArrayBasedRepresentation::GetTreeMemref(mlir::Value treeValue) {
  auto *getTreeOp = treeValue.getDefiningOp();
  AssertOpIsOfType<mlir::decisionforest::GetTreeFromEnsembleOp>(getTreeOp);
  auto getTreeOperationMapIter = getTreeOperationMap.find(getTreeOp);
  assert(getTreeOperationMapIter != getTreeOperationMap.end());
  auto treeMemref = getTreeOperationMapIter->second;
  return treeMemref;
}

mlir::Value ArrayBasedRepresentation::GenerateMoveToChild(mlir::Location location, ConversionPatternRewriter &rewriter, mlir::Value nodeIndex,
                                                          mlir::Value childNumber, int32_t tileSize, std::vector<mlir::Value>& extraLoads) {
  auto oneConstant = rewriter.create<arith::ConstantIndexOp>(location, 1);
  auto tileSizeConstant = rewriter.create<arith::ConstantIndexOp>(location, tileSize+1);
  auto tileSizeTimesIndex = rewriter.create<arith::MulIOp>(location, rewriter.getIndexType(), static_cast<Value>(nodeIndex), static_cast<Value>(tileSizeConstant));
  auto tileSizeTimesIndexPlus1 = rewriter.create<arith::AddIOp>(location, rewriter.getIndexType(), static_cast<Value>(tileSizeTimesIndex), static_cast<Value>(oneConstant));
  
  auto newIndex = rewriter.create<arith::AddIOp>(location, rewriter.getIndexType(), tileSizeTimesIndexPlus1, childNumber);
  return newIndex;
}

void ArrayBasedRepresentation::GenerateTreeMemref(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex) {
    // Create a subview of the model memref corresponding to this ensemble with the index equal to offsetMemref[treeIndex]
    auto location = op->getLoc();
    Operation* ensembleConstOp = ensemble.getDefiningOp();
    AssertOpIsOfType<mlir::decisionforest::EnsembleConstantOp>(ensembleConstOp);
    
    auto mapIter = ensembleConstantToMemrefsMap.find(ensembleConstOp);
    assert (mapIter != ensembleConstantToMemrefsMap.end());
    auto& ensembleInfo = mapIter->second;

    auto modelMemrefIndex = rewriter.create<memref::LoadOp>(location, ensembleInfo.offsetGlobal, treeIndex);
    auto treeLength = rewriter.create<memref::LoadOp>(location, ensembleInfo.lengthGlobal, treeIndex);; // TODO Need to put this into the map too
    auto treeMemref = rewriter.create<memref::SubViewOp>(location, ensembleInfo.modelGlobal, ArrayRef<OpFoldResult>({static_cast<Value>(modelMemrefIndex)}),
                                                         ArrayRef<OpFoldResult>({static_cast<Value>(treeLength)}), ArrayRef<OpFoldResult>({rewriter.getIndexAttr(1)}));
    
    // if (decisionforest::InsertDebugHelpers) {
    //   rewriter.create<decisionforest::PrintTreeToDOTFileOp>(location, treeMemref, treeIndex);
    // }
    getTreeOperationMap[op] = static_cast<Value>(treeMemref);
}

mlir::Value ArrayBasedRepresentation::GenerateGetTreeClassId(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex) {
  Operation* ensembleConstOp = ensemble.getDefiningOp();
  AssertOpIsOfType<mlir::decisionforest::EnsembleConstantOp>(ensembleConstOp);
  
  auto mapIter = ensembleConstantToMemrefsMap.find(ensembleConstOp);
  assert (mapIter != ensembleConstantToMemrefsMap.end());
  auto& ensembleInfo = mapIter->second;

  auto treeClassMemref = ensembleInfo.classInfoGlobal;
  auto treeClassMemrefType = treeClassMemref.getType().cast<mlir::MemRefType>();

  auto classId = rewriter.create<memref::LoadOp>(op->getLoc(), treeClassMemrefType.getElementType(), treeClassMemref, treeIndex);
  return classId;
}

mlir::Value ArrayBasedRepresentation::GenerateGetLeafValueOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, 
                                                             mlir::Value nodeIndex) {
  auto location = op->getLoc();

  auto treeMemref = this->GetTreeMemref(treeValue);
  auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
  assert (treeMemrefType);

  auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
  auto thresholdType = treeTileType.getThresholdFieldType();

  // Load threshold
  // TODO Ideally, this should be a different op for when we deal with tile sizes != 1. We will then need to load 
  // a single threshold value and cast it the trees return type
  Value treeIndex = GetTreeIndex(treeValue);
  auto loadThresholdOp = rewriter.create<decisionforest::LoadTileThresholdsOp>(location, 
                                                                               thresholdType,
                                                                               treeMemref,
                                                                               static_cast<Value>(nodeIndex),
                                                                               treeIndex);
  Value leafValue = loadThresholdOp;
  
  if (treeTileType.getTileSize() != 1) {
    if (decisionforest::InsertDebugHelpers) {
      InsertPrintVectorOp(rewriter, location, 0, treeTileType.getThresholdElementType().getIntOrFloatBitWidth(), treeTileType.getTileSize(), loadThresholdOp);
    }
    auto zeroConst = rewriter.create<arith::ConstantIntOp>(location, int64_t(0), rewriter.getI32Type());
    auto extractElement = rewriter.create<vector::ExtractElementOp>(location, static_cast<Value>(loadThresholdOp), zeroConst);
    leafValue = extractElement;
  }
  return leafValue;
}

mlir::Value ArrayBasedRepresentation::GenerateIsLeafOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, mlir::Value nodeIndex) {
  auto location = op->getLoc();
  auto treeMemref = this->GetTreeMemref(treeValue);
  auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
  assert (treeMemrefType);

  auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
  auto featureIndexType = treeTileType.getIndexFieldType();
  auto treeIndex = GetTreeIndex(treeValue);
  auto loadFeatureIndexOp = rewriter.create<decisionforest::LoadTileFeatureIndicesOp>(location, 
                                                                                      featureIndexType,
                                                                                      treeMemref,
                                                                                      static_cast<Value>(nodeIndex),
                                                                                      treeIndex);    
  
  Value featureIndexValue;
  if (treeTileType.getTileSize() == 1) {
    featureIndexValue = loadFeatureIndexOp;
  }
  else {
    auto indexVectorType = featureIndexType.cast<mlir::VectorType>();
    assert (indexVectorType);
    auto zeroConst = rewriter.create<arith::ConstantIntOp>(location, int64_t(0), rewriter.getI32Type());
    auto extractFirstElement = rewriter.create<vector::ExtractElementOp>(location, static_cast<Value>(loadFeatureIndexOp), zeroConst);
    featureIndexValue = extractFirstElement;
  }
  auto minusOneConstant = rewriter.create<arith::ConstantIntOp>(location, int64_t(-1), treeTileType.getIndexElementType());
  auto comparison = rewriter.create<arith::CmpIOp>(location, mlir::arith::CmpIPredicate::eq, featureIndexValue, static_cast<Value>(minusOneConstant));
  
  if (decisionforest::InsertDebugHelpers) {
    Value outcome = rewriter.create<mlir::arith::ExtUIOp>(location, rewriter.getI32Type(), static_cast<Value>(comparison));
    rewriter.create<decisionforest::PrintIsLeafOp>(location, nodeIndex, featureIndexValue, outcome);
  }
  return static_cast<Value>(comparison);
}

mlir::Value ArrayBasedRepresentation::GenerateIsLeafTileOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, mlir::Value nodeIndex) {
  return this->GenerateIsLeafOp(rewriter, op, treeValue, nodeIndex);
}

void ArrayBasedRepresentation::AddTypeConversions(mlir::MLIRContext& context, LLVMTypeConverter& typeConverter) {
  typeConverter.addConversion([&](decisionforest::TiledNumericalNodeType type) {
                auto thresholdType = type.getThresholdFieldType();
                auto indexType = type.getIndexFieldType();
                if (type.getTileSize() == 1) {
                  return LLVM::LLVMStructType::getLiteral(&context, {thresholdType, indexType});
                }
                else {
                  auto tileShapeIDType = type.getTileShapeType();
                  return LLVM::LLVMStructType::getLiteral(&context, {thresholdType, indexType, tileShapeIDType});
                }
              });
}

void ArrayBasedRepresentation::AddLLVMConversionPatterns(LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<LoadTileFeatureIndicesOpLowering,
               LoadTileThresholdOpLowering,
               LoadTileShapeOpLowering,
               InitTileOpLowering,
               GetModelMemrefSizeOpLowering>(converter);
}

void ArrayBasedRepresentation::LowerCacheRowsOp(ConversionPatternRewriter &rewriter,
                                                mlir::Operation *op,
                                                ArrayRef<Value> operands) {
  LowerCacheRowsOpToCPU(rewriter, op, operands);
}

mlir::Value ArrayBasedRepresentation::GetTreeIndex(Value tree) {
  return ::GetTreeIndexValue(tree);
}

std::shared_ptr<IRepresentation> constructArrayBasedRepresentation() {
  return std::make_shared<ArrayBasedRepresentation>();
}

REGISTER_REPRESENTATION(array, constructArrayBasedRepresentation)

// ===---------------------------------------------------=== //
// Sparse representation
// ===---------------------------------------------------=== //

void SparseRepresentation::InitRepresentation() {
  sparseEnsembleConstantToMemrefsMap.clear();
  sparseGetTreeOperationMap.clear();
}

mlir::LogicalResult SparseRepresentation::GenerateModelGlobals(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter,
                                                               std::shared_ptr<decisionforest::IModelSerializer> m_serializer) {
    mlir::decisionforest::EnsembleConstantOp ensembleConstOp = llvm::dyn_cast<mlir::decisionforest::EnsembleConstantOp>(op);
    assert(ensembleConstOp);
    assert(operands.size() == 0);
    if (!ensembleConstOp)
        return mlir::failure();
    
    auto location = op->getLoc();
    auto owningModule = op->getParentOfType<mlir::ModuleOp>();
    assert (owningModule);
    
    // TODO the names of the model and offset global should be generated so they're unique for each ensemble constant
    // TODO the getter function names need to be persisted with the actual tree values in the JSON so the runtime can call them. 
    std::string modelMemrefName = "model";
    std::string offsetMemrefName = "offsets";
    std::string lengthMemrefName = "lengths";
    std::string leavesMemrefName = "leaves";
    std::string leavesLengthMemrefName = "leavesLengths";
    std::string leavesOffsetMemrefName = "leavesOffsets";
    std::string classInfoMemrefName = "treeClassInfo";

    auto memrefTypes = AddGlobalMemrefs(owningModule, ensembleConstOp, rewriter, location, modelMemrefName, offsetMemrefName, lengthMemrefName, 
                                        leavesMemrefName, leavesLengthMemrefName, leavesOffsetMemrefName, classInfoMemrefName, m_serializer);
    AddModelMemrefInitFunction(owningModule, modelMemrefName, std::get<0>(memrefTypes).cast<MemRefType>(), rewriter, location);
    
    // Add getters for all the globals we've created
    auto getModelGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<0>(memrefTypes), modelMemrefName);
    auto getOffsetGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<1>(memrefTypes), offsetMemrefName);
    auto getLengthGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<1>(memrefTypes), lengthMemrefName);
    auto getLeavesGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<2>(memrefTypes), leavesMemrefName);
    auto getLeavesOffsetGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<1>(memrefTypes), leavesOffsetMemrefName);
    auto getLeavesLengthGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<1>(memrefTypes), leavesLengthMemrefName);
    auto classInfoGlobal = rewriter.create<memref::GetGlobalOp>(location, std::get<3>(memrefTypes), classInfoMemrefName);
    
    Type lookUpTableMemrefType;
    Value getLUT;

    SparseEnsembleConstantLoweringInfo info {static_cast<Value>(getModelGlobal), static_cast<Value>(getOffsetGlobal), 
                                       static_cast<Value>(getLengthGlobal), getLUT,
                                       getLeavesGlobal, getLeavesOffsetGlobal, getLeavesLengthGlobal, classInfoGlobal,
                                       std::get<0>(memrefTypes), std::get<1>(memrefTypes), std::get<1>(memrefTypes), 
                                       lookUpTableMemrefType, std::get<2>(memrefTypes), std::get<3>(memrefTypes)};
    sparseEnsembleConstantToMemrefsMap[op] = info;
    return mlir::success();
}

std::tuple<Type, Type, Type, Type> SparseRepresentation::AddGlobalMemrefs(mlir::ModuleOp module, mlir::decisionforest::EnsembleConstantOp& ensembleConstOp,
                                        ConversionPatternRewriter &rewriter, Location location,
                                        const std::string& modelMemrefName, const std::string& offsetMemrefName, const std::string& lengthMemrefName,
                                        const std::string& leavesMemrefName, const std::string& leavesLengthMemrefName,
                                        const std::string& leavesOffsetMemrefName, const std::string& treeInfo,
                                        std::shared_ptr<IModelSerializer> serializer) {
  mlir::decisionforest::DecisionForestAttribute forestAttribute = ensembleConstOp.getForest();
  mlir::decisionforest::DecisionForest& forest = forestAttribute.GetDecisionForest();

  SaveAndRestoreInsertionPoint saveAndRestoreInsertPoint(rewriter);
  rewriter.setInsertionPoint(&module.front());

  auto forestType = ensembleConstOp.getResult().getType().cast<decisionforest::TreeEnsembleType>();
  assert (forestType.doAllTreesHaveSameTileSize()); // There is still an assumption here that all trees have the same tile size
  auto treeType = forestType.getTreeType(0).cast<decisionforest::TreeType>();

  m_thresholdType = treeType.getThresholdType();
  m_featureIndexType = treeType.getFeatureIndexType(); 
  m_tileSize = treeType.getTileSize();
  m_tileShapeType = treeType.getTileShapeType();
  auto childIndexType = treeType.getChildIndexType();
  // assert (tileSize == 1);
  Type memrefElementType = decisionforest::TiledNumericalNodeType::get(m_thresholdType, m_featureIndexType, m_tileShapeType, 
                                                                       m_tileSize, childIndexType);
  serializer->Persist(forest, forestType);
  
  auto modelMemrefSize = decisionforest::GetTotalNumberOfTiles();
  auto modelMemrefType = MemRefType::get({modelMemrefSize}, memrefElementType);
  rewriter.create<memref::GlobalOp>(location, modelMemrefName,
                                    /*sym_visibility=*/rewriter.getStringAttr("private"),
                                    /*type=*/modelMemrefType,
                                    /*initial_value=*/rewriter.getUnitAttr(),
                                    /*constant=*/false, IntegerAttr());
  AddGlobalMemrefGetter(module, modelMemrefName, modelMemrefType, rewriter, location);
  
  // Create offset memref
  auto offsetSize = (int32_t)forest.NumTrees();
  auto offsetMemrefType = MemRefType::get({offsetSize}, rewriter.getIndexType());
  rewriter.create<memref::GlobalOp>(location, offsetMemrefName, rewriter.getStringAttr("private"),
                                    offsetMemrefType, rewriter.getUnitAttr(), false, IntegerAttr());
  AddGlobalMemrefGetter(module, offsetMemrefName, offsetMemrefType, rewriter, location);
  
  // Create length memref
  rewriter.create<memref::GlobalOp>(location, lengthMemrefName, rewriter.getStringAttr("private"),
                                    offsetMemrefType, rewriter.getUnitAttr(), false, IntegerAttr());
  AddGlobalMemrefGetter(module, lengthMemrefName, offsetMemrefType, rewriter, location);

  auto leavesMemrefSize = decisionforest::GetTotalNumberOfLeaves();
  auto leavesMemrefType = MemRefType::get({leavesMemrefSize}, m_thresholdType);
  rewriter.create<memref::GlobalOp>(location, leavesMemrefName, rewriter.getStringAttr("private"),
                                    leavesMemrefType, rewriter.getUnitAttr(), false, IntegerAttr());
  AddGlobalMemrefGetter(module, leavesMemrefName, leavesMemrefType, rewriter, location);
  
  if (TreeBeard::Logging::loggingOptions.logGenCodeStats)
      TreeBeard::Logging::Log("Leaves memref size : " + std::to_string(leavesMemrefSize * (m_thresholdType.getIntOrFloatBitWidth()/8)));

  // Create leaf offset memref
  rewriter.create<memref::GlobalOp>(location, leavesOffsetMemrefName, rewriter.getStringAttr("private"),
                                    offsetMemrefType, rewriter.getUnitAttr(), false, IntegerAttr());
  AddGlobalMemrefGetter(module, leavesOffsetMemrefName, offsetMemrefType, rewriter, location);

  // Create leaf length memref
  rewriter.create<memref::GlobalOp>(location, leavesLengthMemrefName, rewriter.getStringAttr("private"),
                                    offsetMemrefType, rewriter.getUnitAttr(), false, IntegerAttr());
  AddGlobalMemrefGetter(module, leavesLengthMemrefName, offsetMemrefType, rewriter, location);

  auto classInfoSize = forest.IsMultiClassClassifier() ? offsetSize : 0;
  auto classInfoMemrefType = MemRefType::get({classInfoSize}, treeType.getResultType());
  rewriter.create<memref::GlobalOp>(
    location,
    treeInfo,
    rewriter.getStringAttr("public"),
    classInfoMemrefType,
    rewriter.getUnitAttr(),
    false,
    IntegerAttr());
  AddGlobalMemrefGetter(module, treeInfo, classInfoMemrefType, rewriter, location);

  return std::make_tuple(modelMemrefType, offsetMemrefType, leavesMemrefType, classInfoMemrefType);
}

void SparseRepresentation::GenModelMemrefInitFunctionBody(MemRefType memrefType, Value modelGlobalMemref,
                                                          mlir::OpBuilder &builder, Location location, Value tileIndex,
                                                          Value thresholdMemref, Value indexMemref,
                                                          Value tileShapeIdMemref, Value childIndexMemref) {
  auto modelMemrefElementType = memrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
  int32_t tileSize = modelMemrefElementType.getTileSize();

  // index = tileSize * tileIndex
  auto tileSizeConst = builder.create<arith::ConstantIndexOp>(location, tileSize);
  auto tileSizeTimesi = builder.create<arith::MulIOp>(location, tileIndex, tileSizeConst);
  
  if (tileSize > 1) {
    auto thresholdVec = CreateZeroVectorFPConst(builder, location, modelMemrefElementType.getThresholdElementType(), tileSize);
    auto indexVec = CreateZeroVectorIntConst(builder, location, modelMemrefElementType.getIndexElementType(), tileSize);

    // Load from index to index + (tileSize - 1) into a vector
    for (int32_t j = 0 ; j<tileSize ; ++j) {
      auto offset = builder.create<arith::ConstantIndexOp>(location, j);
      auto index =  builder.create<arith::AddIOp>(location, tileSizeTimesi, offset);
      auto thresholdVal = builder.create<memref::LoadOp>(location, thresholdMemref, static_cast<Value>(index));
      auto jConst = builder.create<arith::ConstantIntOp>(location, j, builder.getI32Type());
      thresholdVec = builder.create<vector::InsertElementOp>(location, thresholdVal, thresholdVec, jConst);
      auto indexVal = builder.create<memref::LoadOp>(location, indexMemref, static_cast<Value>(index));
      indexVec = builder.create<vector::InsertElementOp>(location, indexVal, indexVec, jConst);
    }
    auto tileShapeID = builder.create<memref::LoadOp>(location, tileShapeIdMemref, tileIndex);
    auto childIndex = builder.create<memref::LoadOp>(location, childIndexMemref, tileIndex);
    builder.create<decisionforest::InitSparseTileOp>(location, modelGlobalMemref, tileIndex, thresholdVec, indexVec, tileShapeID, childIndex);
  }
  else {
    // Load from index to index + (tileSize - 1) into a vector
    auto thresholdVal = builder.create<memref::LoadOp>(location, thresholdMemref, static_cast<Value>(tileIndex));
    auto indexVal = builder.create<memref::LoadOp>(location, indexMemref, static_cast<Value>(tileIndex));
    auto childIndex = builder.create<memref::LoadOp>(location, childIndexMemref, static_cast<Value>(tileIndex));
    // TODO check how tileShapeID vector is created when tileSize = 1
    auto tileShapeID = builder.create<arith::ConstantIntOp>(location, 0, builder.getI32Type());
    builder.create<decisionforest::InitSparseTileOp>(location, modelGlobalMemref, tileIndex, thresholdVal, indexVal, tileShapeID, childIndex);
  }
}

void SparseRepresentation::AddModelMemrefInitFunction(mlir::ModuleOp module, std::string globalName, MemRefType memrefType, 
                                                      ConversionPatternRewriter &rewriter, Location location) {
  assert (memrefType.getShape().size() == 1);
  SaveAndRestoreInsertionPoint saveAndRestoreEntryPoint(rewriter);
  auto modelMemrefElementType = memrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
  int32_t tileSize = modelMemrefElementType.getTileSize();
  auto thresholdArgType = MemRefType::get({ memrefType.getShape()[0] * tileSize }, modelMemrefElementType.getThresholdElementType());
  auto indexArgType = MemRefType::get({ memrefType.getShape()[0] * tileSize }, modelMemrefElementType.getIndexElementType());
  auto tileShapeIDArgType = MemRefType::get(memrefType.getShape(), modelMemrefElementType.getTileShapeType());
  auto childrenIndexArgType = MemRefType::get(memrefType.getShape(), modelMemrefElementType.getChildIndexType());
  mlir::FunctionType initMemrefFuncType;
  initMemrefFuncType = rewriter.getFunctionType(TypeRange{thresholdArgType, 
                                                          indexArgType, tileShapeIDArgType, 
                                                          childrenIndexArgType},
                                                rewriter.getI32Type());
  std::string funcName = "Init_" + globalName;
  NamedAttribute visibilityAttribute{module.getSymVisibilityAttrName(), rewriter.getStringAttr("public")};
  auto initModelMemrefFunc = mlir::func::FuncOp::create(location, funcName, initMemrefFuncType, ArrayRef<NamedAttribute>(visibilityAttribute));
  auto &entryBlock = *initModelMemrefFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(&entryBlock);

  // for tileIndex = 0 : len
  auto getGlobalMemref = rewriter.create<memref::GetGlobalOp>(location, memrefType, globalName);
  auto zeroIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
  auto oneIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 1);
  auto lenIndexConst = rewriter.create<arith::ConstantIndexOp>(location, memrefType.getShape()[0]);
  auto forLoop = rewriter.create<scf::ForOp>(location, zeroIndexConst, lenIndexConst, oneIndexConst);
  auto tileIndex = forLoop.getInductionVar();
  rewriter.setInsertionPointToStart(forLoop.getBody());
  GenModelMemrefInitFunctionBody(memrefType, getGlobalMemref, 
                                 rewriter, location, tileIndex,
                                 initModelMemrefFunc.getArgument(0),
                                 initModelMemrefFunc.getArgument(1),
                                 initModelMemrefFunc.getArgument(2),
                                 initModelMemrefFunc.getArgument(3));
  rewriter.setInsertionPointAfter(forLoop);
  
  auto modelSize = rewriter.create<decisionforest::GetModelMemrefSizeOp>(location, rewriter.getI32Type(), getGlobalMemref, lenIndexConst);
  rewriter.create<mlir::func::ReturnOp>(location, static_cast<Value>(modelSize));
  module.push_back(initModelMemrefFunc);
}

mlir::Value SparseRepresentation::GetTreeMemref(mlir::Value treeValue) {
  auto getTreeOp = treeValue.getDefiningOp();
  AssertOpIsOfType<mlir::decisionforest::GetTreeFromEnsembleOp>(getTreeOp);
  auto getTreeOperationMapIter = sparseGetTreeOperationMap.find(getTreeOp);
  assert(getTreeOperationMapIter != sparseGetTreeOperationMap.end());
  auto treeMemref = getTreeOperationMapIter->second.treeMemref;
  return treeMemref;
}

mlir::Value SparseRepresentation::GetLeafMemref(mlir::Value treeValue) {
  auto getTreeOp = treeValue.getDefiningOp();
  AssertOpIsOfType<mlir::decisionforest::GetTreeFromEnsembleOp>(getTreeOp);
  auto getTreeOperationMapIter = sparseGetTreeOperationMap.find(getTreeOp);
  assert(getTreeOperationMapIter != sparseGetTreeOperationMap.end());
  auto leafMemref = getTreeOperationMapIter->second.leavesMemref;
  return leafMemref;
}

std::vector<mlir::Value> SparseRepresentation::GenerateExtraLoads(mlir::Location location,
                                                                  ConversionPatternRewriter &rewriter,
                                                                  mlir::Value tree,
                                                                  mlir::Value nodeIndex) {
  auto treeMemRef = GetTreeMemref(tree);
  auto memrefType = treeMemRef.getType().cast<MemRefType>();
  auto treeTileType = memrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
  auto loadChildIndexOp = rewriter.create<decisionforest::LoadChildIndexOp>(location, treeTileType.getChildIndexType(), treeMemRef, nodeIndex);
  auto childIndex = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(loadChildIndexOp));
  return std::vector<mlir::Value>{childIndex};
}

mlir::Value SparseRepresentation::GenerateMoveToChild(mlir::Location location, ConversionPatternRewriter &rewriter, mlir::Value nodeIndex, 
                                                      mlir::Value childNumber, int32_t tileSize, std::vector<mlir::Value>& extraLoads) {
  assert (extraLoads.size() > 0);
  auto childIndex = extraLoads.front();
  auto newIndex = rewriter.create<arith::AddIOp>(location, rewriter.getIndexType(), childIndex, childNumber);

  if (decisionforest::InsertDebugHelpers) {
    // (child base index, lutLookup result, new index)
    auto zeroVector = CreateZeroVectorIndexConst(rewriter, location, 3);
    auto zeroConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
    auto elem0Set = rewriter.create<vector::InsertElementOp>(location, childIndex, zeroVector, zeroConst);
    auto oneConst = rewriter.create<arith::ConstantIndexOp>(location, 1);
    auto elem1Set = rewriter.create<vector::InsertElementOp>(location, childNumber, elem0Set, oneConst);
    auto twoConst = rewriter.create<arith::ConstantIndexOp>(location, 2);
    auto elem2Set = rewriter.create<vector::InsertElementOp>(location, newIndex, elem1Set, twoConst);
    InsertPrintVectorOp(rewriter, location, 1, 64, 3, elem2Set);
  }
  return newIndex;
}

void SparseRepresentation::GenerateTreeMemref(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex) {
  auto location = op->getLoc();
  Operation* ensembleConstOp = ensemble.getDefiningOp();
  AssertOpIsOfType<mlir::decisionforest::EnsembleConstantOp>(ensembleConstOp);
  
  auto mapIter = sparseEnsembleConstantToMemrefsMap.find(ensembleConstOp);
  assert (mapIter != sparseEnsembleConstantToMemrefsMap.end());
  auto& ensembleInfo = mapIter->second;

  auto modelMemrefIndex = rewriter.create<memref::LoadOp>(location, ensembleInfo.offsetGlobal, treeIndex);
  auto treeLength = rewriter.create<memref::LoadOp>(location, ensembleInfo.lengthGlobal, treeIndex);; // TODO Need to put this into the map too
  auto treeMemref = rewriter.create<memref::SubViewOp>(location, ensembleInfo.modelGlobal, ArrayRef<OpFoldResult>({static_cast<Value>(modelMemrefIndex)}),
                                                        ArrayRef<OpFoldResult>({static_cast<Value>(treeLength)}), ArrayRef<OpFoldResult>({rewriter.getIndexAttr(1)}));

  int32_t tileSize = ensembleInfo.modelGlobal.getType().cast<MemRefType>().getElementType().cast<decisionforest::TiledNumericalNodeType>().getTileSize();
  Value leavesMemref;
  if (tileSize > 1) {
    auto leavesMemrefIndex = rewriter.create<memref::LoadOp>(location, ensembleInfo.leavesOffsetGlobal, treeIndex);
    auto leavesLength = rewriter.create<memref::LoadOp>(location, ensembleInfo.leavesLengthGlobal, treeIndex);; // TODO Need to put this into the map too
    leavesMemref = rewriter.create<memref::SubViewOp>(location, ensembleInfo.leavesGlobal, ArrayRef<OpFoldResult>({static_cast<Value>(leavesMemrefIndex)}),
                                                        ArrayRef<OpFoldResult>({static_cast<Value>(leavesLength)}), ArrayRef<OpFoldResult>({rewriter.getIndexAttr(1)}));
  }   
  // if (decisionforest::InsertDebugHelpers) {
  //   rewriter.create<decisionforest::PrintTreeToDOTFileOp>(location, treeMemref, treeIndex);
  // }
  sparseGetTreeOperationMap[op] = { static_cast<Value>(treeMemref), static_cast<Value>(leavesMemref) };
}

mlir::Value SparseRepresentation::GenerateGetTreeClassId(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex) {
  Operation* ensembleConstOp = ensemble.getDefiningOp();
  AssertOpIsOfType<mlir::decisionforest::EnsembleConstantOp>(ensembleConstOp);
  
  auto mapIter = sparseEnsembleConstantToMemrefsMap.find(ensembleConstOp);
  assert (mapIter != sparseEnsembleConstantToMemrefsMap.end());
  auto& ensembleInfo = mapIter->second;

  auto treeClassMemref = ensembleInfo.classInfoGlobal;
  auto treeClassMemrefType = treeClassMemref.getType().cast<mlir::MemRefType>();

  auto classId = rewriter.create<memref::LoadOp>(op->getLoc(), treeClassMemrefType.getElementType(), treeClassMemref, treeIndex);
  return classId;
}

mlir::Value SparseRepresentation::GenerateGetLeafValueOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, 
                                                         mlir::Value nodeIndex) {
  auto location = op->getLoc();
  auto treeMemref = this->GetTreeMemref(treeValue);
  auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
  assert (treeMemrefType);

  auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
  auto thresholdType = treeTileType.getThresholdFieldType();

  if (decisionforest::InsertDebugHelpers) {
    rewriter.create<decisionforest::PrintTreeNodeOp>(location, nodeIndex);
  }

  if (treeTileType.getTileSize() == 1) {
    Value treeIndex = GetTreeIndex(treeValue);
    auto loadThresholdOp = rewriter.create<decisionforest::LoadTileThresholdsOp>(location, 
                                                                                 thresholdType,
                                                                                 treeMemref,
                                                                                 static_cast<Value>(nodeIndex),
                                                                                 treeIndex);
    Value leafValue = loadThresholdOp;
    return static_cast<Value>(leafValue);
  }
  else {
    auto treeMemrefLen = rewriter.create<memref::DimOp>(location, treeMemref, 0);
    auto leafIndex = rewriter.create<arith::SubIOp>(location, nodeIndex, treeMemrefLen);
    auto leavesMemref = this->GetLeafMemref(treeValue);
    auto leafValue = rewriter.create<memref::LoadOp>(location, leavesMemref, static_cast<Value>(leafIndex));
    
    // auto resultConst = rewriter.create<arith::ConstantFloatOp>(location, APFloat(double(0.5)), rewriter.getF64Type());
    // TODO cast the loaded value to the correct result type of the tree. 
    return static_cast<Value>(leafValue);
  }
}

mlir::Value SparseRepresentation::GenerateIsLeafOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, mlir::Value nodeIndex) {
  auto location = op->getLoc();
  auto treeMemref = this->GetTreeMemref(treeValue);
  auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
  assert (treeMemrefType);

  auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
  if (treeTileType.getTileSize() == 1) {
    auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
    auto featureIndexType = treeTileType.getIndexFieldType();
    auto treeIndex = GetTreeIndex(treeValue);
    auto loadFeatureIndexOp = rewriter.create<decisionforest::LoadTileFeatureIndicesOp>(location, 
                                                                                        featureIndexType,
                                                                                        treeMemref,
                                                                                        static_cast<Value>(nodeIndex),
                                                                                        treeIndex);
    Value featureIndexValue = loadFeatureIndexOp;
    auto minusOneConstant = rewriter.create<arith::ConstantIntOp>(location, int64_t(-1), treeTileType.getIndexElementType());
    auto comparison = rewriter.create<arith::CmpIOp>(location, mlir::arith::CmpIPredicate::eq, featureIndexValue, static_cast<Value>(minusOneConstant));
    
    if (decisionforest::InsertDebugHelpers) {
      Value outcome = rewriter.create<mlir::arith::ExtUIOp>(location, rewriter.getI32Type(), static_cast<Value>(comparison));
      Value featureIndexI32 = featureIndexValue;
      if (!featureIndexType.isInteger(32))
        featureIndexI32 = rewriter.create<mlir::arith::ExtSIOp>(location, rewriter.getI32Type(), featureIndexValue);
      rewriter.create<decisionforest::PrintIsLeafOp>(location, nodeIndex, featureIndexI32, outcome);
    }
    return static_cast<Value>(comparison);
  }
  else {
    // Check if node index is out of bounds
    auto treeMemrefLen = rewriter.create<memref::DimOp>(location, treeMemref, 0);
    auto nodeIndexOutOfBounds = rewriter.create<arith::CmpIOp>(location, arith::CmpIPredicate::sge, nodeIndex, treeMemrefLen);
    return static_cast<Value>(nodeIndexOutOfBounds);
  }
}

mlir::Value SparseRepresentation::GenerateIsLeafTileOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, mlir::Value nodeIndex) {
    auto location = op->getLoc();
    auto treeMemref = GetTreeMemref(treeValue);
    auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
    assert (treeMemrefType);
    
    auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
    auto childIndexType = treeTileType.getChildIndexType();
    auto loadChildIndexOp = rewriter.create<decisionforest::LoadChildIndexOp>(location, childIndexType, treeMemref, static_cast<Value>(nodeIndex));    
    
    Value childIndexValue = static_cast<Value>(loadChildIndexOp);
    assert (treeTileType.getTileSize() != 1);
    
    auto minusOneConstant = rewriter.create<arith::ConstantIntOp>(location, int64_t(-1), childIndexType);
    auto comparison = rewriter.create<arith::CmpIOp>(location, mlir::arith::CmpIPredicate::eq, childIndexValue, static_cast<Value>(minusOneConstant));
    
    if (decisionforest::InsertDebugHelpers) {
      Value outcome = rewriter.create<mlir::arith::ExtUIOp>(location, rewriter.getI32Type(), static_cast<Value>(comparison));
      Value childIndexI32 = childIndexValue;
      if (!childIndexType.isInteger(32))
        childIndexI32 = rewriter.create<mlir::arith::ExtSIOp>(location, rewriter.getI32Type(), childIndexValue);
      rewriter.create<decisionforest::PrintIsLeafOp>(location, nodeIndex, childIndexI32, outcome);
    }
    return comparison;
}

void SparseRepresentation::AddTypeConversions(mlir::MLIRContext& context, LLVMTypeConverter& typeConverter) {
  typeConverter.addConversion([&](decisionforest::TiledNumericalNodeType type) {
              auto thresholdType = type.getThresholdFieldType();
              auto indexType = type.getIndexFieldType();
              auto childIndexType = type.getChildIndexType();
              auto tileShapeIDType = type.getTileShapeType();
              if (type.getTileSize() == 1) {
                return LLVM::LLVMStructType::getLiteral(&context, {thresholdType, indexType, tileShapeIDType, childIndexType});
              }
              else {
                return LLVM::LLVMStructType::getLiteral(&context, {thresholdType, indexType, tileShapeIDType, childIndexType});
              }
            });
}

void SparseRepresentation::AddLLVMConversionPatterns(LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<LoadTileFeatureIndicesOpLowering,
               LoadTileThresholdOpLowering,
               LoadTileShapeOpLowering,
               LoadChildIndexOpLowering,
               InitSparseTileOpLowering,
               GetModelMemrefSizeOpLowering>(converter);
}

mlir::Value SparseRepresentation::GetTreeIndex(Value tree) {
  return ::GetTreeIndexValue(tree);
}

void SparseRepresentation::LowerCacheRowsOp(ConversionPatternRewriter &rewriter,
                                            mlir::Operation *op,
                                            ArrayRef<Value> operands) {
  LowerCacheRowsOpToCPU(rewriter, op, operands);
}

std::shared_ptr<IRepresentation> constructSparseRepresentation() {
  return std::make_shared<SparseRepresentation>();
}

REGISTER_REPRESENTATION(sparse, constructSparseRepresentation)

// ===---------------------------------------------------=== //
// ModelSerializerFactory Methods
// ===---------------------------------------------------=== //

std::shared_ptr<IRepresentation> RepresentationFactory::GetRepresentation(const std::string& name) {
  auto mapIter = m_constructionMap.find(name);
  assert (mapIter != m_constructionMap.end() && "Unknown representation name!");
  return mapIter->second();
}

RepresentationFactory& RepresentationFactory::Get() {
  static std::unique_ptr<RepresentationFactory> sInstancePtr = nullptr;
  if (sInstancePtr == nullptr)
      sInstancePtr = std::make_unique<RepresentationFactory>();
  return *sInstancePtr;
}

bool RepresentationFactory::RegisterRepresentation(const std::string& name,
                                                   RepresentationConstructor_t constructionFunc) {
  assert (m_constructionMap.find(name) == m_constructionMap.end());
  m_constructionMap[name] = constructionFunc;
  return true;
}

std::shared_ptr<IRepresentation> ConstructRepresentation() {
  if (decisionforest::UseSparseTreeRepresentation)
    return RepresentationFactory::Get().GetRepresentation("sparse");
  else
    return RepresentationFactory::Get().GetRepresentation("array");
}

std::shared_ptr<IRepresentation> ConstructGPURepresentation() {
  if (decisionforest::UseSparseTreeRepresentation)
    return RepresentationFactory::Get().GetRepresentation("gpu_sparse");
  else
    return RepresentationFactory::Get().GetRepresentation("gpu_array");
}

} // namespace decisionforest
} // namespace mlir
