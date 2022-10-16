#include "Dialect.h"
// #include "Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/MemoryBuffer.h"

// #include "mlir/Dialect/StandardOps/Transforms/Passes.h"
// #include "mlir/Dialect/SCF/Passes.h"
// #include "mlir/Dialect/Tensor/Transforms/Passes.h"
// #include "mlir/Transforms/Passes.h"
// #include "mlir/Dialect/Linalg/Passes.h"

using namespace mlir;

namespace mlir
{
namespace decisionforest
{
// Defined in LowerDebugHelpers.cpp
void InsertPrintElementAddressIfNeeded(ConversionPatternRewriter& rewriter, Location location, ModuleOp module,
                                       Value extractMemrefBufferPointer, Value indexVal, Value actualIndex, Value elemIndexConst, Value elemPtr);
}
}

namespace {

const int32_t kAlignedPointerIndexInMemrefStruct = 1;
const int32_t kOffsetIndexInMemrefStruct = 2;
const int32_t kThresholdElementNumberInTile = 0;
const int32_t kFeatureIndexElementNumberInTile = 1;
const int32_t kTileShapeElementNumberInTile = 2;
const int32_t kChildIndexElementNumberInTile = 3;

Type GenerateGetElementPtr(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter, Type elementMLIRType,
                           int64_t elementNumber, TypeConverter* typeConverter, Value& elementPtr) {
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
                                                                          rewriter.getI64ArrayAttr(kAlignedPointerIndexInMemrefStruct));

  auto extractMemrefOffset = rewriter.create<LLVM::ExtractValueOp>(location, indexType, operands[kTreeMemrefOperandNum],
                                                                   rewriter.getI64ArrayAttr(kOffsetIndexInMemrefStruct));

  auto actualIndex = rewriter.create<LLVM::AddOp>(location, indexType, static_cast<Value>(extractMemrefOffset), static_cast<Value>(indexVal));

  // Get a pointer to i'th tile's threshold
  auto elementPtrType = LLVM::LLVMPointerType::get(elementType);
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

void GenerateLoadStructElement(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter, 
                               int64_t elementNumber, TypeConverter* typeConverter) {
  
  auto location = op->getLoc();
  Value elementPtr;
  auto elementType = GenerateGetElementPtr(op, operands, rewriter, op->getResult(0).getType(), elementNumber, typeConverter, elementPtr);
  
  // Load the element
  auto elementVal = rewriter.create<LLVM::LoadOp>(location, elementType, static_cast<Value>(elementPtr));
  
  rewriter.replaceOp(op, static_cast<Value>(elementVal));
}

void GenerateStoreStructElement(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter, Type elementMLIRType,
                               int64_t elementNumber, TypeConverter* typeConverter, Value elementVal) {
  
  auto location = op->getLoc();
  Value elementPtr;
  GenerateGetElementPtr(op, operands, rewriter, elementMLIRType, elementNumber, typeConverter, elementPtr);
  
  // Store the element
  rewriter.create<LLVM::StoreOp>(location, elementVal, elementPtr);
}

struct LoadTileThresholdOpLowering: public ConversionPattern {
  LoadTileThresholdOpLowering(LLVMTypeConverter& typeConverter)
  : ConversionPattern(typeConverter, mlir::decisionforest::LoadTileThresholdsOp::getOperationName(), 1 /*benefit*/, &typeConverter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    assert (operands.size() == 2);
    GenerateLoadStructElement(op, operands, rewriter, kThresholdElementNumberInTile, getTypeConverter());
    return mlir::success();
  }
};

struct LoadTileFeatureIndicesOpLowering: public ConversionPattern {
  LoadTileFeatureIndicesOpLowering(LLVMTypeConverter& typeConverter) 
  : ConversionPattern(typeConverter, mlir::decisionforest::LoadTileFeatureIndicesOp::getOperationName(), 1 /*benefit*/, &typeConverter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 2);
    GenerateLoadStructElement(op, operands, rewriter, kFeatureIndexElementNumberInTile, getTypeConverter());
    return mlir::success();
  }
};

struct LoadTileShapeOpLowering : public ConversionPattern {
  LoadTileShapeOpLowering(LLVMTypeConverter& typeConverter) 
  : ConversionPattern(typeConverter, mlir::decisionforest::LoadTileShapeOp::getOperationName(), 1 /*benefit*/, &typeConverter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 2);
    GenerateLoadStructElement(op, operands, rewriter, kTileShapeElementNumberInTile, getTypeConverter());
    return mlir::success();
  }
};

struct LoadChildIndexOpLowering : public ConversionPattern {
  LoadChildIndexOpLowering(LLVMTypeConverter& typeConverter) 
  : ConversionPattern(typeConverter, mlir::decisionforest::LoadChildIndexOp::getOperationName(), 1 /*benefit*/, &typeConverter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 2);
    GenerateLoadStructElement(op, operands, rewriter, kChildIndexElementNumberInTile, getTypeConverter());
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
    GenerateStoreStructElement(op, operands, rewriter, tileOpAdaptor.thresholds().getType(), 0, getTypeConverter(), tileOpAdaptor.thresholds());
    GenerateStoreStructElement(op, operands, rewriter, tileOpAdaptor.featureIndices().getType(), 1, getTypeConverter(), tileOpAdaptor.featureIndices());
    auto modelMemrefType = op->getOperand(0).getType().cast<MemRefType>();
    auto tileType = modelMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
    if (tileType.getTileSize() > 1)
      GenerateStoreStructElement(op, operands, rewriter, tileOpAdaptor.tileShapeID().getType(), 2, getTypeConverter(), tileOpAdaptor.tileShapeID());
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
    GenerateStoreStructElement(op, operands, rewriter, tileOpAdaptor.thresholds().getType(), 0, getTypeConverter(), tileOpAdaptor.thresholds());
    GenerateStoreStructElement(op, operands, rewriter, tileOpAdaptor.featureIndices().getType(), 1, getTypeConverter(), tileOpAdaptor.featureIndices());
    auto modelMemrefType = op->getOperand(0).getType().cast<MemRefType>();
    auto tileType = modelMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
    if (tileType.getTileSize() > 1)
      GenerateStoreStructElement(op, operands, rewriter, tileOpAdaptor.tileShapeID().getType(), 2, getTypeConverter(), tileOpAdaptor.tileShapeID());
    GenerateStoreStructElement(op, operands, rewriter, tileOpAdaptor.childIndex().getType(), 3, getTypeConverter(), tileOpAdaptor.childIndex());

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
                                                                            rewriter.getI64ArrayAttr(kAlignedPointerIndexInMemrefStruct));

    auto extractMemrefOffset = rewriter.create<LLVM::ExtractValueOp>(location, indexType, operands[kTreeMemrefOperandNum],
                                                                    rewriter.getI64ArrayAttr(kOffsetIndexInMemrefStruct));

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

void AddTreebeardTypeConversions(MLIRContext& context, LLVMTypeConverter& typeConverter) {
  if (decisionforest::UseSparseTreeRepresentation == false)
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
  else
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

#ifndef OMP_SUPPORT
struct DecisionForestToLLVMLoweringPass : public PassWrapper<DecisionForestToLLVMLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect, AffineDialect, memref::MemRefDialect, 
                    arith::ArithmeticDialect, vector::VectorDialect>();
  }
  void runOnOperation() final;
};

void DecisionForestToLLVMLoweringPass::runOnOperation() {
  // define the conversion target
  LowerToLLVMOptions options(&getContext());
  // options.useBarePtrCallConv = true;
  // options.emitCWrappers = true;
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();

  auto& context = getContext();
  LLVMTypeConverter typeConverter(&getContext(), options);
  if (decisionforest::UseSparseTreeRepresentation == false)
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
    else
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

  RewritePatternSet patterns(&getContext());
  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  populateVectorToLLVMConversionPatterns(typeConverter, patterns, false);
  vector::populateVectorBroadcastLoweringPatterns(patterns);
  populateMathToLLVMConversionPatterns(typeConverter, patterns);
  arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);

  patterns.add<LoadTileFeatureIndicesOpLowering,
               LoadTileThresholdOpLowering,
               LoadTileShapeOpLowering,
               LoadChildIndexOpLowering,
               InitTileOpLowering,
               InitSparseTileOpLowering,
               GetModelMemrefSizeOpLowering>(typeConverter);
  decisionforest::populateDebugOpLoweringPatterns(patterns, typeConverter);

  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();  
}

#else // OMP_SUPPORT

// ./mlir-opt --convert-scf-to-openmp --convert-memref-to-llvm --convert-scf-to-cf --convert-openmp-to-llvm --reconcile-unrealized-casts ~/temp/omp-mlir.mlir
struct DecisionForestToLLVMLoweringPass : public PassWrapper<DecisionForestToLLVMLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect, AffineDialect, memref::MemRefDialect, 
                    arith::ArithmeticDialect, vector::VectorDialect, omp::OpenMPDialect>();
  }
  void runOnOperation() final;
};

void DecisionForestToLLVMLoweringPass::runOnOperation() {
  // define the conversion target
  LowerToLLVMOptions options(&getContext());
  // options.useBarePtrCallConv = true;
  // options.emitCWrappers = true;
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalDialect<scf::SCFDialect, omp::OpenMPDialect>();
  target.addIllegalDialect<decisionforest::DecisionForestDialect, memref::MemRefDialect>();
  target.addIllegalDialect<arith::ArithmeticDialect, vector::VectorDialect, math::MathDialect>();

  auto& context = getContext();
  LLVMTypeConverter typeConverter(&getContext(), options);
  AddTreebeardTypeConversions(context, typeConverter);

  RewritePatternSet patterns(&getContext());
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateVectorToLLVMConversionPatterns(typeConverter, patterns, false);
  populateMathToLLVMConversionPatterns(typeConverter, patterns);
  arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);

  patterns.add<LoadTileFeatureIndicesOpLowering,
               LoadTileThresholdOpLowering,
               LoadTileShapeOpLowering,
               LoadChildIndexOpLowering,
               InitTileOpLowering,
               InitSparseTileOpLowering,
               GetModelMemrefSizeOpLowering>(typeConverter);
  decisionforest::populateDebugOpLoweringPatterns(patterns, typeConverter);

  auto module = getOperation();

  if (failed(applyFullConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
    llvm::errs() << "Decision forest lowering pass failed\n";
  }
  // module->dump();    
}

struct LowerOMPToLLVMPass : public PassWrapper<LowerOMPToLLVMPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect, AffineDialect, memref::MemRefDialect, 
                    arith::ArithmeticDialect, vector::VectorDialect, omp::OpenMPDialect, func::FuncDialect>();
  }
  void runOnOperation() final {
    LowerToLLVMOptions options(&getContext());
    auto module = getOperation();
    auto& context = getContext();

    LLVMTypeConverter converter(&getContext(), options);
    AddTreebeardTypeConversions(context, converter);
    
    // Convert to OpenMP operations with LLVM IR dialect
    RewritePatternSet patterns(&getContext());
    arith::populateArithmeticToLLVMConversionPatterns(converter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    populateMemRefToLLVMConversionPatterns(converter, patterns);
    populateFuncToLLVMConversionPatterns(converter, patterns);
    populateOpenMPToLLVMConversionPatterns(converter, patterns);

    LLVMConversionTarget target(getContext());
    target.addLegalOp<omp::TerminatorOp, omp::TaskyieldOp, omp::FlushOp,
                      omp::BarrierOp, omp::TaskwaitOp>();
    configureOpenMPToLLVMConversionLegality(target, converter);
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
#endif

struct PrintModulePass : public PassWrapper<PrintModulePass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect, AffineDialect, memref::MemRefDialect, 
                    arith::ArithmeticDialect, vector::VectorDialect, omp::OpenMPDialect>();
  }
  void runOnOperation() final {
    auto module = getOperation();
    module->dump();
  }
};

} // end anonymous namespace

namespace mlir
{
namespace decisionforest
{
#ifndef OMP_SUPPORT
void LowerToLLVM(mlir::MLIRContext& context, mlir::ModuleOp module) {
  // llvm::DebugFlag = false;
  // Lower from high-level IR to mid-level IR
  mlir::PassManager pm(&context);
  // mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  pm.addPass(std::make_unique<DecisionForestToLLVMLoweringPass>());
  pm.addPass(createReconcileUnrealizedCastsPass());
  
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to LLVM failed.\n";
  }
}
#else // OMP_SUPPORT
void LowerToLLVM(mlir::MLIRContext& context, mlir::ModuleOp module) {
  // llvm::DebugFlag = true;
  // Lower from high-level IR to mid-level IR
  mlir::PassManager pm(&context);
  pm.addPass(std::make_unique<DecisionForestToLLVMLoweringPass>());
  pm.addPass(createConvertSCFToOpenMPPass());
  pm.addPass(createMemRefToLLVMPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(std::make_unique<LowerOMPToLLVMPass>());
  // pm.addPass(std::make_unique<PrintModulePass>());
  pm.addPass(createReconcileUnrealizedCastsPass());
  
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to LLVM failed.\n";
  }
  // module->dump();
}
#endif // OMP_SUPPORT
void dumpAssembly(const llvm::Module* llvmModule) {
  LLVMMemoryBufferRef bufferOut;
  char *errorMessage = nullptr;
  std::string error;
  const llvm::Target* target = llvm::TargetRegistry::lookupTarget(llvmModule->getTargetTriple(), error);
  
  if (!target) {
    llvm::errs() << "No target found";
  }
  else if (target->hasTargetMachine()){
    
    // TODO - Revisit these defaults
    auto CPU = "generic";
    auto Features = "";
    llvm::TargetOptions opt;
    auto RM = Optional<llvm::Reloc::Model>();
    
    auto tm = target->createTargetMachine(llvmModule->getTargetTriple(), CPU, Features, opt, RM);
    
    LLVMTargetMachineEmitToMemoryBuffer(
      (LLVMTargetMachineRef)tm, // #TODO - Should use a llvm::wrap function. Didn't find one.
      llvm::wrap(llvmModule),
      LLVMCodeGenFileType::LLVMAssemblyFile,
      &errorMessage, // #TODO - This buffer and the buffer below might leak. Look at it later.
      &bufferOut);
    
    if (errorMessage)
        llvm::errs() <<  errorMessage;
    else {
        llvm::errs() << "<ASM Target =" << target->getName() <<">\n";
        llvm::errs() << llvm::unwrap(bufferOut)->getBuffer() << "\n";
        llvm::errs() << "</ASM>" << "\n";
    }
  }
  else {
    llvm::errs () << "Target machine not found";
  }
}

int dumpLLVMIR(mlir::ModuleOp module, bool dumpAsm) {
  // Init LLVM targets
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR Context
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());
  
  if (dumpAsm) {
    dumpAssembly(llvmModule.get());
  }

  llvm::errs() << *llvmModule << "\n";
  return 0;
}

int dumpLLVMIRToFile(mlir::ModuleOp module, const std::string& filename) {
  // Init LLVM targets
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR Context
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }
  ExecutionEngine::setupTargetTriple(llvmModule.get());
  std::error_code ec;
  llvm::raw_fd_ostream filestream(filename, ec);
  filestream << *llvmModule;
  return 0;
}

// The routine below lowers tensors to memrefs. We don't need it currently
// as we're not using tensors. Commenting it out so we can remove some MLIR 
// link time dependencies. 

// void LowerTensorTypes(mlir::MLIRContext& context, mlir::ModuleOp module) {
//   // Lower from high-level IR to mid-level IR
//   mlir::PassManager pm(&context);
//   // mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
//   // Partial bufferization passes.
//   pm.addPass(createTensorConstantBufferizePass());
//   pm.addNestedPass<FuncOp>(createSCFBufferizePass());
//   pm.addNestedPass<FuncOp>(createStdBufferizePass());
//   pm.addNestedPass<FuncOp>(createLinalgBufferizePass());
//   pm.addNestedPass<FuncOp>(createTensorBufferizePass());
//   pm.addPass(createFuncBufferizePass());

//   // Finalizing bufferization pass.
//   pm.addNestedPass<FuncOp>(createFinalizingBufferizePass());

//   if (mlir::failed(pm.run(module))) {
//     llvm::errs() << "Conversion from NodeType to Index failed.\n";
//   }
// }

} // decisionforest
} // mlir