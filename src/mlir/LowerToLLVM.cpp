#include "Dialect.h"
// #include "Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

// #include "mlir/Dialect/StandardOps/Transforms/Passes.h"
// #include "mlir/Dialect/SCF/Passes.h"
// #include "mlir/Dialect/Tensor/Transforms/Passes.h"
// #include "mlir/Transforms/Passes.h"
// #include "mlir/Dialect/Linalg/Passes.h"

using namespace mlir;

namespace {

const int32_t kAlignedPointerIndexInMemrefStruct = 1;
const int32_t kFeatureIndexElementNumberInTile = 1;

struct LoadTileThresholdOpLowering: public ConversionPattern {
  LoadTileThresholdOpLowering(LLVMTypeConverter& typeConverter)
  : ConversionPattern(typeConverter, mlir::decisionforest::LoadTileThresholdsOp::getOperationName(), 1 /*benefit*/, &typeConverter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    assert (operands.size() == 2);
    decisionforest::LoadTileThresholdsOpAdaptor loadTileThresholdAdaptor(operands);
    auto location = op->getLoc();
    
    auto memrefType = loadTileThresholdAdaptor.treeMemref().getType();
    auto memrefStructType = memrefType.cast<LLVM::LLVMStructType>();
    auto alignedPtrType = memrefStructType.getBody()[kAlignedPointerIndexInMemrefStruct].cast<LLVM::LLVMPointerType>();
    auto tileType = alignedPtrType.getElementType().cast<LLVM::LLVMStructType>();
    
    auto indexVal = loadTileThresholdAdaptor.nodeIndex();
    auto indexType = indexVal.getType();
    // type2 is the integer type corresponding to the mlir index type
    assert (indexType.isa<IntegerType>());
    
    auto resultType = op->getResults()[0].getType();
    auto typeConverter = this->getTypeConverter();
    auto thresholdType = typeConverter->convertType(resultType);

    // Extract the memref's aligned pointer
    auto extractMemrefBufferPointer = rewriter.create<LLVM::ExtractValueOp>(location, alignedPtrType, loadTileThresholdAdaptor.treeMemref(),
                                                                            rewriter.getI64ArrayAttr(kAlignedPointerIndexInMemrefStruct));

    // Get a pointer to i'th tile's threshold
    auto thresholdPtrType = LLVM::LLVMPointerType::get(thresholdType);
    assert(thresholdType == tileType.getBody()[0] && "The result type should be the same as the threshold type in the struct.");
    auto zeroConst = rewriter.create<LLVM::ConstantOp>(location, rewriter.getI32Type(), rewriter.getIntegerAttr(rewriter.getI32Type(), int64_t(0)));
    auto thresholdPtr = rewriter.create<LLVM::GEPOp>(location, thresholdPtrType, static_cast<Value>(extractMemrefBufferPointer), 
                                                   ValueRange({indexVal, static_cast<Value>(zeroConst)}));

    // Cast point to threshold type
    // auto thresholdPtr = rewriter.create<LLVM::BitcastOp>(location, thresholdPtrType, elementPtr);

    // Load the threshold
    auto thresholdVal = rewriter.create<LLVM::LoadOp>(location, thresholdType, static_cast<Value>(thresholdPtr));
    
    rewriter.replaceOp(op, static_cast<Value>(thresholdVal));

    return mlir::success();
  }
};

struct LoadTileFeatureIndicesOpLowering: public ConversionPattern {
  LoadTileFeatureIndicesOpLowering(LLVMTypeConverter& typeConverter) 
  : ConversionPattern(typeConverter, mlir::decisionforest::LoadTileFeatureIndicesOp::getOperationName(), 1 /*benefit*/, &typeConverter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 2);
    decisionforest::LoadTileFeatureIndicesOpAdaptor loadTileFeatureIndicesAdaptor(operands);
    auto location = op->getLoc();

    auto memrefType = loadTileFeatureIndicesAdaptor.treeMemref().getType();
    auto memrefStructType = memrefType.cast<LLVM::LLVMStructType>();
    auto alignedPtrType = memrefStructType.getBody()[kAlignedPointerIndexInMemrefStruct].cast<LLVM::LLVMPointerType>();
    auto tileType = alignedPtrType.getElementType().cast<LLVM::LLVMStructType>();
    
    auto indexVal = loadTileFeatureIndicesAdaptor.nodeIndex();
    auto indexType = indexVal.getType();
    // type2 is the integer type corresponding to the mlir index type
    assert (indexType.isa<IntegerType>());
    
    auto resultType = op->getResults()[0].getType();
    auto typeConverter = this->getTypeConverter();
    auto featureIndexType = typeConverter->convertType(resultType);

    // Extract the memref's aligned pointer
    auto extractMemrefBufferPointer = rewriter.create<LLVM::ExtractValueOp>(location, alignedPtrType, loadTileFeatureIndicesAdaptor.treeMemref(),
                                                                            rewriter.getI64ArrayAttr(kAlignedPointerIndexInMemrefStruct));

    // Get pointer to i'th tile's feature index
    auto featureIndexPtrType = LLVM::LLVMPointerType::get(featureIndexType);
    assert(featureIndexType == tileType.getBody()[1] && "The result type should be the same as the feature index type in the struct.");
    auto oneConst = rewriter.create<LLVM::ConstantOp>(location, rewriter.getI32Type(), rewriter.getIntegerAttr(rewriter.getI32Type(), int64_t(1)));
    auto featureIndexPtr = rewriter.create<LLVM::GEPOp>(location, featureIndexPtrType, static_cast<Value>(extractMemrefBufferPointer), 
                                                        ValueRange({indexVal, static_cast<Value>(oneConst)}));

    // Load the tile so we can extract the feature index
    // auto tileValue = rewriter.create<LLVM::LoadOp>(location, tileType, static_cast<Value>(elementPtr));

    // Extract the feature index
    // auto featureIndex = rewriter.create<LLVM::ExtractValueOp>(location, featureIndexType, static_cast<Value>(tileValue),
    //                                                           rewriter.getI64ArrayAttr(kFeatureIndexElementNumberInTile));
    
    // Load the threshold
    auto featureIndex = rewriter.create<LLVM::LoadOp>(location, featureIndexType, static_cast<Value>(featureIndexPtr));
    
    rewriter.replaceOp(op, static_cast<Value>(featureIndex));

    return mlir::success();
  }
};

struct DecisionForestToLLVMLoweringPass : public PassWrapper<DecisionForestToLLVMLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect, AffineDialect, memref::MemRefDialect, tensor::TensorDialect, StandardOpsDialect>();
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
  typeConverter.addConversion([&](decisionforest::TiledNumericalNodeType type) {
                auto thresholdType = type.getThresholdType();
                auto indexType = type.getIndexType();
                return LLVM::LLVMStructType::getLiteral(&context, {thresholdType, indexType}, true);
              });

  RewritePatternSet patterns(&getContext());
  populateAffineToStdConversionPatterns(patterns);
  populateLoopToStdConversionPatterns(patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateStdToLLVMFuncOpConversionPattern(typeConverter, patterns);
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  patterns.add<LoadTileFeatureIndicesOpLowering,
               LoadTileThresholdOpLowering>(typeConverter);

  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();  
}

} // end anonymous namespace

namespace mlir
{
namespace decisionforest
{
void LowerToLLVM(mlir::MLIRContext& context, mlir::ModuleOp module) {
  // Lower from high-level IR to mid-level IR
  mlir::PassManager pm(&context);
  // mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  pm.addPass(std::make_unique<DecisionForestToLLVMLoweringPass>());
  pm.addPass(createReconcileUnrealizedCastsPass());
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to LLVM failed.\n";
  }
}

int dumpLLVMIR(mlir::ModuleOp module) {
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR Context
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Init LLVM targets
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  llvm::errs() << *llvmModule << "\n";
  return 0;
}


// The routine below lowers tensors to memrefs. We don't need it currently
// as we're not using tensors. Commenting it out so we can remove some MLIR 
// link time dependencies. 

// void LowerTensorTypes(mlir::MLIRContext& context, mlir::ModuleOp module) {
//   // Lower from high-level IR to mid-level IR
//   mlir::PassManager pm(&context);
//   // mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
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