#include "Dialect.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;

namespace
{
FlatSymbolRefAttr getOrInsertFunction(std::string& functionName, LLVM::LLVMFunctionType functionType, PatternRewriter &rewriter,
                                      ModuleOp module) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(functionName))
    return SymbolRefAttr::get(context, functionName);

  // Insert the PrintTreePrediction function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), functionName, functionType);
  return SymbolRefAttr::get(context, functionName);
}

struct PrintTreePredictionOpLowering: public ConversionPattern {
  PrintTreePredictionOpLowering(LLVMTypeConverter& typeConverter)
  : ConversionPattern(typeConverter, mlir::decisionforest::PrintTreePredictionOp::getOperationName(), 1 /*benefit*/, &typeConverter.getContext()) {}

  static FlatSymbolRefAttr getOrInsertPrintTreePrediction(PatternRewriter &rewriter,
                                                          ModuleOp module) {
    auto *context = module.getContext();
    std::string functionName = "PrintTreePrediction";
    // Create a function declaration for PrintTreePrediction, the signature is:
    //   * `i64 (f64, i64)`
    auto llvmF64Ty = FloatType::getF64(context);
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI64Ty, {llvmF64Ty, llvmI64Ty});

    return getOrInsertFunction(functionName, llvmFnType, rewriter, module);
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    assert (operands.size() == 2);
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto printFunctionRef = getOrInsertPrintTreePrediction(rewriter, parentModule);
    Value prediction = operands[0];
    if (prediction.getType() != rewriter.getF64Type()) {
      auto castOp = rewriter.create<LLVM::FPExtOp>(op->getLoc(), rewriter.getF64Type(), operands[0]);
      prediction = static_cast<Value>(castOp);
    }
    Value treeIndex = operands[1];
    if (treeIndex.getType() != rewriter.getI64Type()) {
      auto sextOp = rewriter.create<LLVM::SExtOp>(op->getLoc(), rewriter.getI64Type(), treeIndex);
      treeIndex = static_cast<Value>(sextOp);
    }
    rewriter.create<CallOp>(op->getLoc(), printFunctionRef, rewriter.getI64Type(), ArrayRef<Value>({ prediction, treeIndex }));
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct PrintTreeNodeOpLowering: public ConversionPattern {
  PrintTreeNodeOpLowering(LLVMTypeConverter& typeConverter)
  : ConversionPattern(typeConverter, mlir::decisionforest::PrintTreeNodeOp::getOperationName(), 1 /*benefit*/, &typeConverter.getContext()) {}

  static FlatSymbolRefAttr getOrInsertPrintTreeNode(PatternRewriter &rewriter,
                                                    ModuleOp module) {
    auto *context = module.getContext();
    std::string functionName = "PrintNodeIndex";

    // Create a function declaration for PrintNodeIndex, the signature is:
    //   * `i64 (i64)`
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI64Ty, {llvmI64Ty});

    return getOrInsertFunction(functionName, llvmFnType, rewriter, module);
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    assert (operands.size() == 1);
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto printFunctionRef = getOrInsertPrintTreeNode(rewriter, parentModule);
    Value nodeIndex = operands[0];
    if (nodeIndex.getType() != rewriter.getI64Type()) {
      auto sextOp = rewriter.create<LLVM::SExtOp>(op->getLoc(), rewriter.getI64Type(), nodeIndex);
      nodeIndex = static_cast<Value>(sextOp);
    }
    rewriter.create<CallOp>(op->getLoc(), printFunctionRef, rewriter.getI64Type(), ArrayRef<Value>({ nodeIndex }));
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct PrintComparisonOpLowering: public ConversionPattern {
  PrintComparisonOpLowering(LLVMTypeConverter& typeConverter)
  : ConversionPattern(typeConverter, mlir::decisionforest::PrintComparisonOp::getOperationName(), 1 /*benefit*/, &typeConverter.getContext()) {}

  static FlatSymbolRefAttr getOrInsertPrintComparison(PatternRewriter &rewriter,
                                                      ModuleOp module) {
    auto *context = module.getContext();
    std::string functionName = "PrintComparison";

    // Create a function declaration for PrintComparison, the signature is:
    //   * `i64 (double, double, i64)`
    auto llvmF64Ty = FloatType::getF64(context);
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI64Ty, {llvmF64Ty, llvmF64Ty, llvmI64Ty});

    return getOrInsertFunction(functionName, llvmFnType, rewriter, module);
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    assert (operands.size() == 3);
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto printFunctionRef = getOrInsertPrintComparison(rewriter, parentModule);
    Value feature = operands[0];
    if (feature.getType() != rewriter.getF64Type()) {
      auto castOp = rewriter.create<LLVM::FPExtOp>(op->getLoc(), rewriter.getF64Type(), operands[0]);
      feature = static_cast<Value>(castOp);
    }
    Value threshold = operands[1];
    if (threshold.getType() != rewriter.getF64Type()) {
      auto castOp = rewriter.create<LLVM::FPExtOp>(op->getLoc(), rewriter.getF64Type(), operands[0]);
      threshold = static_cast<Value>(castOp);
    }
    Value nodeIndex = operands[2];
    if (nodeIndex.getType() != rewriter.getI64Type()) {
      auto sextOp = rewriter.create<LLVM::SExtOp>(op->getLoc(), rewriter.getI64Type(), nodeIndex);
      nodeIndex = static_cast<Value>(sextOp);
    }
    rewriter.create<CallOp>(op->getLoc(), printFunctionRef, rewriter.getI64Type(), ArrayRef<Value>({ feature, threshold, nodeIndex }));
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

const int32_t kAlignedPointerIndexInMemrefStruct = 1;
const int32_t kOffsetIndexInMemrefStruct = 2;
const int32_t kLengthIndexInMemrefStruct = 3;

struct PrintTreeToDOTFileOpLowering: public ConversionPattern {
  PrintTreeToDOTFileOpLowering(LLVMTypeConverter& typeConverter)
  : ConversionPattern(typeConverter, mlir::decisionforest::PrintTreeToDOTFileOp::getOperationName(), 1 /*benefit*/, &typeConverter.getContext()) {}

  static FlatSymbolRefAttr getOrInsertPrintTreeNode(PatternRewriter &rewriter,
                                                    ModuleOp module, Type ptrType) {
    auto *context = module.getContext();
    std::string functionName = "PrintTreeToDOTFile";

    // Create a function declaration for PrintNodeIndex, the signature is:
    //   * `i64 (TileType*, i64, i64, i64)`
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI64Ty, {ptrType, llvmI64Ty, llvmI64Ty, llvmI64Ty});

    return getOrInsertFunction(functionName, llvmFnType, rewriter, module);
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    assert (operands.size() == 2);
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    const int32_t kTreeMemrefOperandNum = 0;
    const int32_t kIndexOperandNum = 1;
    auto location = op->getLoc();
    
    auto memrefType = operands[kTreeMemrefOperandNum].getType();
    auto memrefStructType = memrefType.cast<LLVM::LLVMStructType>();
    auto alignedPtrType = memrefStructType.getBody()[kAlignedPointerIndexInMemrefStruct].cast<LLVM::LLVMPointerType>();
    
    auto indexVal = operands[kIndexOperandNum];
    auto indexType = indexVal.getType();
    assert (indexType.isa<IntegerType>());

    // Get the pointer value
    // Extract the memref's aligned pointer
    auto extractMemrefBufferPointer = rewriter.create<LLVM::ExtractValueOp>(location, alignedPtrType, operands[kTreeMemrefOperandNum],
                                                                            rewriter.getI64ArrayAttr(kAlignedPointerIndexInMemrefStruct));

    auto extractMemrefOffset = rewriter.create<LLVM::ExtractValueOp>(location, indexType, operands[kTreeMemrefOperandNum],
                                                                    rewriter.getI64ArrayAttr(kOffsetIndexInMemrefStruct));

    // Get a pointer to first tile in this tree
    auto elementPtr = rewriter.create<LLVM::GEPOp>(location, alignedPtrType, static_cast<Value>(extractMemrefBufferPointer), 
                                                  ValueRange({ static_cast<Value>(extractMemrefOffset) }));

    // Get the length of this tree memref
    auto extractMemrefLength = rewriter.create<LLVM::ExtractValueOp>(location, indexType, operands[kTreeMemrefOperandNum],
                                                                     rewriter.getI64ArrayAttr({ kLengthIndexInMemrefStruct, 0 }));

    // Create a tile size constant
    auto mlirModelMemrefType = op->getOperand(0).getType().cast<mlir::MemRefType>();
    auto tileType = mlirModelMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
    int64_t tileSize = tileType.getTileSize();
    auto tileSizeConst = rewriter.create<LLVM::ConstantOp>(location, rewriter.getI64Type(), rewriter.getIntegerAttr(rewriter.getI64Type(), tileSize));
    auto printFunctionRef = getOrInsertPrintTreeNode(rewriter, parentModule, alignedPtrType);

    // int64_t PrintTreeToDOTFile(TreeTileType *treeBuf, int64_t length, int64_t treeIndex, int64_t tileSize)
    rewriter.create<CallOp>(op->getLoc(), printFunctionRef, rewriter.getI64Type(), ArrayRef<Value>({ elementPtr, extractMemrefLength, indexVal, tileSizeConst }));
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct PrintInputRowOpLowering: public ConversionPattern {
  PrintInputRowOpLowering(LLVMTypeConverter& typeConverter)
  : ConversionPattern(typeConverter, mlir::decisionforest::PrintInputRowOp::getOperationName(), 1 /*benefit*/, &typeConverter.getContext()) {}

  static FlatSymbolRefAttr getOrInsertPrintRow(PatternRewriter &rewriter,
                                                    ModuleOp module) {
    auto *context = module.getContext();
    std::string functionName = "PrintInputRow";

    // Create a function declaration for PrintInputRow, the signature is:
    //   * `i64 (TileType*, i64, i64)`
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmF64PtrTy = LLVM::LLVMPointerType::get(FloatType::getF64(context));
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI64Ty, {llvmF64PtrTy, llvmI64Ty, llvmI64Ty});

    return getOrInsertFunction(functionName, llvmFnType, rewriter, module);
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    assert (operands.size() == 2);
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    const int32_t kRowMemrefOperandNum = 0;
    const int32_t kIndexOperandNum = 1;
    auto location = op->getLoc();
    
    auto memrefType = operands[kRowMemrefOperandNum].getType();
    auto memrefStructType = memrefType.cast<LLVM::LLVMStructType>();
    auto alignedPtrType = memrefStructType.getBody()[kAlignedPointerIndexInMemrefStruct].cast<LLVM::LLVMPointerType>();
    
    auto indexVal = operands[kIndexOperandNum];
    auto indexType = indexVal.getType();
    assert (indexType.isa<IntegerType>());

    // Get the pointer value
    // Extract the memref's aligned pointer
    auto extractMemrefBufferPointer = rewriter.create<LLVM::ExtractValueOp>(location, alignedPtrType, operands[kRowMemrefOperandNum],
                                                                            rewriter.getI64ArrayAttr(kAlignedPointerIndexInMemrefStruct));

    auto extractMemrefOffset = rewriter.create<LLVM::ExtractValueOp>(location, indexType, operands[kRowMemrefOperandNum],
                                                                    rewriter.getI64ArrayAttr(kOffsetIndexInMemrefStruct));

    // Get a pointer to the first element of this row
    auto elementPtr = rewriter.create<LLVM::GEPOp>(location, alignedPtrType, static_cast<Value>(extractMemrefBufferPointer), 
                                                  ValueRange({ static_cast<Value>(extractMemrefOffset) }));

    // Get the length of this memref (since its a 2D memref, get the second element of the length array to get number of cols)
    auto extractMemrefLength = rewriter.create<LLVM::ExtractValueOp>(location, indexType, operands[kRowMemrefOperandNum],
                                                                     rewriter.getI64ArrayAttr({ kLengthIndexInMemrefStruct, 1 }));

    auto printFunctionRef = getOrInsertPrintRow(rewriter, parentModule);

    // int64_t PrintInputRow(double *treeBuf, int64_t length, int64_t rowIndex)
    rewriter.create<CallOp>(op->getLoc(), printFunctionRef, rewriter.getI64Type(), ArrayRef<Value>({ elementPtr, extractMemrefLength, indexVal }));
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct PrintIsLeafOpLowering: public ConversionPattern {
  PrintIsLeafOpLowering(LLVMTypeConverter& typeConverter)
  : ConversionPattern(typeConverter, mlir::decisionforest::PrintIsLeafOp::getOperationName(), 1 /*benefit*/, &typeConverter.getContext()) {}

  static FlatSymbolRefAttr getOrInsertPrintIsLeaf(PatternRewriter &rewriter,
                                                  ModuleOp module) {
    auto *context = module.getContext();
    std::string functionName = "PrintIsLeaf";
    // Create a function declaration for PrintIsLeaf, the signature is:
    //   * `i64 (i64, i32, i32)`
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI64Ty, {llvmI64Ty, llvmI32Ty, llvmI32Ty});

    return getOrInsertFunction(functionName, llvmFnType, rewriter, module);
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    assert (operands.size() == 3);
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto printFunctionRef = getOrInsertPrintIsLeaf(rewriter, parentModule);
    rewriter.create<CallOp>(op->getLoc(), printFunctionRef, rewriter.getI64Type(), ArrayRef<Value>{operands[0], operands[1], operands[2]});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct PrintVectorOpLowering: public ConversionPattern {
  PrintVectorOpLowering(LLVMTypeConverter& typeConverter)
  : ConversionPattern(typeConverter, mlir::decisionforest::PrintVectorOp::getOperationName(), 1 /*benefit*/, &typeConverter.getContext()) {}

  static FlatSymbolRefAttr getOrInsertPrintVector(PatternRewriter &rewriter,
                                                  ModuleOp module) {
    auto *context = module.getContext();
    std::string functionName = "PrintVector";
    // Create a function declaration for PrintIsLeaf, the signature is:
    //   * `i64 (i32, i32, i32, ...)`
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI64Ty, {llvmI32Ty, llvmI32Ty, llvmI32Ty}, true);

    return getOrInsertFunction(functionName, llvmFnType, rewriter, module);
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    // assert (operands.size() == 3);
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto printFunctionRef = getOrInsertPrintVector(rewriter, parentModule);
    rewriter.create<CallOp>(op->getLoc(), printFunctionRef, rewriter.getI64Type(), operands);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

} // anonymous namespace

namespace mlir
{
namespace decisionforest
{

void populateDebugOpLoweringPatterns(RewritePatternSet& patterns, LLVMTypeConverter& typeConverter) {
  if (InsertDebugHelpers)
    patterns.add<PrintTreePredictionOpLowering,
                 PrintTreeNodeOpLowering,
                 PrintTreeToDOTFileOpLowering,
                 PrintInputRowOpLowering,
                 PrintComparisonOpLowering,
                 PrintIsLeafOpLowering,
                 PrintVectorOpLowering>(typeConverter);
}  

void InsertPrintElementAddressIfNeeded(ConversionPatternRewriter& rewriter, Location location, ModuleOp module,
                                       Value bufferPtr, Value indexVal, Value actualIndex, Value elemIndex, Value elemPtr) {
    auto context = rewriter.getContext();
    
    std::string functionName = "PrintElementAddress";
    // Create a function declaration for PrintElementAddress, the signature is:
    //      int64_t PrintElementAddress(void *bufPtr, int64_t index, int64_t actualIndex, int32_t elementIndex, void *elemPtr)
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmI32PtrTy = LLVM::LLVMPointerType::get(llvmI32Ty);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI64Ty, {llvmI32PtrTy, llvmI64Ty, llvmI64Ty, llvmI32Ty, llvmI32PtrTy});

    auto printFunctionRef = getOrInsertFunction(functionName, llvmFnType, rewriter, module);
    // Cast the pointers to int* and then pass them 
    auto castedBufferPtr = rewriter.create<LLVM::BitcastOp>(location, llvmI32PtrTy, bufferPtr);
    auto castedElemPtr = rewriter.create<LLVM::BitcastOp>(location, llvmI32PtrTy, elemPtr);

    rewriter.create<CallOp>(location, printFunctionRef, rewriter.getI64Type(), ValueRange{castedBufferPtr, indexVal, actualIndex, elemIndex, castedElemPtr});
}

} // decisionforest
} // mlir
