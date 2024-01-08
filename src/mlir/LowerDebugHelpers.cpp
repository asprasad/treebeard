#include "Dialect.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#include "LIRLoweringHelpers.h"

using namespace mlir;

namespace mlir {
namespace decisionforest {
FlatSymbolRefAttr getOrInsertFunction(std::string &functionName,
                                      LLVM::LLVMFunctionType functionType,
                                      PatternRewriter &rewriter,
                                      ModuleOp module) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(functionName))
    return SymbolRefAttr::get(context, functionName);

  // Insert the PrintTreePrediction function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), functionName,
                                    functionType);
  return SymbolRefAttr::get(context, functionName);
}

/// Return a value representing an access into a global string with the given
/// name, creating the string if necessary.
static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                     StringRef name, StringRef value,
                                     ModuleOp module) {
  // Create the global at the entry of the module.
  LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMArrayType::get(
        IntegerType::get(builder.getContext(), 8), value.size());
    global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                            LLVM::Linkage::Internal, name,
                                            builder.getStringAttr(value),
                                            /*alignment=*/0);
  }

  // Get the pointer to the first character in the global string.
  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                builder.getIndexAttr(0));
  return builder.create<LLVM::GEPOp>(
      loc,
      LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
      globalPtr, ArrayRef<Value>({cst0, cst0}));
}

/// Return a symbol reference to the printf function, inserting it into the
/// module if necessary.
static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                           ModuleOp module) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
    return SymbolRefAttr::get(context, "printf");

  // Create a function declaration for printf, the signature is:
  //   * `i32 (i8*, ...)`
  auto llvmI32Ty = IntegerType::get(context, 32);
  auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
  auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                /*isVarArg=*/true);

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
  return SymbolRefAttr::get(context, "printf");
}

struct PrintfOpLowering : public ConversionPattern {
  PrintfOpLowering(LLVMTypeConverter &typeConverter)
      : ConversionPattern(typeConverter,
                          mlir::decisionforest::PrintfOp::getOperationName(),
                          1 /*benefit*/, &typeConverter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto printfOp = AssertOpIsOfType<mlir::decisionforest::PrintfOp>(op);

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto printFunctionRef = getOrInsertPrintf(rewriter, parentModule);
    auto formatStr = printfOp.getFormat();
    auto formatCst = getOrCreateGlobalString(
        op->getLoc(), rewriter,
        std::string("format_string") +
            std::to_string(reinterpret_cast<uintptr_t>(op)),
        formatStr, parentModule);
    std::vector<Value> operandsArrayRef;
    operandsArrayRef.push_back(formatCst);
    operandsArrayRef.insert(operandsArrayRef.end(), operands.begin(),
                            operands.end());
    // rewriter.create<func::CallOp>(op->getLoc(), printFunctionRef,
    //                               rewriter.getI32Type(),
    //                               ArrayRef<Value>(operandsArrayRef));
    rewriter.create<LLVM::CallOp>(op->getLoc(), rewriter.getI32Type(),
                                  printFunctionRef,
                                  ArrayRef<Value>(operandsArrayRef));
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct PrintTreePredictionOpLowering : public ConversionPattern {
  PrintTreePredictionOpLowering(LLVMTypeConverter &typeConverter)
      : ConversionPattern(
            typeConverter,
            mlir::decisionforest::PrintTreePredictionOp::getOperationName(),
            1 /*benefit*/, &typeConverter.getContext()) {}

  static FlatSymbolRefAttr
  getOrInsertPrintTreePrediction(PatternRewriter &rewriter, ModuleOp module) {
    auto *context = module.getContext();
    std::string functionName = "PrintTreePrediction";
    // Create a function declaration for PrintTreePrediction, the signature is:
    //   * `i64 (f64, i64)`
    auto llvmF64Ty = FloatType::getF64(context);
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmFnType =
        LLVM::LLVMFunctionType::get(llvmI64Ty, {llvmF64Ty, llvmI64Ty});

    return getOrInsertFunction(functionName, llvmFnType, rewriter, module);
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 2);
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto printFunctionRef =
        getOrInsertPrintTreePrediction(rewriter, parentModule);
    Value prediction = operands[0];
    if (prediction.getType() != rewriter.getF64Type()) {
      auto castOp = rewriter.create<LLVM::FPExtOp>(
          op->getLoc(), rewriter.getF64Type(), operands[0]);
      prediction = static_cast<Value>(castOp);
    }
    Value treeIndex = operands[1];
    if (treeIndex.getType() != rewriter.getI64Type()) {
      auto sextOp = rewriter.create<LLVM::SExtOp>(
          op->getLoc(), rewriter.getI64Type(), treeIndex);
      treeIndex = static_cast<Value>(sextOp);
    }
    rewriter.create<func::CallOp>(op->getLoc(), printFunctionRef,
                                  rewriter.getI64Type(),
                                  ArrayRef<Value>({prediction, treeIndex}));
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct PrintTreeNodeOpLowering : public ConversionPattern {
  PrintTreeNodeOpLowering(LLVMTypeConverter &typeConverter)
      : ConversionPattern(
            typeConverter,
            mlir::decisionforest::PrintTreeNodeOp::getOperationName(),
            1 /*benefit*/, &typeConverter.getContext()) {}

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
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 1);
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto printFunctionRef = getOrInsertPrintTreeNode(rewriter, parentModule);
    Value nodeIndex = operands[0];
    if (nodeIndex.getType() != rewriter.getI64Type()) {
      auto sextOp = rewriter.create<LLVM::SExtOp>(
          op->getLoc(), rewriter.getI64Type(), nodeIndex);
      nodeIndex = static_cast<Value>(sextOp);
    }
    rewriter.create<func::CallOp>(op->getLoc(), printFunctionRef,
                                  rewriter.getI64Type(),
                                  ArrayRef<Value>({nodeIndex}));
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct PrintComparisonOpLowering : public ConversionPattern {
  PrintComparisonOpLowering(LLVMTypeConverter &typeConverter)
      : ConversionPattern(
            typeConverter,
            mlir::decisionforest::PrintComparisonOp::getOperationName(),
            1 /*benefit*/, &typeConverter.getContext()) {}

  static FlatSymbolRefAttr getOrInsertPrintComparison(PatternRewriter &rewriter,
                                                      ModuleOp module) {
    auto *context = module.getContext();
    std::string functionName = "PrintComparison";

    // Create a function declaration for PrintComparison, the signature is:
    //   * `i64 (double, double, i64)`
    auto llvmF64Ty = FloatType::getF64(context);
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmFnType = LLVM::LLVMFunctionType::get(
        llvmI64Ty, {llvmF64Ty, llvmF64Ty, llvmI64Ty});

    return getOrInsertFunction(functionName, llvmFnType, rewriter, module);
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 3);
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto printFunctionRef = getOrInsertPrintComparison(rewriter, parentModule);
    Value feature = operands[0];
    if (feature.getType() != rewriter.getF64Type()) {
      auto castOp = rewriter.create<LLVM::FPExtOp>(
          op->getLoc(), rewriter.getF64Type(), operands[0]);
      feature = static_cast<Value>(castOp);
    }
    Value threshold = operands[1];
    if (threshold.getType() != rewriter.getF64Type()) {
      auto castOp = rewriter.create<LLVM::FPExtOp>(
          op->getLoc(), rewriter.getF64Type(), operands[0]);
      threshold = static_cast<Value>(castOp);
    }
    Value nodeIndex = operands[2];
    if (nodeIndex.getType() != rewriter.getI64Type()) {
      auto sextOp = rewriter.create<LLVM::SExtOp>(
          op->getLoc(), rewriter.getI64Type(), nodeIndex);
      nodeIndex = static_cast<Value>(sextOp);
    }
    rewriter.create<func::CallOp>(
        op->getLoc(), printFunctionRef, rewriter.getI64Type(),
        ArrayRef<Value>({feature, threshold, nodeIndex}));
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

const int32_t kAlignedPointerIndexInMemrefStruct = 1;
const int32_t kOffsetIndexInMemrefStruct = 2;
const int32_t kLengthIndexInMemrefStruct = 3;

struct PrintTreeToDOTFileOpLowering : public ConversionPattern {
  PrintTreeToDOTFileOpLowering(LLVMTypeConverter &typeConverter)
      : ConversionPattern(
            typeConverter,
            mlir::decisionforest::PrintTreeToDOTFileOp::getOperationName(),
            1 /*benefit*/, &typeConverter.getContext()) {}

  static FlatSymbolRefAttr getOrInsertPrintTreeNode(PatternRewriter &rewriter,
                                                    ModuleOp module,
                                                    Type ptrType) {
    auto *context = module.getContext();
    std::string functionName = "PrintTreeToDOTFile";

    // Create a function declaration for PrintNodeIndex, the signature is:
    //   * `i64 (TileType*, i64, i64, i64)`
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmFnType = LLVM::LLVMFunctionType::get(
        llvmI64Ty, {ptrType, llvmI64Ty, llvmI64Ty, llvmI64Ty});

    return getOrInsertFunction(functionName, llvmFnType, rewriter, module);
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 2);
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    const int32_t kTreeMemrefOperandNum = 0;
    const int32_t kIndexOperandNum = 1;
    auto location = op->getLoc();

    auto memrefType = operands[kTreeMemrefOperandNum].getType();
    auto memrefStructType = memrefType.cast<LLVM::LLVMStructType>();
    auto alignedPtrType =
        memrefStructType.getBody()[kAlignedPointerIndexInMemrefStruct]
            .cast<LLVM::LLVMPointerType>();

    auto indexVal = operands[kIndexOperandNum];
    auto indexType = indexVal.getType();
    assert(indexType.isa<IntegerType>());

    // Get the pointer value
    // Extract the memref's aligned pointer
    auto extractMemrefBufferPointer = rewriter.create<LLVM::ExtractValueOp>(
        location, alignedPtrType, operands[kTreeMemrefOperandNum],
        rewriter.getDenseI64ArrayAttr(kAlignedPointerIndexInMemrefStruct));

    auto extractMemrefOffset = rewriter.create<LLVM::ExtractValueOp>(
        location, indexType, operands[kTreeMemrefOperandNum],
        rewriter.getDenseI64ArrayAttr(kOffsetIndexInMemrefStruct));

    // Get a pointer to first tile in this tree
    auto elementPtr = rewriter.create<LLVM::GEPOp>(
        location, alignedPtrType,
        static_cast<Value>(extractMemrefBufferPointer),
        ValueRange({static_cast<Value>(extractMemrefOffset)}));

    // Get the length of this tree memref
    auto extractMemrefLength = rewriter.create<LLVM::ExtractValueOp>(
        location, indexType, operands[kTreeMemrefOperandNum],
        rewriter.getDenseI64ArrayAttr({kLengthIndexInMemrefStruct, 0}));

    // Create a tile size constant
    auto mlirModelMemrefType =
        op->getOperand(0).getType().cast<mlir::MemRefType>();
    auto tileType = mlirModelMemrefType.getElementType()
                        .cast<decisionforest::TiledNumericalNodeType>();
    int64_t tileSize = tileType.getTileSize();
    auto tileSizeConst = rewriter.create<LLVM::ConstantOp>(
        location, rewriter.getI64Type(),
        rewriter.getIntegerAttr(rewriter.getI64Type(), tileSize));
    auto printFunctionRef =
        getOrInsertPrintTreeNode(rewriter, parentModule, alignedPtrType);

    // int64_t PrintTreeToDOTFile(TreeTileType *treeBuf, int64_t length, int64_t
    // treeIndex, int64_t tileSize)
    rewriter.create<func::CallOp>(
        op->getLoc(), printFunctionRef, rewriter.getI64Type(),
        ArrayRef<Value>(
            {elementPtr, extractMemrefLength, indexVal, tileSizeConst}));
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct PrintInputRowOpLowering : public ConversionPattern {
  PrintInputRowOpLowering(LLVMTypeConverter &typeConverter)
      : ConversionPattern(
            typeConverter,
            mlir::decisionforest::PrintInputRowOp::getOperationName(),
            1 /*benefit*/, &typeConverter.getContext()) {}

  static FlatSymbolRefAttr getOrInsertPrintRow(PatternRewriter &rewriter,
                                               ModuleOp module) {
    auto *context = module.getContext();
    std::string functionName = "PrintInputRow";

    // Create a function declaration for PrintInputRow, the signature is:
    //   * `i64 (TileType*, i64, i64)`
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmF64PtrTy = LLVM::LLVMPointerType::get(FloatType::getF64(context));
    auto llvmFnType = LLVM::LLVMFunctionType::get(
        llvmI64Ty, {llvmF64PtrTy, llvmI64Ty, llvmI64Ty});

    return getOrInsertFunction(functionName, llvmFnType, rewriter, module);
  }

  static FlatSymbolRefAttr getOrInsertPrintRowFloat(PatternRewriter &rewriter,
                                                    ModuleOp module) {
    auto *context = module.getContext();
    std::string functionName = "PrintInputRow_Float";

    // Create a function declaration for PrintInputRow, the signature is:
    //   * `i64 (TileType*, i64, i64)`
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmF64PtrTy = LLVM::LLVMPointerType::get(FloatType::getF32(context));
    auto llvmFnType = LLVM::LLVMFunctionType::get(
        llvmI64Ty, {llvmF64PtrTy, llvmI64Ty, llvmI64Ty});

    return getOrInsertFunction(functionName, llvmFnType, rewriter, module);
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 2);
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    const int32_t kRowMemrefOperandNum = 0;
    const int32_t kIndexOperandNum = 1;
    auto location = op->getLoc();

    auto memrefType = operands[kRowMemrefOperandNum].getType();
    auto memrefStructType = memrefType.cast<LLVM::LLVMStructType>();
    auto alignedPtrType =
        memrefStructType.getBody()[kAlignedPointerIndexInMemrefStruct]
            .cast<LLVM::LLVMPointerType>();

    auto indexVal = operands[kIndexOperandNum];
    auto indexType = indexVal.getType();
    assert(indexType.isa<IntegerType>());

    // Get the pointer value
    // Extract the memref's aligned pointer
    auto extractMemrefBufferPointer = rewriter.create<LLVM::ExtractValueOp>(
        location, alignedPtrType, operands[kRowMemrefOperandNum],
        rewriter.getDenseI64ArrayAttr(kAlignedPointerIndexInMemrefStruct));

    auto extractMemrefOffset = rewriter.create<LLVM::ExtractValueOp>(
        location, indexType, operands[kRowMemrefOperandNum],
        rewriter.getDenseI64ArrayAttr(kOffsetIndexInMemrefStruct));

    // Get a pointer to the first element of this row
    auto elementPtr = rewriter.create<LLVM::GEPOp>(
        location, alignedPtrType,
        static_cast<Value>(extractMemrefBufferPointer),
        ValueRange({static_cast<Value>(extractMemrefOffset)}));

    // Get the length of this memref (since its a 2D memref, get the second
    // element of the length array to get number of cols)
    auto extractMemrefLength = rewriter.create<LLVM::ExtractValueOp>(
        location, indexType, operands[kRowMemrefOperandNum],
        rewriter.getDenseI64ArrayAttr({kLengthIndexInMemrefStruct, 1}));

    if (alignedPtrType.getElementType().isF64()) {
      auto printFunctionRef = getOrInsertPrintRow(rewriter, parentModule);
      // int64_t PrintInputRow(double *treeBuf, int64_t length, int64_t
      // rowIndex)
      rewriter.create<func::CallOp>(
          op->getLoc(), printFunctionRef, rewriter.getI64Type(),
          ArrayRef<Value>({elementPtr, extractMemrefLength, indexVal}));
    } else if (alignedPtrType.getElementType().isF32()) {
      auto printFunctionRef = getOrInsertPrintRowFloat(rewriter, parentModule);
      // int64_t PrintInputRow_Float(float *treeBuf, int64_t length, int64_t
      // rowIndex)
      rewriter.create<func::CallOp>(
          op->getLoc(), printFunctionRef, rewriter.getI64Type(),
          ArrayRef<Value>({elementPtr, extractMemrefLength, indexVal}));
    } else {
      assert(false && "Unsupported type");
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct PrintIsLeafOpLowering : public ConversionPattern {
  PrintIsLeafOpLowering(LLVMTypeConverter &typeConverter)
      : ConversionPattern(
            typeConverter,
            mlir::decisionforest::PrintIsLeafOp::getOperationName(),
            1 /*benefit*/, &typeConverter.getContext()) {}

  static FlatSymbolRefAttr getOrInsertPrintIsLeaf(PatternRewriter &rewriter,
                                                  ModuleOp module) {
    auto *context = module.getContext();
    std::string functionName = "PrintIsLeaf";
    // Create a function declaration for PrintIsLeaf, the signature is:
    //   * `i64 (i64, i32, i32)`
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmFnType = LLVM::LLVMFunctionType::get(
        llvmI64Ty, {llvmI64Ty, llvmI32Ty, llvmI32Ty});

    return getOrInsertFunction(functionName, llvmFnType, rewriter, module);
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 3);
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto printFunctionRef = getOrInsertPrintIsLeaf(rewriter, parentModule);
    rewriter.create<func::CallOp>(
        op->getLoc(), printFunctionRef, rewriter.getI64Type(),
        ArrayRef<Value>{operands[0], operands[1], operands[2]});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct PrintVectorOpLowering : public ConversionPattern {
  PrintVectorOpLowering(LLVMTypeConverter &typeConverter)
      : ConversionPattern(
            typeConverter,
            mlir::decisionforest::PrintVectorOp::getOperationName(),
            1 /*benefit*/, &typeConverter.getContext()) {}

  static FlatSymbolRefAttr getOrInsertPrintVector(PatternRewriter &rewriter,
                                                  ModuleOp module) {
    auto *context = module.getContext();
    std::string functionName = "PrintVector";
    // Create a function declaration for PrintIsLeaf, the signature is:
    //   * `i64 (i32, i32, i32, ...)`
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmFnType = LLVM::LLVMFunctionType::get(
        llvmI64Ty, {llvmI32Ty, llvmI32Ty, llvmI32Ty}, true);

    return getOrInsertFunction(functionName, llvmFnType, rewriter, module);
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // assert (operands.size() == 3);
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto printFunctionRef = getOrInsertPrintVector(rewriter, parentModule);
    rewriter.create<func::CallOp>(op->getLoc(), printFunctionRef,
                                  rewriter.getI64Type(), operands);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

} // namespace decisionforest
} // namespace mlir

namespace mlir {
namespace decisionforest {

void populateDebugOpLoweringPatterns(RewritePatternSet &patterns,
                                     LLVMTypeConverter &typeConverter) {

  patterns.add<PrintfOpLowering>(typeConverter);

  if (InsertDebugHelpers)
    patterns.add<PrintTreePredictionOpLowering, PrintTreeNodeOpLowering,
                 PrintTreeToDOTFileOpLowering, PrintInputRowOpLowering,
                 PrintComparisonOpLowering, PrintIsLeafOpLowering,
                 PrintVectorOpLowering>(typeConverter);
}

void InsertPrintElementAddressIfNeeded(ConversionPatternRewriter &rewriter,
                                       Location location, ModuleOp module,
                                       Value bufferPtr, Value indexVal,
                                       Value actualIndex, Value elemIndex,
                                       Value elemPtr) {
  auto context = rewriter.getContext();

  std::string functionName = "PrintElementAddress";
  // Create a function declaration for PrintElementAddress, the signature is:
  //      int64_t PrintElementAddress(void *bufPtr, int64_t index, int64_t
  //      actualIndex, int32_t elementIndex, void *elemPtr)
  auto llvmI64Ty = IntegerType::get(context, 64);
  auto llvmI32Ty = IntegerType::get(context, 32);
  auto llvmI32PtrTy = LLVM::LLVMPointerType::get(llvmI32Ty);
  auto llvmFnType = LLVM::LLVMFunctionType::get(
      llvmI64Ty, {llvmI32PtrTy, llvmI64Ty, llvmI64Ty, llvmI32Ty, llvmI32PtrTy});

  auto printFunctionRef =
      getOrInsertFunction(functionName, llvmFnType, rewriter, module);
  // Cast the pointers to int* and then pass them
  auto castedBufferPtr =
      rewriter.create<LLVM::BitcastOp>(location, llvmI32PtrTy, bufferPtr);
  auto castedElemPtr =
      rewriter.create<LLVM::BitcastOp>(location, llvmI32PtrTy, elemPtr);

  rewriter.create<func::CallOp>(
      location, printFunctionRef, rewriter.getI64Type(),
      ValueRange{castedBufferPtr, indexVal, actualIndex, elemIndex,
                 castedElemPtr});
}

} // namespace decisionforest
} // namespace mlir
