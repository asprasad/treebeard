#include "Dialect.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

} // anonymous namespace

namespace mlir
{
namespace decisionforest
{

void populateDebugOpLoweringPatterns(RewritePatternSet& patterns, LLVMTypeConverter& typeConverter) {
  if (InsertDebugHelpers)
    patterns.add<PrintTreePredictionOpLowering,
                 PrintTreeNodeOpLowering>(typeConverter);
}  

} // decisionforest
} // mlir
