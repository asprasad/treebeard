#ifdef TREEBEARD_GPU_SUPPORT

#include <optional>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/Sequence.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "Dialect.h"
#include "OpLoweringUtils.h"

using namespace mlir;

namespace {

struct SharedMemoryGlobalRewritePattern : public ConversionPattern {
  std::unique_ptr<std::map<std::string, Value>> globalToAttributionMap;

  SharedMemoryGlobalRewritePattern(MLIRContext *ctx)
      : ConversionPattern(mlir::memref::GetGlobalOp::getOperationName(),
                          1 /*benefit*/, ctx),
        globalToAttributionMap(
            std::make_unique<std::map<std::string, Value>>()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    auto location = op->getLoc();
    auto getGlobalOp = AssertOpIsOfType<memref::GetGlobalOp>(op);
    auto globalType = getGlobalOp.getResult().getType();
    if (!globalType.isa<MemRefType>())
      return mlir::failure();
    auto globalMemrefType = globalType.cast<MemRefType>();
    auto memorySpaceAttr = globalMemrefType.getMemorySpace();
    if (!memorySpaceAttr)
      return mlir::failure();

    std::string globalName = getGlobalOp.getName().str();
    if (globalToAttributionMap->find(globalName) !=
        globalToAttributionMap->end()) {
      rewriter.replaceOp(op, globalToAttributionMap->at(globalName));
      return mlir::success();
    }
    auto memorySpace =
        memorySpaceAttr.cast<IntegerAttr>().getValue().getSExtValue();
    if (memorySpace != 3) // not a shared memory buffer
      return mlir::failure();

    // Assume that we've run gpu-outlining pass before we get here.
    auto gpuFuncOp = op->getParentOfType<gpu::GPUFuncOp>();
    if (!gpuFuncOp)
      return mlir::failure();

    Value attribution =
        gpuFuncOp.addWorkgroupAttribution(globalMemrefType, location);

    rewriter.replaceOp(op, attribution);
    (*globalToAttributionMap)[globalName] = attribution;

    return mlir::success();
  }
};

class ConvertGlobalsToWorkgroupAllocations
    : public ::mlir::OperationPass<gpu::GPUModuleOp> {
public:
  ConvertGlobalsToWorkgroupAllocations()
      : ::mlir::OperationPass<gpu::GPUModuleOp>(
            ::mlir::TypeID::get<ConvertGlobalsToWorkgroupAllocations>()) {}

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();
    ConversionTarget target(getContext());

    target.addLegalDialect<
        AffineDialect, memref::MemRefDialect, scf::SCFDialect,
        decisionforest::DecisionForestDialect, vector::VectorDialect,
        math::MathDialect, arith::ArithDialect, func::FuncDialect,
        gpu::GPUDialect>();

    target.addIllegalOp<memref::GetGlobalOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<SharedMemoryGlobalRewritePattern>(patterns.getContext());

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
  }

  ::llvm::StringRef getName() const override {
    return "ConvertGlobalsToWorkgroupAllocations";
  }

  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<ConvertGlobalsToWorkgroupAllocations>(*this);
  }
};

struct DeleteSharedMemoryGlobalsPattern : public ConversionPattern {
  DeleteSharedMemoryGlobalsPattern(MLIRContext *ctx)
      : ConversionPattern(mlir::memref::GlobalOp::getOperationName(),
                          1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto globalOp = AssertOpIsOfType<memref::GlobalOp>(op);
    auto globalType = globalOp.getType();
    if (!globalType.isa<MemRefType>())
      return mlir::failure();
    auto globalMemrefType = globalType.cast<MemRefType>();
    auto memorySpaceAttr = globalMemrefType.getMemorySpace();
    if (!memorySpaceAttr)
      return mlir::failure();
    auto memorySpace =
        memorySpaceAttr.cast<IntegerAttr>().getValue().getSExtValue();
    if (memorySpace != 3) // not a shared memory buffer
      return mlir::failure();
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class DeleteSharedMemoryGlobals : public ::mlir::OperationPass<ModuleOp> {
public:
  DeleteSharedMemoryGlobals()
      : ::mlir::OperationPass<ModuleOp>(
            ::mlir::TypeID::get<DeleteSharedMemoryGlobals>()) {}
  DeleteSharedMemoryGlobals(const DeleteSharedMemoryGlobals &other)
      : ::mlir::OperationPass<ModuleOp>(other) {}

  ::llvm::StringRef getName() const override {
    return "DeleteSharedMemoryGlobals";
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DeleteSharedMemoryGlobals>(*this);
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());

    target.addLegalDialect<
        AffineDialect, memref::MemRefDialect, scf::SCFDialect,
        decisionforest::DecisionForestDialect, vector::VectorDialect,
        math::MathDialect, arith::ArithDialect, func::FuncDialect,
        gpu::GPUDialect>();

    target.addIllegalOp<memref::GlobalOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<DeleteSharedMemoryGlobalsPattern>(patterns.getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
      llvm::errs() << "Delete shared memory globals failed.\n";
    }
  }
};

} // Anonymous namespace

namespace mlir {
namespace decisionforest {

std::unique_ptr<mlir::Pass> createConvertGlobalsToWorkgroupAllocationsPass() {
  return std::make_unique<ConvertGlobalsToWorkgroupAllocations>();
}

std::unique_ptr<mlir::Pass> createDeleteSharedMemoryGlobalsPass() {
  return std::make_unique<DeleteSharedMemoryGlobals>();
}

} // namespace decisionforest
} // namespace mlir

#endif // TREEBEARD_GPU_SUPPORT