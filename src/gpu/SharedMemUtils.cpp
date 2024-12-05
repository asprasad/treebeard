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
#include "Representations.h"

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
        mlir::affine::AffineDialect, memref::MemRefDialect, scf::SCFDialect,
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
  int32_t &m_sharedMemorySize;
  std::shared_ptr<decisionforest::IRepresentation> m_representation;
  DeleteSharedMemoryGlobalsPattern(
      MLIRContext *ctx, int32_t &sharedMemorySize,
      std::shared_ptr<decisionforest::IRepresentation> representation)
      : ConversionPattern(mlir::memref::GlobalOp::getOperationName(),
                          1 /*benefit*/, ctx),
        m_sharedMemorySize(sharedMemorySize), m_representation(representation) {
  }

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

    if (op->getParentOfType<gpu::GPUModuleOp>()) {
      int32_t numElems = 1;
      for (auto dim : globalMemrefType.getShape())
        numElems *= dim;
      auto elemType = globalMemrefType.getElementType();
      if (elemType.isIntOrFloat()) {
        m_sharedMemorySize += numElems * elemType.getIntOrFloatBitWidth() / 8;
      } else {
        auto numBits = m_representation->getTypeBitWidth(elemType);
        m_sharedMemorySize += numElems * numBits / 8;
        // std::cout << op << " " << numElems << " " << numBits << "\n";
      }
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class DeleteSharedMemoryGlobals : public ::mlir::OperationPass<ModuleOp> {
public:
  int32_t &m_sharedMemorySize;
  std::shared_ptr<decisionforest::IRepresentation> m_representation;
  const int32_t MAX_SHARED_MEMORY_SIZE = 49152;
  DeleteSharedMemoryGlobals(
      int32_t &sharedMemorySize,
      std::shared_ptr<decisionforest::IRepresentation> representation)
      : ::mlir::OperationPass<ModuleOp>(
            ::mlir::TypeID::get<DeleteSharedMemoryGlobals>()),
        m_sharedMemorySize(sharedMemorySize), m_representation(representation) {
    m_sharedMemorySize = 0;
  }
  DeleteSharedMemoryGlobals(const DeleteSharedMemoryGlobals &other)
      : ::mlir::OperationPass<ModuleOp>(other),
        m_sharedMemorySize(other.m_sharedMemorySize),
        m_representation(other.m_representation) {}

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
        mlir::affine::AffineDialect, memref::MemRefDialect, scf::SCFDialect,
        decisionforest::DecisionForestDialect, vector::VectorDialect,
        math::MathDialect, arith::ArithDialect, func::FuncDialect,
        gpu::GPUDialect>();

    // target.addIllegalOp<memref::GlobalOp>();
    target.addDynamicallyLegalOp<memref::GlobalOp>([](memref::GlobalOp op) {
      auto globalType = op.getType();
      if (!globalType.isa<MemRefType>())
        return true;
      auto globalMemrefType = globalType.cast<MemRefType>();
      auto memorySpaceAttr = globalMemrefType.getMemorySpace();
      if (!memorySpaceAttr)
        return true;
      auto memorySpace =
          memorySpaceAttr.cast<IntegerAttr>().getValue().getSExtValue();
      if (memorySpace != 3) // not a shared memory buffer
        return true;
      return false;
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<DeleteSharedMemoryGlobalsPattern>(
        patterns.getContext(), m_sharedMemorySize, m_representation);
    // getOperation()->dump();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
      llvm::errs() << "Delete shared memory globals failed.\n";
    }

    if (m_sharedMemorySize >= MAX_SHARED_MEMORY_SIZE) {
      llvm::errs() << "Shared memory size exceeds maximum allowed size.\n";
      signalPassFailure();
    }
  }
};

} // Anonymous namespace

namespace mlir {
namespace decisionforest {

std::unique_ptr<mlir::Pass> createConvertGlobalsToWorkgroupAllocationsPass() {
  return std::make_unique<ConvertGlobalsToWorkgroupAllocations>();
}

std::unique_ptr<mlir::Pass> createDeleteSharedMemoryGlobalsPass(
    int32_t &sharedMemorySize,
    std::shared_ptr<decisionforest::IRepresentation> representation) {
  return std::make_unique<DeleteSharedMemoryGlobals>(sharedMemorySize,
                                                     representation);
}

} // namespace decisionforest
} // namespace mlir

#endif // TREEBEARD_GPU_SUPPORT