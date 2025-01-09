#ifdef TREEBEARD_GPU_SUPPORT

#include <optional>

#include "mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"


#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
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
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "Dialect.h"
#include "Representations.h"
#include "TreebeardContext.h"

#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "CompileUtils.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"


#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/FormatVariadic.h"



#define DEBUG_TYPE "gpu-to-llvm"


// ---------------------------------------//


using namespace mlir;

namespace mlir {
namespace decisionforest {

FlatSymbolRefAttr getOrInsertCacheOpSyncFunc(std::string &functionName,
                                             PatternRewriter &rewriter,
                                             gpu::GPUModuleOp module) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(functionName))
    return SymbolRefAttr::get(context, functionName);

  LLVM::LLVMFunctionType functionType =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context), {}, false);

  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto func = rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), functionName,
                                                functionType);
  auto entryBlock = func.addEntryBlock(rewriter);
  rewriter.setInsertionPointToStart(entryBlock);

  rewriter.create<gpu::BarrierOp>(module.getLoc());
  rewriter.create<LLVM::ReturnOp>(module.getLoc(), ValueRange{});
  return SymbolRefAttr::get(context, functionName);
}

struct CacheOpBeginOpLowering : public ConversionPattern {
  CacheOpBeginOpLowering(LLVMTypeConverter &typeConverter)
      : ConversionPattern(
            typeConverter,
            mlir::decisionforest::CacheOpBeginOp::getOperationName(),
            1 /*benefit*/, &typeConverter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // assert (operands.size() == 3);
    auto parentModule = op->getParentOfType<gpu::GPUModuleOp>();
    std::string functionName = "CacheOpBeginBarrierFunc";
    auto syncFunctionRef =
        getOrInsertCacheOpSyncFunc(functionName, rewriter, parentModule);

    // auto returnType = LLVM::LLVMVoidType::get(rewriter.getContext());
    rewriter.create<LLVM::CallOp>(op->getLoc(), TypeRange{}, syncFunctionRef,
                                  ValueRange{});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct CacheOpEndOpLowering : public ConversionPattern {
  CacheOpEndOpLowering(LLVMTypeConverter &typeConverter)
      : ConversionPattern(
            typeConverter,
            mlir::decisionforest::CacheOpEndOp::getOperationName(),
            1 /*benefit*/, &typeConverter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // assert (operands.size() == 3);
    auto parentModule = op->getParentOfType<gpu::GPUModuleOp>();
    std::string functionName = "CacheOpEndBarrierFunc";
    auto syncFunctionRef =
        getOrInsertCacheOpSyncFunc(functionName, rewriter, parentModule);

    // auto returnType = LLVM::LLVMVoidType::get(rewriter.getContext());
    rewriter.create<LLVM::CallOp>(op->getLoc(), TypeRange{}, syncFunctionRef,
                                  ValueRange{});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

template <typename DerivedT>
class LowerGpuOpsToTargetBase : public ::mlir::OperationPass<gpu::GPUModuleOp> {
  std::shared_ptr<decisionforest::IRepresentation> m_representation;

public:
  using Base = LowerGpuOpsToTargetBase;

  LowerGpuOpsToTargetBase(
      std::shared_ptr<decisionforest::IRepresentation> representation)
      : ::mlir::OperationPass<gpu::GPUModuleOp>(
            ::mlir::TypeID::get<DerivedT>()),
        m_representation(representation) {}
  LowerGpuOpsToTargetBase(const LowerGpuOpsToTargetBase &other)
      : ::mlir::OperationPass<gpu::GPUModuleOp>(other),
        m_representation(other.m_representation) {}

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  // Run re-writes that do not require a special type converter.
  virtual void populateMemorySpaceConversion(TypeConverter typeConverter) = 0;

    // Run re-writes that do not require a special type converter.
  void populateLowerMemorySpaceOpLegality(ConversionTarget &target);

  // Add rewrites that require the custom type converter
  virtual LogicalResult populateTargetSpecificRewritesAndConversions(
      RewritePatternSet &set, LLVMTypeConverter &typeConverter) = 0;

  virtual void doPostLoweringFixup(gpu::GPUModuleOp module) {
  } // do nothing by default.

  virtual void
  configureTargetConversionLegality(LLVMConversionTarget &target) = 0;

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();

    // This just lowers gpu.allreduce to a bunch of simpler ops from the
    // gpu dialect and other dialects. We probably don't need this for
    // Treebeard?
    {
      RewritePatternSet patterns(m.getContext());
      populateGpuRewritePatterns(patterns);
      (void)applyPatternsAndFoldGreedily(m, std::move(patterns));
    }

    {
      RewritePatternSet patterns(m.getContext());
      TypeConverter typeConverter;
      typeConverter.addConversion([](Type t) { return t; });

      populateMemorySpaceConversion(typeConverter);

      // gpu::populateMemorySpaceLoweringPatterns(typeConverter, patterns);
      ConversionTarget target(getContext());
      // populateLowerMemorySpaceOpLegality(target);
      target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return true; });

      if (failed(applyFullConversion(m, target, std::move(patterns))))
        return signalPassFailure();
    }

    LowerToLLVMOptions options(
        m.getContext(),
        DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));
    LLVMTypeConverter converter(m.getContext(), options);
    RewritePatternSet llvmPatterns(m.getContext());

    arith::populateCeilFloorDivExpandOpsPatterns(llvmPatterns);
    mlir::arith::populateArithToLLVMConversionPatterns(converter, llvmPatterns);
    populateAffineToStdConversionPatterns(llvmPatterns);
    populateSCFToControlFlowConversionPatterns(llvmPatterns);

    if (failed(populateTargetSpecificRewritesAndConversions(llvmPatterns,
                                                            converter))) {
      return signalPassFailure();
    }

    m_representation->AddTypeConversions(*m.getContext(), converter);
    m_representation->AddLLVMConversionPatterns(converter, llvmPatterns);
    decisionforest::populateDebugOpLoweringPatterns(llvmPatterns, converter);
    llvmPatterns.add<CacheOpBeginOpLowering>(converter);
    llvmPatterns.add<CacheOpEndOpLowering>(converter);

    LLVMConversionTarget target(getContext());
    configureTargetConversionLegality(target);
    target.addIllegalDialect<decisionforest::DecisionForestDialect,
                             math::MathDialect>();

    if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
      signalPassFailure();

    doPostLoweringFixup(m);
  }
};

/// A pass that replaces all occurrences of GPU device operations with their
/// corresponding NVVM equivalent.
///
/// This pass only handles device code and is not meant to be run on GPU host
/// code.
struct LowerGpuOpsToNVVMOpsPass
    : public LowerGpuOpsToTargetBase<LowerGpuOpsToNVVMOpsPass> {

  LowerGpuOpsToNVVMOpsPass(
      std::shared_ptr<decisionforest::IRepresentation> representation)
      : LowerGpuOpsToTargetBase(representation) {}
  LowerGpuOpsToNVVMOpsPass(const LowerGpuOpsToNVVMOpsPass &other)
      : LowerGpuOpsToTargetBase(other) {}

  ::llvm::StringRef getName() const override {
    return "Treebeard.LowerGpuOpsToNVVMOpsPass";
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    registry.insert<NVVM::NVVMDialect>();
    registry.insert<gpu::GPUDialect>();
    NVVM::registerNVVMTargetInterfaceExternalModels(registry);
    registerNVVMDialectTranslation(registry);
    registerLLVMDialectTranslation(registry);
  }

   void populateLowerMemorySpaceOpLegality(ConversionTarget &target) {
      target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  }

  void populateMemorySpaceConversion(TypeConverter typeConverter) override {
    // NVVM uses alloca in the default address space to represent private
    // memory allocations, so drop private annotations. NVVM uses address
    // space 3 for shared memory. NVVM uses the default address space to
    // represent global memory.
    mlir::populateGpuMemorySpaceAttributeConversions(
        typeConverter, [](gpu::AddressSpace space) -> unsigned {
          switch (space) {
          case gpu::AddressSpace::Global:
            return static_cast<unsigned>(
                NVVM::NVVMMemorySpace::kGlobalMemorySpace);
          case gpu::AddressSpace::Workgroup:
            return static_cast<unsigned>(
                NVVM::NVVMMemorySpace::kSharedMemorySpace);
          case gpu::AddressSpace::Private:
            return 0;
          }
          llvm_unreachable("unknown address space enum value");
          return 0;
        });
  }

  // Add rewrites that require the custom type converter
  LogicalResult populateTargetSpecificRewritesAndConversions(
      RewritePatternSet &llvmPatterns, LLVMTypeConverter &converter) override {
    cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
    populateFuncToLLVMConversionPatterns(converter, llvmPatterns);
    populateVectorToLLVMConversionPatterns(converter, llvmPatterns);
    populateFinalizeMemRefToLLVMConversionPatterns(converter, llvmPatterns);
    populateMathToLLVMConversionPatterns(converter, llvmPatterns);
    populateGpuToNVVMConversionPatterns(converter, llvmPatterns);
    populateGpuWMMAToNVVMConversionPatterns(converter, llvmPatterns);
    populateGpuSubgroupReduceOpLoweringPattern(converter, llvmPatterns);
    return LogicalResult::success();
  }

  void
  configureTargetConversionLegality(LLVMConversionTarget &target) override {
    configureGpuToNVVMConversionLegality(target);
  }
};

/// A pass that replaces all occurrences of GPU device operations with their
/// corresponding ROCDL equivalent. This is used for AMD gpus
///
/// This pass only handles device code and is not meant to be run on GPU host
/// code.
struct LowerGpuOpsToROCDLOpsPass
    : public LowerGpuOpsToTargetBase<LowerGpuOpsToROCDLOpsPass> {

private:
  gpu::amd::Runtime m_runtime;
  std::string m_chipset;

public:
  LowerGpuOpsToROCDLOpsPass(
      std::string chipset, gpu::amd::Runtime runtime,
      std::shared_ptr<decisionforest::IRepresentation> representation)
      : LowerGpuOpsToTargetBase(representation), m_runtime(runtime),
        m_chipset(chipset) {}
  LowerGpuOpsToROCDLOpsPass(const LowerGpuOpsToROCDLOpsPass &other)
      : LowerGpuOpsToTargetBase(other), m_runtime(other.m_runtime),
        m_chipset(other.m_chipset) {}

  ::llvm::StringRef getName() const override {
    return "Treebeard.LowerGpuOpsToROCDLOpsPass";
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<ROCDL::ROCDLDialect>();
    registry.insert<cf::ControlFlowDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void populateMemorySpaceConversion(TypeConverter typeConverter) override {
    mlir::populateGpuMemorySpaceAttributeConversions(
        typeConverter, [](gpu::AddressSpace space) {
          switch (space) {
          case gpu::AddressSpace::Global:
            return 1;
          case gpu::AddressSpace::Workgroup:
            return 3;
          case gpu::AddressSpace::Private:
            return 5;
          }
          llvm_unreachable("unknown address space enum value");
          return 0;
        });
  }

  // Add rewrites that require the custom type converter
  LogicalResult populateTargetSpecificRewritesAndConversions(
      RewritePatternSet &llvmPatterns, LLVMTypeConverter &converter) override {

    FailureOr<amdgpu::Chipset> maybeChipset = amdgpu::Chipset::parse(m_chipset);
    if (failed(maybeChipset)) {
      emitError(UnknownLoc::get(llvmPatterns.getContext()),
                "Invalid chipset name: " + m_chipset);
      maybeChipset;
    }

    populateAMDGPUToROCDLConversionPatterns(converter, llvmPatterns,
                                            *maybeChipset);
    populateVectorToLLVMConversionPatterns(converter, llvmPatterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
    populateFuncToLLVMConversionPatterns(converter, llvmPatterns);
    populateFinalizeMemRefToLLVMConversionPatterns(converter, llvmPatterns);
    populateGpuToROCDLConversionPatterns(converter, llvmPatterns, m_runtime);

    return LogicalResult::success();
  }

  void
  configureTargetConversionLegality(LLVMConversionTarget &target) override {
    configureGpuToROCDLConversionLegality(target);
  }

  void doPostLoweringFixup(gpu::GPUModuleOp module) override {
    // Manually rewrite known block size attributes so the LLVMIR translation
    // infrastructure can pick them up.
    module.walk([ctx = module.getContext()](LLVM::LLVMFuncOp op) {
      OperationName llvmFuncOpName(LLVM::LLVMFuncOp::getOperationName(), ctx);
      if (auto blockSizes =
              op->removeAttr(gpu::GPUFuncOp::getKnownBlockSizeAttrName(llvmFuncOpName))
                  .dyn_cast_or_null<DenseI32ArrayAttr>()) {
        op->setAttr(ROCDL::ROCDLDialect::getReqdWorkGroupSizeAttrName(),
                    blockSizes);
        // Also set up the rocdl.flat_work_group_size attribute to prevent
        // conflicting metadata.
        uint32_t flatSize = 1;
        for (uint32_t size : blockSizes.asArrayRef()) {
          flatSize *= size;
        }
        StringAttr flatSizeAttr =
            StringAttr::get(ctx, Twine(flatSize) + "," + Twine(flatSize));
        op->setAttr(ROCDL::ROCDLDialect::getFlatWorkGroupSizeAttrName(),
                    flatSizeAttr);
      }
    });
  }
};

template <typename DerivedT>
class GpuToLLVMConversionPassBase : public ::mlir::OperationPass<ModuleOp> {
public:
  using Base = GpuToLLVMConversionPassBase;

  GpuToLLVMConversionPassBase()
      : ::mlir::OperationPass<ModuleOp>(::mlir::TypeID::get<DerivedT>()) {}
  GpuToLLVMConversionPassBase(const GpuToLLVMConversionPassBase &other)
      : ::mlir::OperationPass<ModuleOp>(other) {}

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("gpu-to-llvm");
  }
  ::llvm::StringRef getArgument() const override { return "gpu-to-llvm"; }

  ::llvm::StringRef getDescription() const override {
    return "Convert GPU dialect to LLVM dialect with GPU runtime calls";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("GpuToLLVMConversionPass");
  }
  ::llvm::StringRef getName() const override {
    return "GpuToLLVMConversionPass";
  }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    // Base::getDependentDialects(registry);
    arith::registerConvertArithToLLVMInterface(registry);
    registerConvertComplexToLLVMInterface(registry);
    cf::registerConvertControlFlowToLLVMInterface(registry);
    registerConvertFuncToLLVMInterface(registry);
    index::registerConvertIndexToLLVMInterface(registry);
    registerConvertMathToLLVMInterface(registry);
    registerConvertMemRefToLLVMInterface(registry);
    registerConvertNVVMToLLVMInterface(registry);
    ub::registerConvertUBToLLVMInterface(registry);
    // mlir::registerGPUDialectTranslation(registry);
  }

  /// Explicitly declare the TypeID for this class. We declare an explicit
  /// private instantiation because Pass classes should only be visible by the
  /// current library.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GpuToLLVMConversionPassBase<DerivedT>)

protected:
};

class GpuToLLVMConversionPass
    : public GpuToLLVMConversionPassBase<GpuToLLVMConversionPass> {

  std::shared_ptr<decisionforest::IRepresentation> m_representation;

public:
  GpuToLLVMConversionPass(
      std::shared_ptr<decisionforest::IRepresentation> representation)
      : m_representation(representation) {}

  GpuToLLVMConversionPass(const GpuToLLVMConversionPass &other)
      : GpuToLLVMConversionPassBase(other),
        m_representation(other.m_representation) {}

  void getDependentDialects(DialectRegistry &registry) const final {
    Base::getDependentDialects(registry);
    registerConvertToLLVMDependentDialectLoading(registry);
  }
  // Run the dialect converter on the module.
  void runOnOperation() override;

private:
  // Option<std::string> gpuBinaryAnnotation{
  //     *this, "gpu-binary-annotation",
  //     llvm::cl::desc("Annotation attribute string for GPU binary"),
  //     llvm::cl::init(gpu::getDefaultGpuBinaryAnnotation())};
};

void GpuToLLVMConversionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  LowerToLLVMOptions options(context);
  options.useBarePtrCallConv = false;
  RewritePatternSet patterns(context);
  ConversionTarget target(*context);
  target.addLegalDialect<LLVM::LLVMDialect>();
  LLVMTypeConverter converter(context, options);

  // target.addIllegalDialect<gpu::GPUDialect>();
  target.addIllegalDialect<decisionforest::DecisionForestDialect,
                           math::MathDialect>();


  // Populate all patterns from all dialects that implement the
  // `ConvertToLLVMPatternInterface` interface.
  for (Dialect *dialect : context->getLoadedDialects()) {
    auto iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect);
    if (!iface)
      continue;
    iface->populateConvertToLLVMConversionPatterns(target, converter, patterns);
  }

  // Preserve GPU modules and binaries. Modules are preserved as they can be
  // converted later by `gpu-module-to-binary`.
  target.addLegalOp<gpu::GPUModuleOp, gpu::BinaryOp>();
  // Accept as legal LaunchFuncOps if the operands have been lowered.
  target.addDynamicallyLegalOp<gpu::LaunchFuncOp>(
      [&](gpu::LaunchFuncOp op) -> bool { return converter.isLegal(op); });


  m_representation->AddTypeConversions(getContext(), converter);

  mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(converter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  populateMathToLLVMConversionPatterns(converter, patterns);
  populateAsyncStructuralTypeConversionsAndLegality(converter, patterns,
                                                    target);
  populateGpuToLLVMConversionPatterns(converter, patterns /*, gpuBinaryAnnotation */);
  populateAffineToStdConversionPatterns(patterns);
  populateMathToLLVMConversionPatterns(converter, patterns);
  decisionforest::populateDebugOpLoweringPatterns(patterns, converter);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

struct PrintModulePass
    : public PassWrapper<PrintModulePass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect, mlir::affine::AffineDialect,
                    memref::MemRefDialect, arith::ArithDialect,
                    vector::VectorDialect, omp::OpenMPDialect>();
  }
  void runOnOperation() final {
    auto module = getOperation();
    module->dump();
  }
};

} // namespace decisionforest
} // namespace mlir

namespace mlir {
namespace decisionforest {

std::unique_ptr<mlir::Pass> createConvertGlobalsToWorkgroupAllocationsPass();
std::unique_ptr<mlir::Pass> createDeleteSharedMemoryGlobalsPass(
    int32_t &sharedMemorySize,
    std::shared_ptr<decisionforest::IRepresentation> representation);

void InitializeGPUTarget(TreeBeard::GPUCompileInfo &compileInfo) {
#ifdef TREEBEARD_NV_GPU_SUPPORT
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
#elif defined(TREEBEARD_AMD_GPU_SUPPORT)
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUAsmPrinter();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
#endif // GPU support
}

void registerTranslations(MLIRContext& context) {
  DialectRegistry registry;
  registerBuiltinDialectTranslation(registry);
  registerAllGPUToLLVMIRTranslations(registry);
  // registerGPUDialectTranslation(registry);
  registerNVVMDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}

void LowerGPUToLLVM(
    mlir::MLIRContext &context, mlir::ModuleOp module,
    std::shared_ptr<decisionforest::IRepresentation> representation,
    TreeBeard::GPUCompileInfo &compileInfo) {
  InitializeGPUTarget(compileInfo);
  // llvm::DebugFlag = true;
  // Lower from high-level IR to mid-level IR  

  mlir::PassManager pm(&context);
  
  // Call the function to enable IR printing if PRINT_AFTER_ALL is set
   TreeBeard::EnablePrintIRAfter(context, pm);

  pm.addPass(createConvertNVGPUToNVVMPass());
  // pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createGpuKernelOutliningPass());
  // pm.addPass(std::make_unique<PrintModulePass>());
  pm.addPass(createConvertVectorToSCFPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createConvertNVVMToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
#ifdef TREEBEARD_NV_GPU_SUPPORT
 // Set up options for NVIDIA GPU
  GpuNVVMAttachTargetOptions nvvmTargetOptions;

  // Assign the specified values to gpuModuleToBinaryPassOptions
  nvvmTargetOptions.triple = "nvptx64-nvidia-cuda";   // Set the triple value
  nvvmTargetOptions.chip = "sm_50";                   // Set the chip value
  nvvmTargetOptions.features = "+ptx60";               // Set the features value
  nvvmTargetOptions.optLevel = 3;                      // Set the optimization level to 3
  pm.addPass(createGpuNVVMAttachTarget(nvvmTargetOptions));
#elif defined(TREEBEARD_AMD_GPU_SUPPORT)
    // Set up options for AMD GPU
  GpuROCDLAttachTargetOptions amdTargetOptions;
  amdTargetOptions.triple = "amdgcn-amd-amdhsa";          // Hardcoded for AMD
  amdTargetOptions.chip = TREEBEARD_AMD_GPU_CHIPSET;     // Use your defined constant
  amdTargetOptions.features = "";                       // Set any required features if needed
  amdTargetOptions.optLevel = 3;                        // Set the optimization level to 3 for AMD
  pm.addPass(createGpuROCDLAttachTarget(amdTargetOptions));
#endif

  pm.addPass(createLowerAffinePass());
  pm.addPass(createArithToLLVMConversionPass());
  ConvertIndexToLLVMPassOptions convertIndexToLLVMPassOpt;
  convertIndexToLLVMPassOpt.indexBitwidth = 64;
  pm.addPass(createConvertIndexToLLVMPass(convertIndexToLLVMPassOpt));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  
  // pm.addPass(createMemRefToLLVMPass());
  pm.addNestedPass<gpu::GPUModuleOp>(
      createConvertGlobalsToWorkgroupAllocationsPass());
  pm.addPass(createDeleteSharedMemoryGlobalsPass(
      compileInfo.sharedMemoryInBytes, representation));
  pm.addNestedPass<gpu::GPUModuleOp>(createStripDebugInfoPass());
  // pm.addPass(std::make_unique<PrintModulePass>());
#ifdef TREEBEARD_NV_GPU_SUPPORT
  pm.addNestedPass<gpu::GPUModuleOp>(
      std::make_unique<LowerGpuOpsToNVVMOpsPass>(representation));
#elif defined(TREEBEARD_AMD_GPU_SUPPORT)
  pm.addNestedPass<gpu::GPUModuleOp>(
      std::make_unique<LowerGpuOpsToROCDLOpsPass>(TREEBEARD_AMD_GPU_CHIPSET,
                                                  gpu::amd::Runtime::Unknown,
                                                  representation));
#endif // GPU support

  // pm.addPass(std::make_unique<PrintModulePass>());
  pm.addNestedPass<gpu::GPUModuleOp>(createCanonicalizerPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createReconcileUnrealizedCastsPass());
  // pm.addPass(std::make_unique<PrintModulePass>());
  pm.addPass(std::make_unique<GpuToLLVMConversionPass>(representation));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  GpuModuleToBinaryPassOptions gpuModuleToBinaryPassOptions;
  gpuModuleToBinaryPassOptions.compilationTarget = "fatbin";
  registerTranslations(context);
  pm.addPass(createGpuModuleToBinaryPass(gpuModuleToBinaryPassOptions));
  // pm.addPass(std::make_unique<PrintModulePass>());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  // pm.addPass(std::make_unique<PrintModulePass>());

  // module->dump();

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to LLVM failed.\n";
  }
  // module->dump();
  llvm::DebugFlag = false;
}
} // namespace decisionforest
} // namespace mlir

#endif