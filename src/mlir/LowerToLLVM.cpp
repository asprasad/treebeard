// #include "Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Dialect/MemRef/Transforms/Passes.h"

#include "Dialect.h"
#include "Representations.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

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

using namespace mlir;

namespace mlir {
namespace decisionforest {
// Defined in LowerDebugHelpers.cpp
void InsertPrintElementAddressIfNeeded(ConversionPatternRewriter &rewriter,
                                       Location location, ModuleOp module,
                                       Value extractMemrefBufferPointer,
                                       Value indexVal, Value actualIndex,
                                       Value elemIndexConst, Value elemPtr);
} // namespace decisionforest
} // namespace mlir

namespace {

struct DecisionForestToLLVMLoweringPass
    : public PassWrapper<DecisionForestToLLVMLoweringPass,
                         OperationPass<ModuleOp>> {
  std::shared_ptr<decisionforest::IRepresentation> m_representation;

  DecisionForestToLLVMLoweringPass(
      std::shared_ptr<mlir::decisionforest::IRepresentation> representation)
      : m_representation(representation) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect, AffineDialect,
                    memref::MemRefDialect, arith::ArithDialect,
                    vector::VectorDialect, omp::OpenMPDialect>();
  }
  void runOnOperation() final;
};

void DecisionForestToLLVMLoweringPass::runOnOperation() {
  // define the conversion target
  LowerToLLVMOptions options(&getContext());
  // options.useBarePtrCallConv = true;
  // options.emitCWrappers = true;
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp, func::CallOp>();
  target.addLegalDialect<scf::SCFDialect, omp::OpenMPDialect>();
  // target.addLegalDialect<gpu::GPUDialect, NVVM::NVVMDialect>();
  target.addIllegalDialect<decisionforest::DecisionForestDialect,
                           memref::MemRefDialect>();
  target.addIllegalDialect<arith::ArithDialect, vector::VectorDialect,
                           math::MathDialect>();

  auto &context = getContext();
  LLVMTypeConverter typeConverter(&context, options);

  RewritePatternSet patterns(&context);
  populateAffineToStdConversionPatterns(patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateVectorToLLVMConversionPatterns(typeConverter, patterns, false);
  populateMathToLLVMConversionPatterns(typeConverter, patterns);
  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  m_representation->AddTypeConversions(context, typeConverter);
  m_representation->AddLLVMConversionPatterns(typeConverter, patterns);
  decisionforest::populateDebugOpLoweringPatterns(patterns, typeConverter);

  auto module = getOperation();

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
    llvm::errs() << "Decision forest lowering pass failed\n";
  }
  // module->dump();
}

struct LowerOMPToLLVMPass
    : public PassWrapper<LowerOMPToLLVMPass, OperationPass<ModuleOp>> {
  std::shared_ptr<decisionforest::IRepresentation> m_representation;

  LowerOMPToLLVMPass(
      std::shared_ptr<decisionforest::IRepresentation> representation)
      : m_representation(representation) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<LLVM::LLVMDialect, scf::SCFDialect, AffineDialect,
                memref::MemRefDialect, arith::ArithDialect,
                vector::VectorDialect, omp::OpenMPDialect, func::FuncDialect>();
  }

  void runOnOperation() final {
    LowerToLLVMOptions options(&getContext());
    auto module = getOperation();
    auto &context = getContext();

    LLVMTypeConverter converter(&getContext(), options);
    m_representation->AddTypeConversions(context, converter);

    // Convert to OpenMP operations with LLVM IR dialect
    RewritePatternSet patterns(&getContext());
    arith::populateArithToLLVMConversionPatterns(converter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    populateMemRefToLLVMConversionPatterns(converter, patterns);
    populateFuncToLLVMConversionPatterns(converter, patterns);
    populateOpenMPToLLVMConversionPatterns(converter, patterns);

    LLVMConversionTarget target(getContext());
    target.addLegalOp<omp::TerminatorOp, omp::TaskyieldOp, omp::FlushOp,
                      omp::BarrierOp, omp::TaskwaitOp>();
    target.addLegalDialect<gpu::GPUDialect, NVVM::NVVMDialect>();
    configureOpenMPToLLVMConversionLegality(target, converter);
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

struct PrintModulePass
    : public PassWrapper<PrintModulePass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect, AffineDialect,
                    memref::MemRefDialect, arith::ArithDialect,
                    vector::VectorDialect, omp::OpenMPDialect>();
  }
  void runOnOperation() final {
    auto module = getOperation();
    module->dump();
  }
};

} // end anonymous namespace

namespace mlir {
namespace decisionforest {

void LowerToLLVM(mlir::MLIRContext &context, mlir::ModuleOp module,
                 std::shared_ptr<IRepresentation> representation) {
  // llvm::DebugFlag = true;
  // Lower from low-level IR to LLVM IR
  mlir::PassManager pm(&context);
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(
      std::make_unique<DecisionForestToLLVMLoweringPass>(representation));
  pm.addPass(createConvertSCFToOpenMPPass());
  // pm.addPass(std::make_unique<PrintModulePass>());
  pm.addPass(createMemRefToLLVMConversionPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(std::make_unique<LowerOMPToLLVMPass>(representation));
  pm.addPass(createReconcileUnrealizedCastsPass());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to LLVM failed.\n";
  }
  // module->dump();
  // llvm::DebugFlag = false;
}

// ===---------------------------------------------------=== //
// LLVM debugging helper methods
// ===---------------------------------------------------=== //

void dumpAssembly(const llvm::Module *llvmModule) {
  LLVMMemoryBufferRef bufferOut;
  char *errorMessage = nullptr;
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(llvmModule->getTargetTriple(), error);

  if (!target) {
    llvm::errs() << "No target found";
  } else if (target->hasTargetMachine()) {

    // TODO - Revisit these defaults
    auto CPU = "generic";
    auto Features = "";
    llvm::TargetOptions opt;
    auto RM = Optional<llvm::Reloc::Model>();

    auto tm = target->createTargetMachine(llvmModule->getTargetTriple(), CPU,
                                          Features, opt, RM);

    LLVMTargetMachineEmitToMemoryBuffer(
        (LLVMTargetMachineRef)
            tm, // #TODO - Should use a llvm::wrap function. Didn't find one.
        llvm::wrap(llvmModule), LLVMCodeGenFileType::LLVMAssemblyFile,
        &errorMessage, // #TODO - This buffer and the buffer below might leak.
                       // Look at it later.
        &bufferOut);

    if (errorMessage)
      llvm::errs() << errorMessage;
    else {
      llvm::errs() << "<ASM Target =" << target->getName() << ">\n";
      llvm::errs() << llvm::unwrap(bufferOut)->getBuffer() << "\n";
      llvm::errs() << "</ASM>"
                   << "\n";
    }
  } else {
    llvm::errs() << "Target machine not found";
  }
}

int dumpLLVMIR(mlir::ModuleOp module, bool dumpAsm) {
  // Init LLVM targets
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::registerLLVMDialectTranslation(*module->getContext());
  mlir::registerOpenMPDialectTranslation(*module->getContext());

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

int dumpLLVMIRToFile(mlir::ModuleOp module, const std::string &filename) {
  // Init LLVM targets
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::registerLLVMDialectTranslation(*module->getContext());
  mlir::registerOpenMPDialectTranslation(*module->getContext());

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

} // namespace decisionforest
} // namespace mlir