#ifndef _LIRLOWERINGHELPERS_H_
#define _LIRLOWERINGHELPERS_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialect.h"
#include "OpLoweringUtils.h"

namespace mlir {
namespace decisionforest {
namespace helpers {

class SaveAndRestoreInsertionPoint {
  mlir::OpBuilder::InsertPoint m_insertPoint;
  mlir::ConversionPatternRewriter &m_builder;

public:
  SaveAndRestoreInsertionPoint(mlir::ConversionPatternRewriter &builder)
      : m_builder(builder) {
    m_insertPoint = m_builder.saveInsertionPoint();
  }

  ~SaveAndRestoreInsertionPoint() {
    m_builder.restoreInsertionPoint(m_insertPoint);
  }
};

inline void InsertPrintVectorOp(ConversionPatternRewriter &rewriter,
                                Location location, int32_t kind,
                                int32_t bitWidth, int32_t tileSize,
                                Value vectorValue) {
  auto tileSizeConst = rewriter.create<arith::ConstantIntOp>(
      location, tileSize, rewriter.getI32Type());
  auto kindConst = rewriter.create<arith::ConstantIntOp>(location, kind,
                                                         rewriter.getI32Type());
  auto bitWidthConst = rewriter.create<arith::ConstantIntOp>(
      location, bitWidth, rewriter.getI32Type());
  std::vector<Value> vectorValues;
  for (int32_t i = 0; i < tileSize; ++i) {
    auto iConst = rewriter.create<arith::ConstantIntOp>(location, int64_t(i),
                                                        rewriter.getI32Type());
    auto ithValue = rewriter.create<vector::ExtractElementOp>(
        location, vectorValue, iConst);
    vectorValues.push_back(ithValue);
  }
  rewriter.create<decisionforest::PrintVectorOp>(location, kindConst,
                                                 bitWidthConst, tileSizeConst,
                                                 ValueRange(vectorValues));
}

inline Value CreateZeroVectorFPConst(mlir::OpBuilder &rewriter,
                                     Location location, Type fpType,
                                     int32_t tileSize) {
  Value zeroConst;
  auto vectorType = VectorType::get(tileSize, fpType);
  if (fpType.isa<mlir::Float64Type>())
    zeroConst = rewriter.create<arith::ConstantFloatOp>(
        location, llvm::APFloat(0.0), fpType.cast<FloatType>());
  else if (fpType.isa<mlir::Float32Type>())
    zeroConst = rewriter.create<arith::ConstantFloatOp>(
        location, llvm::APFloat((float)0.0), fpType.cast<FloatType>());
  else
    assert(false && "Unsupported floating point type");
  auto vectorValue =
      rewriter.create<vector::BroadcastOp>(location, vectorType, zeroConst);
  return vectorValue;
}

inline Value CreateZeroVectorIntConst(mlir::OpBuilder &rewriter,
                                      Location location, Type intType,
                                      int32_t tileSize) {
  Value zeroConst = rewriter.create<arith::ConstantIntOp>(location, 0, intType);
  auto vectorType = VectorType::get(tileSize, intType);
  auto vectorValue =
      rewriter.create<vector::BroadcastOp>(location, vectorType, zeroConst);
  return vectorValue;
}

inline Value CreateZeroVectorIndexConst(ConversionPatternRewriter &rewriter,
                                        Location location, int32_t tileSize) {
  Value zeroConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
  auto vectorType = VectorType::get(tileSize, rewriter.getIndexType());
  auto vectorValue =
      rewriter.create<vector::BroadcastOp>(location, vectorType, zeroConst);
  return vectorValue;
}

inline void AddGlobalMemrefGetter(mlir::ModuleOp module, std::string globalName,
                                  Type memrefType,
                                  ConversionPatternRewriter &rewriter,
                                  Location location) {
  SaveAndRestoreInsertionPoint saveAndRestoreEntryPoint(rewriter);
  auto getMemrefFuncType = rewriter.getFunctionType(TypeRange({}), memrefType);
  std::string funcName = "Get_" + globalName;
  NamedAttribute visibilityAttribute{module.getSymVisibilityAttrName(),
                                     rewriter.getStringAttr("public")};
  auto getGlobalMemrefFunc =
      mlir::func::FuncOp::create(location, funcName, getMemrefFuncType,
                                 ArrayRef<NamedAttribute>(visibilityAttribute));
  auto &entryBlock = *getGlobalMemrefFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(&entryBlock);

  auto getGlobalOffsets =
      rewriter.create<memref::GetGlobalOp>(location, memrefType, globalName);
  rewriter.create<mlir::func::ReturnOp>(location,
                                        static_cast<Value>(getGlobalOffsets));

  module.push_back(getGlobalMemrefFunc);
}

inline Value GetTreeIndexValue(Value tree) {
  auto definingOp = tree.getDefiningOp();
  auto getTreeOp =
      AssertOpIsOfType<decisionforest::GetTreeFromEnsembleOp>(definingOp);
  auto treeIndex = getTreeOp.getTreeIndex();
  assert(treeIndex.getType().isa<mlir::IndexType>());
  return treeIndex;
}

} // namespace helpers
} // namespace decisionforest
} // namespace mlir

#endif // _LIRLOWERINGHELPERS_H_