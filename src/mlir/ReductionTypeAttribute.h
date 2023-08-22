#ifndef _REDUCTIONTYPEATTRIBUTE_H_
#define _REDUCTIONTYPEATTRIBUTE_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <string>

namespace mlir {
namespace decisionforest {
enum Reduction {
  kAdd,
  kSoftMax,
};

struct ReductionTypeAttrStorage : public ::mlir::AttributeStorage {
  ReductionTypeAttrStorage(::mlir::Type type, Reduction reductionType)
      : m_reductionType(reductionType) {}

  using KeyTy = std::tuple<::mlir::Type, mlir::decisionforest::Reduction>;

  bool operator==(const KeyTy &tblgenKey) const {
    if (!(m_reductionType == std::get<1>(tblgenKey)))
      return false;
    return true;
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    auto reductionType = std::get<1>(tblgenKey);
    return ::llvm::hash_combine(std::get<0>(tblgenKey), reductionType);
  }

  std::string getReductionTypeString() {
    switch (m_reductionType) {
    case Reduction::kAdd:
      return "Add";
    case Reduction::kSoftMax:
      return "SoftMax";
    default:
      assert(false && "Unknown reduction type");
    }
  }

  static ReductionTypeAttrStorage *
  construct(::mlir::AttributeStorageAllocator &allocator,
            const KeyTy &tblgenKey) {
    auto type = std::get<0>(tblgenKey);
    auto reductionType = std::get<1>(tblgenKey);
    return new (allocator.allocate<ReductionTypeAttrStorage>())
        ReductionTypeAttrStorage(type, reductionType);
  }
  Reduction m_reductionType;
};

class ReductionAttrType
    : public mlir::Type::TypeBase<ReductionAttrType, mlir::Type, TypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;
  void print(mlir::DialectAsmPrinter &printer) {
    printer << "ReductionAttrType";
  }
};

class ReductionTypeAttribute
    : public ::mlir::Attribute::AttrBase<
          ReductionTypeAttribute, ::mlir::Attribute, ReductionTypeAttrStorage> {
public:
  /// Inherit some necessary constructors from 'AttrBase'.
  using Base::Base;
  // using ValueType = APInt;
  static ReductionTypeAttribute get(Type type, Reduction reductionType) {
    return Base::get(type.getContext(), type, reductionType);
  }

  Reduction getReductionType() { return getImpl()->m_reductionType; }

  void Print(::mlir::DialectAsmPrinter &os) {
    std::string reductionType = getImpl()->getReductionTypeString();
    os << "Reduction = " << reductionType;
  }
};

} // namespace decisionforest
} // namespace mlir

#endif // _REDUCTIONTYPEATTRIBUTE_H_