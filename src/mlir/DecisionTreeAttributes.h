#ifndef _TREEATTRIBUTES_H_
#define _TREEATTRIBUTES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "DecisionForest.h"

namespace mlir 
{
namespace decisionforest
{
namespace detail
{
struct DecisionTreeAttrStorage : public ::mlir::AttributeStorage
{
    DecisionTreeAttrStorage(::mlir::Type type, int value)
      : ::mlir::AttributeStorage(type), value(value) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<::mlir::Type, int>;

    bool operator==(const KeyTy &tblgenKey) const {
        if (!(getType() == std::get<0>(tblgenKey)))
            return false;
        if (!(value == std::get<1>(tblgenKey)))
            return false;
        return true;
    }
    
    static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
        return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static DecisionTreeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &tblgenKey) {
      auto type = std::get<0>(tblgenKey);
      auto value = std::get<1>(tblgenKey);

      return new (allocator.allocate<DecisionTreeAttrStorage>())
          DecisionTreeAttrStorage(type, value);
    }
    int value;
};

// TODO How do we use templatization of DecisionForest here?
struct DecisionForestAttrStorage : public ::mlir::AttributeStorage
{
    DecisionForestAttrStorage(::mlir::Type type, const DecisionForest<>& forest)
      : ::mlir::AttributeStorage(type), m_forest(forest) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<::mlir::Type, DecisionForest<>>;

    bool operator==(const KeyTy &tblgenKey) const {
        if (!(getType() == std::get<0>(tblgenKey)))
            return false;
        if (!(m_forest == std::get<1>(tblgenKey)))
            return false;
        return true;
    }
    
    static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
        auto& forest = std::get<1>(tblgenKey);
        return ::llvm::hash_combine(std::get<0>(tblgenKey), forest.Serialize());
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static DecisionForestAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &tblgenKey) {
      auto type = std::get<0>(tblgenKey);
      auto forest = std::get<1>(tblgenKey);

      return new (allocator.allocate<DecisionForestAttrStorage>())
          DecisionForestAttrStorage(type, forest);
    }
    DecisionForest<> m_forest;
};

} // namespace detail

class DecisionTreeAttr : public ::mlir::Attribute::AttrBase<DecisionTreeAttr, ::mlir::Attribute,
                                         detail::DecisionTreeAttrStorage>
{
public:
    /// Inherit some necessary constructors from 'AttrBase'.
    using Base::Base;
    // using ValueType = APInt;
    static DecisionTreeAttr get(Type type, const int& value) {
        return Base::get(type.getContext(), type, value);
    }
};

class DecisionForestAttribute : public ::mlir::Attribute::AttrBase<DecisionForestAttribute, ::mlir::Attribute,
                                                                   detail::DecisionForestAttrStorage>
{
public:
    /// Inherit some necessary constructors from 'AttrBase'.
    using Base::Base;
    // using ValueType = APInt;
    static DecisionForestAttribute get(Type type, DecisionForest<>& value) {
        return Base::get(type.getContext(), type, value);
    }
};

}
}
#endif // _TREEATTRIBUTES_H_