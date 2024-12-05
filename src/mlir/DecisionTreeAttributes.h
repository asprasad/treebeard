#ifndef _TREEATTRIBUTES_H_
#define _TREEATTRIBUTES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "DecisionForest.h"
#include "DecisionTreeTypes.h"

namespace mlir 
{
namespace decisionforest
{
namespace detail
{
struct DecisionTreeAttrStorage : public ::mlir::AttributeStorage
{
    DecisionTreeAttrStorage(::mlir::Type type, const DecisionForest& forest, int64_t index)
      : m_type(type), m_forest(forest), m_index(index) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<::mlir::Type, DecisionForest, int64_t>;

    bool operator==(const KeyTy &tblgenKey) const {
        if (!(m_type == std::get<0>(tblgenKey)))
            return false;
        if (!(m_forest == std::get<1>(tblgenKey)))
            return false;
        if (!(m_index == std::get<2>(tblgenKey)))
            return false;
        return true;
    }
    
    static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
        auto& forest = std::get<1>(tblgenKey);
        auto index = std::get<2>(tblgenKey);
        return ::llvm::hash_combine(std::get<0>(tblgenKey), forest.Serialize(), std::to_string(index));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static DecisionTreeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &tblgenKey) {
        auto type = std::get<0>(tblgenKey);
        auto& forest = std::get<1>(tblgenKey);
        auto index = std::get<2>(tblgenKey);
        return new (allocator.allocate<DecisionTreeAttrStorage>())
                    DecisionTreeAttrStorage(type, forest, index);
    }
    ::mlir::Type m_type;
    DecisionForest m_forest;
    int64_t m_index;
};

// TODO How do we use templatization of DecisionForest here?
struct DecisionForestAttrStorage : public ::mlir::AttributeStorage
{
    DecisionForestAttrStorage(::mlir::Type type, const DecisionForest& forest)
      : m_type(type), m_forest(forest) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<::mlir::Type, DecisionForest>;

    bool operator==(const KeyTy &tblgenKey) const {
        if (!(m_type == std::get<0>(tblgenKey)))
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
    ::mlir::Type m_type;
    DecisionForest m_forest;
};

} // namespace detail

class DecisionTreeAttribute : public ::mlir::Attribute::AttrBase<DecisionTreeAttribute, ::mlir::Attribute,
                                         detail::DecisionTreeAttrStorage>
{
public:
    /// Inherit some necessary constructors from 'AttrBase'.
    using Base::Base;
    static llvm::StringRef getName() {
        return "DecisionTreeAttribute";
    }

    // Static member for 'name', if the static method approach is not used
    static constexpr const char *name = "DecisionTreeAttribute";
    // using ValueType = APInt;
    static DecisionTreeAttribute get(Type type,  DecisionForest& forest, int64_t index) {
        return Base::get(type.getContext(), type, forest, index);
    }
    std::string Serialize() {
        int64_t index = getImpl()->m_index;
        return getImpl()->m_forest.GetTree(index).Serialize();
    }
    void Print(::mlir::DialectAsmPrinter &os) {
        int64_t index = getImpl()->m_index;
        auto& tree = getImpl()->m_forest.GetTree(index);
        std::string treeStr = tree.PrintToString();
        os << "Tree = ( " << treeStr << ") treeType = (" << getImpl()->m_type << ")";
    }

};

class DecisionForestAttribute : public ::mlir::Attribute::AttrBase<DecisionForestAttribute, ::mlir::Attribute,
                                                                   detail::DecisionForestAttrStorage>
{
public:
    /// Inherit some necessary constructors from 'AttrBase'.
    using Base::Base;
    static llvm::StringRef getName() {
        return "DecisionForestAttribute";
    }

    // Static member for 'name', if the static method approach is not used
    static constexpr const char *name = "DecisionForestAttribute";
    static DecisionForestAttribute get(Type type, DecisionForest& value) {
        return Base::get(type.getContext(), type, value);
    }
    std::string Serialize() {
        return getImpl()->m_forest.Serialize();
    }
    void Print(::mlir::DialectAsmPrinter &os) {
        std::string forestStr = getImpl()->m_forest.PrintToString();
        auto ensembleMLIRType = getImpl()->m_type;
        mlir::decisionforest::TreeEnsembleType ensembleType = ensembleMLIRType.cast<mlir::decisionforest::TreeEnsembleType>();
        os << "Forest = ( " << forestStr << " ) forestType = (" << ensembleType << ")";
    }
    DecisionForest& GetDecisionForest() {
        return getImpl()->m_forest;
    }
    mlir::Type getType() const {
        return getImpl()->m_type;
     }
};

}
}
#endif // _TREEATTRIBUTES_H_