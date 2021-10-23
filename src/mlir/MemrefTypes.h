#ifndef _MEMREFTYPES_H_
#define _MEMREFTYPES_H_
#include <algorithm>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Dialect/Vector/VectorOps.h"

namespace mlir
{
namespace decisionforest
{

struct TiledNumericalNodeTypeKey {
    Type thresholdType;
    Type indexType;
    int32_t tileSize;
    bool operator==(const TiledNumericalNodeTypeKey& that) const {
        return this->thresholdType == that.thresholdType && this->indexType == that.indexType &&
               this->tileSize == that.tileSize;
    }
};

struct TiledNumericalNodeTypeStorage : public TypeStorage {
    TiledNumericalNodeTypeStorage(Type thresholdType, Type indexType, int32_t tileSize)
        : m_thresholdType(thresholdType), m_indexType(indexType), m_tileSize(tileSize) {}

    /// The hash key for this storage is a pair of the integer and type params.
    using KeyTy = TiledNumericalNodeTypeKey;

    /// Define the comparison function for the key type.
    bool operator==(const KeyTy &key) const {
        KeyTy myKey{m_thresholdType, m_indexType, m_tileSize};
        return key == myKey;
    }

    /// Define a hash function for the key type.
    /// Note: This isn't necessary because std::pair, unsigned, and Type all have
    /// hash functions already available.
    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_combine(key.thresholdType, key.indexType, key.tileSize);
    }

    /// Define a construction function for the key type.
    /// Note: This isn't necessary because KeyTy can be directly constructed with
    /// the given parameters.
    static KeyTy getKey(Type thresholdType, Type indexType, int32_t tileSize) {
        return KeyTy{thresholdType, indexType, tileSize};
    }

    /// Define a construction method for creating a new instance of this storage.
    static TiledNumericalNodeTypeStorage *construct(TypeStorageAllocator &allocator,
                                                    const KeyTy &key) {
    return new (allocator.allocate<TiledNumericalNodeTypeStorage>())
        TiledNumericalNodeTypeStorage(key.thresholdType, key.indexType, key.tileSize);
    }

    /// The parametric data held by the storage class.
    Type m_thresholdType;
    Type m_indexType;
    int32_t m_tileSize;
public:
    void print(mlir::DialectAsmPrinter &printer) { printer << "TiledNumericalNode(" << m_thresholdType << ", " << m_indexType << ", " << m_tileSize << ")"; }
};


// An element type that represents a tree node in the memref
class TiledNumericalNodeType : public mlir::Type::TypeBase<TiledNumericalNodeType, mlir::Type,
                                                           TiledNumericalNodeTypeStorage, DataLayoutTypeInterface::Trait, MemRefElementTypeInterface::Trait> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// Create an instance of a `NumericalNodeType` with the given element types. There
    /// *must* be atleast one element type.
    static TiledNumericalNodeType get(mlir::Type thresholdType, mlir::Type indexType, int32_t tileSize)
    {
        // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
        // of this type. The first parameter is the context to unique in. The
        // parameters after the context are forwarded to the storage instance.
        mlir::MLIRContext *ctx = thresholdType.getContext();
        assert (ctx == indexType.getContext());
        return Base::get(ctx, thresholdType, indexType, tileSize);
    }

    mlir::Type getThresholdElementType() const { return getImpl()->m_thresholdType; }
    mlir::Type getIndexElementType() const { return getImpl()->m_indexType; }
    int32_t getTileSize() const { return getImpl()->m_tileSize; }
    
    mlir::Type getThresholdFieldType() const { 
        if (getTileSize() == 1)
            return getThresholdElementType();
        else
            return mlir::VectorType::get({ getTileSize() }, getThresholdElementType());
    }

    mlir::Type getIndexFieldType() const { 
        if (getTileSize() == 1)
            return getIndexElementType();
        else
            return mlir::VectorType::get({ getTileSize() }, getIndexElementType());
    }

    void print(mlir::DialectAsmPrinter &printer) { getImpl()->print(printer); }

    // InterfaceMethod<
    //   /*description=*/"Returns the size of this type in bits.",
    //   /*retTy=*/"unsigned",
    //   /*methodName=*/"getTypeSizeInBits",
    //   /*args=*/(ins "const ::mlir::DataLayout &":$dataLayout,
    //                 "::mlir::DataLayoutEntryListRef":$params)
    // >,
    unsigned getTypeSizeInBits(const DataLayout &layout,
                               DataLayoutEntryListRef params) const {
        // TODO We need to take care of padding here so the alignment for whatever we store second is satisfied!
        unsigned thresholdSize = layout.getTypeSizeInBits(getThresholdElementType());
        unsigned indexSize = layout.getTypeSizeInBits(getIndexElementType());
        return (thresholdSize+indexSize) * getTileSize();
    }
    // InterfaceMethod<
    //   /*description=*/"Returns the ABI-required alignment for this type, "
    //                   "in bytes",
    //   /*retTy=*/"unsigned",
    //   /*methodName=*/"getABIAlignment",
    //   /*args=*/(ins "const ::mlir::DataLayout &":$dataLayout,
    //                 "::mlir::DataLayoutEntryListRef":$params)
    // >,
    unsigned getABIAlignment(const DataLayout &layout,
                            DataLayoutEntryListRef params) const {
        unsigned thresholdAlignment = layout.getTypeABIAlignment(getThresholdElementType());
        unsigned indexAlignment = layout.getTypeABIAlignment(getIndexElementType());
        return std::max(thresholdAlignment, indexAlignment);
    }
    // InterfaceMethod<
    //   /*description=*/"Returns the preferred alignemnt for this type, "
    //                   "in bytes.",
    //   /*retTy=*/"unsigned",
    //   /*methodName=*/"getPreferredAlignment",
    //   /*args=*/(ins "const ::mlir::DataLayout &":$dataLayout,
    //                 "::mlir::DataLayoutEntryListRef":$params)
    // >
    unsigned getPreferredAlignment(const DataLayout &layout,
                            DataLayoutEntryListRef params) const {
        unsigned thresholdAlignment = layout.getTypePreferredAlignment(getThresholdElementType());
        unsigned indexAlignment = layout.getTypePreferredAlignment(getIndexElementType());
        return std::max(thresholdAlignment, indexAlignment);
    }
};

} // decisionforest
} // mlir

#endif // _MEMREFTYPES_H_