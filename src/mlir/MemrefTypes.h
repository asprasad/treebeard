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
    Type tileShapeType;
    int32_t tileSize;
    bool sparseRepresentation;
    Type childIndexType;
    bool operator==(const TiledNumericalNodeTypeKey& that) const {
        return this->thresholdType == that.thresholdType && this->indexType == that.indexType &&
               this->tileShapeType == that.tileShapeType && this->tileSize == that.tileSize && 
               this->sparseRepresentation == that.sparseRepresentation && 
                // If the representation is not sparse, don't compare child index type
               (this->sparseRepresentation ? this->childIndexType==that.childIndexType : true);
    }
};

struct TiledNumericalNodeTypeStorage : public TypeStorage {
    TiledNumericalNodeTypeStorage(Type thresholdType, Type indexType, Type tileShapeType, 
                                  int32_t tileSize, bool sparseRepresentation, Type childIndexType)
        : m_thresholdType(thresholdType), m_indexType(indexType), m_tileShapeType(tileShapeType), 
          m_tileSize(tileSize), m_sparseRepresentation(sparseRepresentation), m_childIndexType(childIndexType) {}

    /// The hash key for this storage is a pair of the integer and type params.
    using KeyTy = TiledNumericalNodeTypeKey;

    /// Define the comparison function for the key type.
    bool operator==(const KeyTy &key) const {
        KeyTy myKey{m_thresholdType, m_indexType, m_tileShapeType, m_tileSize, m_sparseRepresentation, m_childIndexType};
        return key == myKey;
    }

    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_combine(key.thresholdType, key.indexType, key.tileShapeType, key.tileSize, key.sparseRepresentation, key.childIndexType);
    }

    static KeyTy getKey(Type thresholdType, Type indexType, int32_t tileSize) {
        auto i32Type = IntegerType::get(thresholdType.getContext(), 32);
        return KeyTy{thresholdType, indexType, i32Type, tileSize, false, i32Type};
    }

    static KeyTy getKey(Type thresholdType, Type indexType, Type tileShapeType, int32_t tileSize) {
        auto i32Type = IntegerType::get(thresholdType.getContext(), 32);
        return KeyTy{thresholdType, indexType, tileShapeType, tileSize, false, i32Type};
    }

    static KeyTy getKey(Type thresholdType, Type indexType, Type tileShapeType, int32_t tileSize, bool sparseRep, Type childIndexType) {
        return KeyTy{thresholdType, indexType, tileShapeType, tileSize, sparseRep, childIndexType};
    }

    /// Define a construction method for creating a new instance of this storage.
    static TiledNumericalNodeTypeStorage *construct(TypeStorageAllocator &allocator,
                                                    const KeyTy &key) {
    return new (allocator.allocate<TiledNumericalNodeTypeStorage>())
        TiledNumericalNodeTypeStorage(key.thresholdType, key.indexType, key.tileShapeType, key.tileSize, key.sparseRepresentation, key.childIndexType);
    }

    /// The parametric data held by the storage class.
    Type m_thresholdType;
    Type m_indexType;
    Type m_tileShapeType;
    int32_t m_tileSize;
    bool m_sparseRepresentation;
    Type m_childIndexType;
public:
    void print(mlir::DialectAsmPrinter &printer) { 
        printer << "TiledNumericalNode(" << m_thresholdType << ", " << m_indexType << ", " 
                << m_tileShapeType << ", " << m_tileSize << ", " << m_sparseRepresentation 
                << ", " << m_childIndexType << ")";
    }
};


// An element type that represents a tree node in the memref
class TiledNumericalNodeType : public mlir::Type::TypeBase<TiledNumericalNodeType, mlir::Type,
                                                           TiledNumericalNodeTypeStorage, DataLayoutTypeInterface::Trait, MemRefElementTypeInterface::Trait> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// Create an instance of a `NumericalNodeType` with the given element types. There
    /// *must* be atleast one element type.
    static TiledNumericalNodeType get(mlir::Type thresholdType, mlir::Type indexType, int32_t tileSize) {
        // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
        // of this type. The first parameter is the context to unique in. The
        // parameters after the context are forwarded to the storage instance.
        mlir::MLIRContext *ctx = thresholdType.getContext();
        assert (ctx == indexType.getContext());
        return Base::get(ctx, thresholdType, indexType, tileSize);
    }

    static TiledNumericalNodeType get(mlir::Type thresholdType, mlir::Type indexType, mlir::Type tileShapeType, int32_t tileSize) {
        mlir::MLIRContext *ctx = thresholdType.getContext();
        assert (ctx == indexType.getContext());
        return Base::get(ctx, thresholdType, indexType, tileShapeType, tileSize);
    }

    static TiledNumericalNodeType get(mlir::Type thresholdType, mlir::Type indexType, mlir::Type tileShapeType, 
                                      int32_t tileSize, bool sparseRep, mlir::Type childIndexType) {
        mlir::MLIRContext *ctx = thresholdType.getContext();
        assert (ctx == indexType.getContext());
        return Base::get(ctx, thresholdType, indexType, tileShapeType, tileSize, sparseRep, childIndexType);
    }

    mlir::Type getThresholdElementType() const { return getImpl()->m_thresholdType; }
    mlir::Type getIndexElementType() const { return getImpl()->m_indexType; }
    mlir::Type getTileShapeType() const { return getImpl()->m_tileShapeType; }
    int32_t getTileSize() const { return getImpl()->m_tileSize; }
    bool isSparseRepresentation() const { return getImpl()->m_sparseRepresentation; }
    mlir::Type getChildIndexType() const { return getImpl()->m_childIndexType; }
    
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