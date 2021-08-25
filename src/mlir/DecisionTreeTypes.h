//===- Dialect.h - Dialect definition for the Toy IR ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR Dialect for the Toy language.
// See docs/Tutorials/Toy/Ch-2.md for more information.
//
//===----------------------------------------------------------------------===//

#ifndef _TREE_TYPES_H_
#define _TREE_TYPES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "DecisionForest.h"
#include "TreeTilingDescriptor.h"

namespace mlir {
namespace decisionforest {

class IDecisionForestTypePrintInterface {
public:
    virtual void print(mlir::DialectAsmPrinter &printer) = 0;
};

//===----------------------------------------------------------------------===//
// Decision Forest Types
//===----------------------------------------------------------------------===//


struct NumericalNodeTypeKey {
    Type thresholdType;
    Type indexType;
    bool operator==(const NumericalNodeTypeKey& that) const
    {
        return this->thresholdType == that.thresholdType && this->indexType == that.indexType;
    }
};

struct LeafNodeTypeKey {
    Type returnType;
    bool operator==(const LeafNodeTypeKey& that) const
    {
        return this->returnType == that.returnType;
    }
};

//// Defines the type of a tree node. Has the following fields
//// 1. Threshold type
//// 2. Feature index type (an int type)
//// 3. TODO : Do we need categorical or numerical here? 
//// Or the type of index needed for node indices (this seems like it should be on the tree)
struct NumericalNodeTypeStorage : public TypeStorage, IDecisionForestTypePrintInterface {
    NumericalNodeTypeStorage(Type thresholdType, Type indexType)
        : m_thresholdType(thresholdType), m_indexType(indexType) {}

    /// The hash key for this storage is a pair of the integer and type params.
    using KeyTy = NumericalNodeTypeKey;

    /// Define the comparison function for the key type.
    bool operator==(const KeyTy &key) const {
        KeyTy myKey{m_thresholdType, m_indexType};
        return key == myKey;
    }

    /// Define a hash function for the key type.
    /// Note: This isn't necessary because std::pair, unsigned, and Type all have
    /// hash functions already available.
    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_combine(key.thresholdType, key.indexType);
    }

    /// Define a construction function for the key type.
    /// Note: This isn't necessary because KeyTy can be directly constructed with
    /// the given parameters.
    static KeyTy getKey(Type thresholdType, Type indexType) {
        return KeyTy{thresholdType, indexType};
    }

    /// Define a construction method for creating a new instance of this storage.
    static NumericalNodeTypeStorage *construct(TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
    return new (allocator.allocate<NumericalNodeTypeStorage>())
        NumericalNodeTypeStorage(key.thresholdType, key.indexType);
    }

    /// The parametric data held by the storage class.
    Type m_thresholdType;
    Type m_indexType;
public:
    void print(mlir::DialectAsmPrinter &printer) override;
};

struct LeafNodeTypeStorage : public TypeStorage, IDecisionForestTypePrintInterface {
    LeafNodeTypeStorage(Type returnType)
        : m_returnType(returnType) {}

    /// The hash key for this storage is a pair of the integer and type params.
    using KeyTy = LeafNodeTypeKey;

    /// Define the comparison function for the key type.
    bool operator==(const KeyTy &key) const {
        KeyTy myKey{m_returnType};
        return key == myKey;
    }

    /// Define a hash function for the key type.
    /// Note: This isn't necessary because std::pair, unsigned, and Type all have
    /// hash functions already available.
    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_combine(key.returnType);
    }

    /// Define a construction function for the key type.
    /// Note: This isn't necessary because KeyTy can be directly constructed with
    /// the given parameters.
    static KeyTy getKey(Type returnType) {
        return KeyTy{returnType};
    }

    /// Define a construction method for creating a new instance of this storage.
    static LeafNodeTypeStorage *construct(TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
    return new (allocator.allocate<LeafNodeTypeStorage>())
        LeafNodeTypeStorage(key.returnType);
    }

    /// The parametric data held by the storage class.
    Type m_returnType;
public:
    void print(mlir::DialectAsmPrinter &printer) override;
};

// struct CategoricalNodeTypeStorage : public TypeStorage, IDecisionForestTypePrintInterface {
// };

class NodeType : public Type::TypeBase<NodeType, mlir::Type, TypeStorage>
{
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;
    virtual void print(mlir::DialectAsmPrinter &printer) { printer << "NodeType"; }   
};

class NumericalNodeType : public mlir::Type::TypeBase<NumericalNodeType, NodeType,
                                               NumericalNodeTypeStorage> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// Create an instance of a `NumericalNodeType` with the given element types. There
    /// *must* be atleast one element type.
    static NumericalNodeType get(mlir::Type thresholdType, mlir::Type indexType)
    {
        // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
        // of this type. The first parameter is the context to unique in. The
        // parameters after the context are forwarded to the storage instance.
        mlir::MLIRContext *ctx = thresholdType.getContext();
        assert (ctx == indexType.getContext());
        return Base::get(ctx, thresholdType, indexType);
    }

    mlir::Type getThresholdType() { return getImpl()->m_thresholdType; }
    mlir::Type getIndexType() { return getImpl()->m_indexType; }

    void print(mlir::DialectAsmPrinter &printer) override { getImpl()->print(printer); }
};

class LeafNodeType : public mlir::Type::TypeBase<LeafNodeType, NodeType,
                                                 LeafNodeTypeStorage> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static LeafNodeType get(mlir::Type returnType)
    {
        // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
        // of this type. The first parameter is the context to unique in. The
        // parameters after the context are forwarded to the storage instance.
        mlir::MLIRContext *ctx = returnType.getContext();
        return Base::get(ctx, returnType);
    }

    mlir::Type getReturnType() { return getImpl()->m_returnType; }

    void print(mlir::DialectAsmPrinter &printer) override { getImpl()->print(printer); }
};


// class CategoricalNodeType : public mlir::Type::TypeBase<CategoricalNodeType, NodeType,
//                                                         CategoricalNodeTypeStorage> {
// };                

// TODO We currently store the resultType here and including the "tensor type" due to the batch size.
// Should it just be the numerical type? (f64 instead of Tensor<16xf64>)
// TODO Should we store the input row type (apart from the batch size) here?
struct TreeEnsembleTypeKey {
    Type resultType;
    size_t numberOfTrees;
    Type rowType;
    ReductionType reductionType;

    bool operator==(const TreeEnsembleTypeKey& that) const
    {
        return this->resultType == that.resultType && 
               this->numberOfTrees == that.numberOfTrees && 
               this->rowType == that.rowType && 
               this->reductionType == that.reductionType;
    }
};

struct TreeTypeKey {
    Type resultType;
    TreeTilingDescriptor tilingDescriptor;
    bool operator==(const TreeTypeKey& that) const
    {
        return this->resultType==that.resultType && this->tilingDescriptor==that.tilingDescriptor;
    }
};

//// Defines the type of a tree ensemble. 
struct TreeEnsembleTypeStorage : public TypeStorage, IDecisionForestTypePrintInterface {
    TreeEnsembleTypeStorage(Type resultType, size_t numTrees, Type rowType, ReductionType reductionType)
        : m_resultType(resultType), m_numTrees(numTrees), m_rowType(rowType), m_reductionType(reductionType) {}

    /// The hash key for this storage is a pair of the integer and type params.
    using KeyTy = TreeEnsembleTypeKey;

    /// Define the comparison function for the key type.
    bool operator==(const KeyTy &key) const {
        KeyTy myKey{ m_resultType, m_numTrees, m_rowType, m_reductionType };
        return key == myKey;
    }

    /// Define a hash function for the key type.
    /// Note: This isn't necessary because std::pair, unsigned, and Type all have
    /// hash functions already available.
    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_combine(key.resultType, key.numberOfTrees, key.rowType, key.reductionType);
    }

    /// Define a construction function for the key type.
    /// Note: This isn't necessary because KeyTy can be directly constructed with
    /// the given parameters.
    static KeyTy getKey(Type resultType, size_t numTrees, Type rowType, ReductionType reductionType) {
        return KeyTy{ resultType, numTrees, rowType, reductionType};
    }

    /// Define a construction method for creating a new instance of this storage.
    static TreeEnsembleTypeStorage *construct(TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
        return new (allocator.allocate<TreeEnsembleTypeStorage>())
                    TreeEnsembleTypeStorage(key.resultType, key.numberOfTrees, key.rowType, key.reductionType);
    }

    /// The parametric data held by the storage class.
    Type m_resultType;
    size_t m_numTrees;
    Type m_rowType;
    ReductionType m_reductionType;
public:
    void print(mlir::DialectAsmPrinter &printer) override;
};

class TreeEnsembleType : public mlir::Type::TypeBase<TreeEnsembleType, mlir::Type,
                                                     TreeEnsembleTypeStorage> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static TreeEnsembleType get(Type resultType, size_t numTrees, Type rowType, ReductionType reductionType)
    {
        // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
        // of this type. The first parameter is the context to unique in. The
        // parameters after the context are forwarded to the storage instance.
        mlir::MLIRContext *ctx = resultType.getContext();
        return Base::get(ctx, resultType, numTrees, rowType, reductionType);
    }

    mlir::Type getResultType() { return getImpl()->m_resultType; }
    size_t getNumberOfTrees() { return getImpl()->m_numTrees; }
    mlir::Type getRowType() { return getImpl()->m_rowType; }
    ReductionType getReductionType() { return getImpl()->m_reductionType; }

    void print(mlir::DialectAsmPrinter &printer) { getImpl()->print(printer); }
};

struct TreeTypeStorage : public TypeStorage, IDecisionForestTypePrintInterface {
    TreeTypeStorage(Type resultType, const TreeTilingDescriptor& tilingDescriptor)
        : m_resultType(resultType), m_tilingDescriptor(tilingDescriptor) {}

    /// The hash key for this storage is a pair of the integer and type params.
    using KeyTy = TreeTypeKey;

    /// Define the comparison function for the key type.
    bool operator==(const KeyTy &key) const {
        KeyTy myKey{ m_resultType, m_tilingDescriptor };
        return key == myKey;
    }

    /// Define a hash function for the key type.
    static llvm::hash_code hashKey(const KeyTy &key) {
        std::string tilingDescStr = key.tilingDescriptor.ToHashString();
        return llvm::hash_combine(key.resultType, tilingDescStr);
    }

    /// Define a construction function for the key type.
    /// Note: This isn't necessary because KeyTy can be directly constructed with
    /// the given parameters.
    static KeyTy getKey(Type resultType) {
        return KeyTy{ resultType, TreeTilingDescriptor() };
    }

    static KeyTy getKey(Type resultType, TreeTilingDescriptor tilingDescriptor) {
        return KeyTy{ resultType, tilingDescriptor };
    }

    /// Define a construction method for creating a new instance of this storage.
    static TreeTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
        return new (allocator.allocate<TreeTypeStorage>()) TreeTypeStorage(key.resultType, key.tilingDescriptor);
    }

    /// The parametric data held by the storage class.
    Type m_resultType;
    TreeTilingDescriptor m_tilingDescriptor;
public:
    void print(mlir::DialectAsmPrinter &printer) override;
};

class TreeType : public mlir::Type::TypeBase<TreeType, mlir::Type,
                                             TreeTypeStorage> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static TreeType get(Type resultType)
    {
        // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
        // of this type. The first parameter is the context to unique in. The
        // parameters after the context are forwarded to the storage instance.
        mlir::MLIRContext *ctx = resultType.getContext();
        return Base::get(ctx, resultType);
    }

    static TreeType get(Type resultType, const TreeTilingDescriptor& tilingDescriptor)
    {
        mlir::MLIRContext *ctx = resultType.getContext();
        return Base::get(ctx, resultType, tilingDescriptor);
    }

    mlir::Type getResultType() { return getImpl()->m_resultType; }
    const TreeTilingDescriptor& getTilingDescriptor() const { return getImpl()->m_tilingDescriptor; }

    void print(mlir::DialectAsmPrinter &printer) { getImpl()->print(printer); }
};

} // end namespace decisionforest
} // end namespace mlir

#endif // _TREE_TYPES_H_
