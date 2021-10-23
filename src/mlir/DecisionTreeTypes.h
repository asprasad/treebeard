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

//===----------------------------------------------------------------------===//
// Node Types
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

    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_combine(key.thresholdType, key.indexType);
    }

    static KeyTy getKey(Type thresholdType, Type indexType) {
        return KeyTy{thresholdType, indexType};
    }

    /// Define a construction method for creating a new instance of this storage.
    static NumericalNodeTypeStorage *construct(TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
    return new (allocator.allocate<NumericalNodeTypeStorage>())
        NumericalNodeTypeStorage(key.thresholdType, key.indexType);
    }

    Type m_thresholdType;
    Type m_indexType;
public:
    void print(mlir::DialectAsmPrinter &printer) override;
};

struct LeafNodeTypeStorage : public TypeStorage, IDecisionForestTypePrintInterface {
    LeafNodeTypeStorage(Type returnType)
        : m_returnType(returnType) {}

    using KeyTy = LeafNodeTypeKey;

    bool operator==(const KeyTy &key) const {
        KeyTy myKey{m_returnType};
        return key == myKey;
    }

    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_combine(key.returnType);
    }

    static KeyTy getKey(Type returnType) {
        return KeyTy{returnType};
    }

    static LeafNodeTypeStorage *construct(TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
    return new (allocator.allocate<LeafNodeTypeStorage>())
        LeafNodeTypeStorage(key.returnType);
    }

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
    using Base::Base;

    static NumericalNodeType get(mlir::Type thresholdType, mlir::Type indexType) {
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
    using Base::Base;

    static LeafNodeType get(mlir::Type returnType) {
        mlir::MLIRContext *ctx = returnType.getContext();
        return Base::get(ctx, returnType);
    }

    mlir::Type getReturnType() { return getImpl()->m_returnType; }

    void print(mlir::DialectAsmPrinter &printer) override { getImpl()->print(printer); }
};


// class CategoricalNodeType : public mlir::Type::TypeBase<CategoricalNodeType, NodeType,
//                                                         CategoricalNodeTypeStorage> {
// };                

//===----------------------------------------------------------------------===//
// Tree Type
//===----------------------------------------------------------------------===//
struct TreeTypeKey {
    Type resultType;
    TreeTilingDescriptor tilingDescriptor;
    Type thresholdType;
    Type featureIndexType;
    bool operator==(const TreeTypeKey& that) const
    {
        return this->resultType==that.resultType && this->tilingDescriptor==that.tilingDescriptor &&
               this->thresholdType==that.thresholdType && this->featureIndexType==that.featureIndexType;
    }
};

struct TreeTypeStorage : public TypeStorage, IDecisionForestTypePrintInterface {
    TreeTypeStorage(Type resultType, const TreeTilingDescriptor& tilingDescriptor, Type thresholdType, Type featureIndexType)
        : m_resultType(resultType), m_tilingDescriptor(tilingDescriptor), m_thresholdType(thresholdType), m_featureIndexType(featureIndexType) {}

    using KeyTy = TreeTypeKey;

    bool operator==(const KeyTy &key) const {
        KeyTy myKey{ m_resultType, m_tilingDescriptor, m_thresholdType, m_featureIndexType };
        return key == myKey;
    }

    static llvm::hash_code hashKey(const KeyTy &key) {
        std::string tilingDescStr = key.tilingDescriptor.ToHashString();
        return llvm::hash_combine(key.resultType, tilingDescStr, key.thresholdType, key.featureIndexType);
    }

    static KeyTy getKey(Type resultType) {
        auto context = resultType.getContext();
        return KeyTy{ resultType, TreeTilingDescriptor(), FloatType::getF64(context), IntegerType::get(context, 32) };
    }

    static KeyTy getKey(Type resultType, TreeTilingDescriptor tilingDescriptor) {
        auto context = resultType.getContext();
        return KeyTy{ resultType, tilingDescriptor, FloatType::getF64(context), IntegerType::get(context, 32) };
    }

    static KeyTy getKey(Type resultType, TreeTilingDescriptor tilingDescriptor, Type thresholdType, Type featureIndexType) {
        return KeyTy{ resultType, tilingDescriptor, thresholdType, featureIndexType };
    }

    static TreeTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
        return new (allocator.allocate<TreeTypeStorage>()) TreeTypeStorage(key.resultType, key.tilingDescriptor, key.thresholdType, key.featureIndexType);
    }

    Type m_resultType;
    TreeTilingDescriptor m_tilingDescriptor;
    Type m_thresholdType;
    Type m_featureIndexType;
public:
    void print(mlir::DialectAsmPrinter &printer) override;
};

class TreeType : public mlir::Type::TypeBase<TreeType, mlir::Type,
                                             TreeTypeStorage> {
public:
    using Base::Base;

    static TreeType get(Type resultType, const TreeTilingDescriptor& tilingDescriptor, Type thresholdType, Type featureIndexType) {
        mlir::MLIRContext *ctx = resultType.getContext();
        return Base::get(ctx, resultType, tilingDescriptor, thresholdType, featureIndexType);
    }

    mlir::Type getResultType() const { return getImpl()->m_resultType; }
    const TreeTilingDescriptor& getTilingDescriptor() const { return getImpl()->m_tilingDescriptor; }
    mlir::Type getThresholdType() const { return getImpl()->m_thresholdType; }
    mlir::Type getFeatureIndexType() const { return getImpl()->m_featureIndexType; }

    void print(mlir::DialectAsmPrinter &printer) { getImpl()->print(printer); }
};

//===----------------------------------------------------------------------===//
// Tree Ensemble Type
//===----------------------------------------------------------------------===//

// TODO We currently store the resultType here and including the "tensor type" due to the batch size.
// Should it just be the numerical type? (f64 instead of Tensor<16xf64>)
// TODO Should we store the input row type (apart from the batch size) here?
struct TreeEnsembleTypeKey {
    Type resultType;
    size_t numberOfTrees;
    Type rowType;
    ReductionType reductionType;
    bool sameTypeTrees;
    Type treeType;
    std::vector<Type> treeTypes;

    bool operator==(const TreeEnsembleTypeKey& that) const
    {
        // This maybe a problem if we don't enforce that the vector can't have the
        // same type repeated
        if (this->sameTypeTrees != that.sameTypeTrees)
            return false;
        
        bool treesHaveSameType = sameTypeTrees ? treeType==that.treeType : treeTypes==that.treeTypes;
        if (!treesHaveSameType)
            return false;
         
        return this->resultType == that.resultType && 
               this->numberOfTrees == that.numberOfTrees && 
               this->rowType == that.rowType && 
               this->reductionType == that.reductionType;
    }
};

struct TreeEnsembleTypeStorage : public TypeStorage, IDecisionForestTypePrintInterface {
    TreeEnsembleTypeStorage(Type resultType, size_t numTrees, Type rowType, ReductionType reductionType, 
                            bool treesHaveSameType, Type treeType, const std::vector<Type>& treeTypes)
        : m_resultType(resultType), m_numTrees(numTrees), m_rowType(rowType), m_reductionType(reductionType),
          m_treesHaveSameType(treesHaveSameType), m_treeType(treeType), m_treeTypes(treeTypes) 
         {}

    using KeyTy = TreeEnsembleTypeKey;

    bool operator==(const KeyTy &key) const {
        KeyTy myKey{ m_resultType, m_numTrees, m_rowType, m_reductionType, m_treesHaveSameType, m_treeType, m_treeTypes };
        return key == myKey;
    }

    static llvm::hash_code hashKey(const KeyTy &key) {
        std::vector<Type> treeTypes = key.sameTypeTrees ? std::vector<Type>(key.numberOfTrees, key.treeType) :
                                                              key.treeTypes;
        return llvm::hash_combine(key.resultType, key.numberOfTrees, key.rowType, key.reductionType, treeTypes);
    }

    static KeyTy getKey(Type resultType, size_t numTrees, Type rowType, ReductionType reductionType,
                        Type treeType) {
        return KeyTy{ resultType, numTrees, rowType, reductionType, true, treeType, std::vector<Type>()};
    }

    static KeyTy getKey(Type resultType, size_t numTrees, Type rowType, ReductionType reductionType,
                        const std::vector<Type>& treeTypes) {
        return KeyTy{ resultType, numTrees, rowType, reductionType, false, Type(), treeTypes};
    }

    static TreeEnsembleTypeStorage *construct(TypeStorageAllocator &allocator,
                                              const KeyTy &key) {
        return new (allocator.allocate<TreeEnsembleTypeStorage>())
                    TreeEnsembleTypeStorage(key.resultType, key.numberOfTrees, key.rowType, key.reductionType,
                                            key.sameTypeTrees, key.treeType, key.treeTypes);
    }

    Type m_resultType;
    size_t m_numTrees;
    Type m_rowType;
    ReductionType m_reductionType;
    bool m_treesHaveSameType;
    Type m_treeType;
    std::vector<Type> m_treeTypes;
public:
    void print(mlir::DialectAsmPrinter &printer) override;
};

class TreeEnsembleType : public mlir::Type::TypeBase<TreeEnsembleType, mlir::Type,
                                                     TreeEnsembleTypeStorage> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static TreeEnsembleType get(Type resultType, size_t numTrees, Type rowType, ReductionType reductionType, Type treeType) {
        mlir::MLIRContext *ctx = resultType.getContext();
        return Base::get(ctx, resultType, numTrees, rowType, reductionType, treeType);
    }
    static TreeEnsembleType get(Type resultType, size_t numTrees, Type rowType, ReductionType reductionType, 
                                const std::vector<Type>& treeTypes) {
        mlir::MLIRContext *ctx = resultType.getContext();
        return Base::get(ctx, resultType, numTrees, rowType, reductionType, treeTypes);
    }

    mlir::Type getResultType() const { return getImpl()->m_resultType; }
    size_t getNumberOfTrees() const { return getImpl()->m_numTrees; }
    mlir::Type getRowType() const { return getImpl()->m_rowType; }
    ReductionType getReductionType() const { return getImpl()->m_reductionType; }
    Type getTreeType(int32_t treeIndex) const { 
        assert (static_cast<size_t>(treeIndex) < getNumberOfTrees());
        if (getImpl()->m_treesHaveSameType)
            return getImpl()->m_treeType;
        else
            return getImpl()->m_treeTypes[treeIndex];
    }
    bool doAllTreesHaveSameType() const { return getImpl()->m_treesHaveSameType; }
    bool doAllTreesHaveSameTileSize() const {
        if (doAllTreesHaveSameType())
            return true;
        auto firstTreeTileSize = getImpl()->m_treeTypes.at(0).cast<TreeType>().getTilingDescriptor().MaxTileSize();
        for (size_t i=1; i<getNumberOfTrees() ; ++i) {
            auto treeTileSize = getImpl()->m_treeTypes.at(i).cast<TreeType>().getTilingDescriptor().MaxTileSize();
            if (firstTreeTileSize != treeTileSize)
                return false;
        }
        return true;
    }
    void print(mlir::DialectAsmPrinter &printer) { getImpl()->print(printer); }
};

} // end namespace decisionforest
} // end namespace mlir

#endif // _TREE_TYPES_H_
