#ifndef _MODEL_JSON_PARSER_H_
#define _MODEL_JSON_PARSER_H_

#include <string>
#include <vector>
#include "json.hpp"
#include "DecisionForest.h"
#include "../mlir/Dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/STLExtras.h"

using json = nlohmann::json;

/*
//++
// The abstract base class from which all JSON model parsers derive.
// Also implements callbacks that construct the actual MLIR structures.
//--
*/
namespace TreeBeard
{

template<typename T>
mlir::Type GetMLIRType(const T& val, mlir::OpBuilder& builder) {
    assert (false);
    return mlir::Type();
}

template<>
inline mlir::Type GetMLIRType(const int8_t& val, mlir::OpBuilder& builder) {
    return builder.getIntegerType(8);
}

template<>
inline mlir::Type GetMLIRType(const int16_t& val, mlir::OpBuilder& builder) {
    return builder.getIntegerType(16);
}

template<>
inline mlir::Type GetMLIRType(const int32_t& val, mlir::OpBuilder& builder) {
    return builder.getI32Type();
}

template<>
inline mlir::Type GetMLIRType(const float& val, mlir::OpBuilder& builder) {
    return builder.getF32Type();
}

template<>
inline mlir::Type GetMLIRType(const double& val, mlir::OpBuilder& builder) {
    return builder.getF64Type();
}

inline mlir::Type GetMLIRTypeFromString(const std::string& typestr, mlir::OpBuilder& builder)
{
    if (typestr == "float")
        return builder.getF64Type();
    else
        assert(false);
    return mlir::Type();
}

template<typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType, typename InputElementType>
class ModelJSONParser
{
protected:
    using DecisionForestType = mlir::decisionforest::DecisionForest<>; //<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType>;
    using DecisionTreeType = typename DecisionForestType::DecisionTreeType;

    DecisionForestType *m_forest;
    DecisionTreeType *m_currentTree;
    mlir::MLIRContext& m_context;
    mlir::ModuleOp m_module;
    mlir::OpBuilder m_builder;
    int32_t m_batchSize;

    void SetReductionType(mlir::decisionforest::ReductionType reductionType) { m_forest->SetReductionType(reductionType); }
    void AddFeature(const std::string& featureName, const std::string& type) { m_forest->AddFeature(featureName, type); }
    void NewTree() { m_currentTree = &(m_forest->NewTree()); }
    void EndTree() { m_currentTree = nullptr; }
    void SetTreeNumberOfFeatures(size_t numFeatures) { m_currentTree->SetNumberOfFeatures(numFeatures); }
    void SetTreeScalingFactor(ThresholdType scale) { m_currentTree->SetTreeScalingFactor(scale); }

    // Create a new node in the current tree
    NodeIndexType NewNode(ThresholdType threshold, FeatureIndexType featureIndex) { return m_currentTree->NewNode(threshold, featureIndex); }
    // Set the parent of a node
    void SetNodeParent(NodeIndexType node, NodeIndexType parent) { m_currentTree->SetNodeParent(node, parent); }
    // Set right child of a node
    void SetNodeRightChild(NodeIndexType node, NodeIndexType child) { m_currentTree->SetNodeRightChild(node, child); }
    // Set left child of a node
    void SetNodeLeftChild(NodeIndexType node, NodeIndexType child) { m_currentTree->SetNodeLeftChild(node, child); }
    mlir::Type GetInputRowType()
    {
        const auto& features = m_forest->GetFeatures();
        mlir::Type elementType = GetMLIRType(InputElementType(), m_builder); // GetMLIRTypeFromString(features.front().type, m_builder);
        int64_t shape[] = { static_cast<int64_t>(features.size()) };
        return mlir::MemRefType::get(shape, elementType);
    }
    mlir::Type GetFunctionArgumentType()
    {
        const auto& features = m_forest->GetFeatures();
        mlir::Type elementType = GetMLIRType(InputElementType(), m_builder); //GetMLIRTypeFromString(features.front().type, m_builder);
        int64_t shape[] = { m_batchSize, static_cast<int64_t>(features.size())};
        // auto affineMap = mlir::makeStridedLinearLayoutMap(mlir::ArrayRef<int64_t>({ static_cast<int64_t>(features.size()), 1 }), 0, elementType.getContext());
        // return mlir::MemRefType::get(shape, elementType, affineMap);
        return mlir::MemRefType::get(shape, elementType);
    }
    mlir::Type GetFunctionResultType()
    {
        return mlir::MemRefType::get(m_batchSize, GetMLIRType(ReturnType(), m_builder));
    }
    mlir::FunctionType GetFunctionType()
    {
        auto argType = GetFunctionArgumentType();
        auto resultType = GetFunctionResultType();
        return m_builder.getFunctionType({argType, resultType}, resultType);
    }
    mlir::FuncOp GetFunctionPrototype()
    {
        auto location = m_builder.getUnknownLoc();
        auto functionType = GetFunctionType();
        // TODO the function name needs to be an input or derived from the input
        // return mlir::FuncOp::create(location, std::string("Prediction_Function"), functionType);
        return m_builder.create<mlir::FuncOp>(location, std::string("Prediction_Function"), functionType, m_builder.getStringAttr("public"));
    }
    virtual mlir::decisionforest::TreeEnsembleType GetEnsembleType()
    {
        // All trees have the default tiling to start with.
        int32_t tileSize = 1;
        auto treeType = mlir::decisionforest::TreeType::get(GetMLIRType(ReturnType(), m_builder), tileSize, 
                                                            GetMLIRType(ThresholdType(), m_builder), 
                                                            GetMLIRType(FeatureIndexType(), m_builder));

        auto forestType = mlir::decisionforest::TreeEnsembleType::get(GetMLIRType(ReturnType(), m_builder),
                                                                      m_forest->NumTrees(), GetInputRowType(), 
                                                                      mlir::decisionforest::ReductionType::kAdd, treeType);
        return forestType;
    }
public:
    ModelJSONParser(mlir::MLIRContext& context, int32_t batchSize, double_t initialValue)
        : m_forest(new DecisionForestType(initialValue)), m_currentTree(nullptr), m_context(context), m_builder(&context), m_batchSize(batchSize)
    {
        m_module = mlir::ModuleOp::create(m_builder.getUnknownLoc(), llvm::StringRef("MyModule"));
    }

    ModelJSONParser(mlir::MLIRContext& context, int32_t batchSize)
        : ModelJSONParser(context, batchSize, 0.0)
    {
    }
    
    virtual void Parse() = 0;

    // Get the forest pointer
    DecisionForestType* GetForest() { return m_forest; }

    mlir::ModuleOp GetEvaluationFunction() 
    { 
        mlir::FuncOp function(GetFunctionPrototype());
        if (!function)
            return nullptr;

        // Function body. MLIR's entry block has the same arg list as function
        auto &entryBlock = *function.addEntryBlock();

        m_builder.setInsertionPointToStart(&entryBlock);
        auto zeroIndexAttr = m_builder.getIndexAttr(0);
        auto oneIndexAttr = m_builder.getIndexAttr(1);
        auto rowSizeAttr = m_builder.getIndexAttr(m_forest->GetFeatures().size());
        auto batchSizeAttr = m_builder.getIndexAttr(m_batchSize);

        auto subviewOfArg = m_builder.create<mlir::memref::SubViewOp>(m_builder.getUnknownLoc(), static_cast<mlir::Value>(entryBlock.getArguments()[0]), 
                                                                mlir::ArrayRef<mlir::OpFoldResult>({zeroIndexAttr, zeroIndexAttr}),
                                                                mlir::ArrayRef<mlir::OpFoldResult>({batchSizeAttr, rowSizeAttr}), 
                                                                mlir::ArrayRef<mlir::OpFoldResult>({oneIndexAttr, oneIndexAttr}));
        auto forestType = GetEnsembleType();
        auto forestAttribute = mlir::decisionforest::DecisionForestAttribute::get(forestType, *m_forest);

        auto predictOp = m_builder.create<mlir::decisionforest::PredictForestOp>(
            m_builder.getUnknownLoc(),GetFunctionResultType(),
            forestAttribute,
            subviewOfArg,
            entryBlock.getArguments()[1]);

        m_builder.create<mlir::ReturnOp>(m_builder.getUnknownLoc(), static_cast<mlir::Value>(predictOp));
        if (failed(mlir::verify(m_module))) {
            m_module.emitError("Module verification error");
            return nullptr;
        }
        m_module.push_back(function);

        return m_module;
    }
};
}

#endif //_MODEL_JSON_PARSER_H_