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

#include "llvm/ADT/STLExtras.h"

using json = nlohmann::json;

/*
//++
// The abstract base class from which all JSON model parsers derive.
// Also implements callbacks that construct the actual MLIR structures.
//--
*/
namespace TreeHeavy
{
enum FeatureType { kNumerical, kCategorical };

template<typename T>
mlir::Type GetMLIRFloatType(const T& val, mlir::OpBuilder& builder)
{
    assert (false);
    return mlir::Type();
}

template<>
mlir::Type GetMLIRFloatType(const float& val, mlir::OpBuilder& builder)
{
    return builder.getF32Type();
}

template<>
mlir::Type GetMLIRFloatType(const double& val, mlir::OpBuilder& builder)
{
    return builder.getF64Type();
}

mlir::Type GetMLIRTypeFromString(const std::string& typestr, mlir::OpBuilder& builder)
{
    if (typestr == "float")
        return builder.getF64Type();
    else
        assert(false);
    return mlir::Type();
}

template<typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
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

    mlir::Type GetFunctionArgumentType()
    {
        const auto& features = m_forest->GetFeatures();
        mlir::Type elementType = GetMLIRTypeFromString(features.front().type, m_builder);
        int64_t shape[] = { m_batchSize, static_cast<int64_t>(features.size())};
        return mlir::RankedTensorType::get(shape, elementType);
    }
    mlir::Type GetFunctionReturnType()
    {
        return mlir::RankedTensorType::get(m_batchSize, GetMLIRFloatType(ReturnType(), m_builder));
    }
    mlir::FunctionType GetFunctionType()
    {
        auto argType = GetFunctionArgumentType();
        auto returnType = GetFunctionReturnType();
        return m_builder.getFunctionType(argType, returnType);
    }
    mlir::FuncOp GetFunctionPrototype()
    {
        auto location = m_builder.getUnknownLoc();
        auto functionType = GetFunctionType();
        // TODO the function name needs to be an input or derived from the input
        // return mlir::FuncOp::create(location, std::string("Prediction_Function"), functionType);
        return m_builder.create<mlir::FuncOp>(location, std::string("Prediction_Function"), functionType);
    }
public:
    ModelJSONParser(mlir::MLIRContext& context, int32_t batchSize)
        : m_forest(new DecisionForestType), m_currentTree(nullptr), m_context(context), m_builder(&context), m_batchSize(batchSize)
    {
        m_module = mlir::ModuleOp::create(m_builder.getUnknownLoc(), llvm::StringRef("MyModule"));
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

        auto forestType = mlir::decisionforest::TreeEnsembleType::get(GetFunctionReturnType(),
                                                                      m_forest->NumTrees(), GetFunctionArgumentType(), mlir::decisionforest::kAdd);
        auto forestAttribute = mlir::decisionforest::DecisionForestAttribute::get(forestType, *m_forest);

        // ::mlir::Type resultType0, ::mlir::decisionforest::DecisionForestAttribute ensemble, ::mlir::Value data)
        auto predictOp = m_builder.create<mlir::decisionforest::PredictForestOp>(m_builder.getUnknownLoc(), GetFunctionReturnType(),
                                                                                 forestAttribute, entryBlock.getArguments()[0]);
        m_builder.create<mlir::decisionforest::ReturnOp>(m_builder.getUnknownLoc(), predictOp);
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