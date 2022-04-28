#ifndef _MODEL_JSON_PARSER_H_
#define _MODEL_JSON_PARSER_H_

#include <string>
#include <vector>
#include "json.hpp"
#include "DecisionForest.h"
#include "TreeTilingUtils.h"
#include "Dialect.h"
#include "StatsUtils.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

    std::string m_jsonFilePath;
    std::string m_modelGlobalsJSONFilePath;
    std::string m_statsProfileCSV;
    DecisionForestType *m_forest;
    DecisionTreeType *m_currentTree;
    mlir::decisionforest::Schedule *m_schedule;
    mlir::MLIRContext& m_context;
    mlir::ModuleOp m_module;
    mlir::OpBuilder m_builder;
    int32_t m_batchSize;
    int32_t m_childIndexBitWidth;

    void SetReductionType(mlir::decisionforest::ReductionType reductionType) { m_forest->SetReductionType(reductionType); }
    void AddFeature(const std::string& featureName, const std::string& type) { m_forest->AddFeature(featureName, type); }
    void NewTree() { m_currentTree = &(m_forest->NewTree()); }
    void EndTree() { m_currentTree = nullptr; }
    void SetTreeNumberOfFeatures(size_t numFeatures) { m_currentTree->SetNumberOfFeatures(numFeatures); }
    void SetTreeScalingFactor(ThresholdType scale) { m_currentTree->SetTreeScalingFactor(scale); }
    void SetTreeClassId(int32_t classId) { 
        if (m_forest->IsMultiClassClassifier())
            assert(classId < m_forest->GetNumClasses() && "ClassId should be lesser than number of classes for multi-class classifiers.");
        
        m_currentTree->SetClassId(classId);
    }
    void SetInitialOffset(ReturnType val) { m_forest->SetInitialOffset(val); } 
    void SetNumberOfClasses(int32_t numClasses) { m_forest->SetNumClasses(numClasses); }
    
    // Create a new node in the current tree
    NodeIndexType NewNode(ThresholdType threshold, FeatureIndexType featureIndex) { return m_currentTree->NewNode(threshold, featureIndex); }
    // Set the parent of a node
    void SetNodeParent(NodeIndexType node, NodeIndexType parent) { m_currentTree->SetNodeParent(node, parent); }
    // Set right child of a node
    void SetNodeRightChild(NodeIndexType node, NodeIndexType child) { m_currentTree->SetNodeRightChild(node, child); }
    // Set left child of a node
    void SetNodeLeftChild(NodeIndexType node, NodeIndexType child) { m_currentTree->SetNodeLeftChild(node, child); }
    mlir::Type GetInputRowType() {
        const auto& features = m_forest->GetFeatures();
        mlir::Type elementType = GetMLIRType(InputElementType(), m_builder); // GetMLIRTypeFromString(features.front().type, m_builder);
        int64_t shape[] = { static_cast<int64_t>(features.size()) };
        return mlir::MemRefType::get(shape, elementType);
    }
    mlir::Type GetFunctionArgumentType() {
        const auto& features = m_forest->GetFeatures();
        mlir::Type elementType = GetMLIRType(InputElementType(), m_builder); //GetMLIRTypeFromString(features.front().type, m_builder);
        int64_t shape[] = { m_batchSize, static_cast<int64_t>(features.size())};
        // TODO This needs to be moved elsewhere. Seems too obscure a place for this!
        mlir::decisionforest::ForestJSONReader::GetInstance().SetRowSize(features.size());
        // auto affineMap = mlir::makeStridedLinearLayoutMap(mlir::ArrayRef<int64_t>({ static_cast<int64_t>(features.size()), 1 }), 0, elementType.getContext());
        // return mlir::MemRefType::get(shape, elementType, affineMap);
        return mlir::MemRefType::get(shape, elementType);
    }
    mlir::Type GetFunctionResultType() {
        return mlir::MemRefType::get(m_batchSize, GetMLIRType(ReturnType(), m_builder));
    }
    mlir::FunctionType GetFunctionType() {
        auto argType = GetFunctionArgumentType();
        auto resultType = GetFunctionResultType();
        return m_builder.getFunctionType({argType, resultType}, resultType);
    }
    mlir::func::FuncOp GetFunctionPrototype() {
        auto location = m_builder.getUnknownLoc();
        auto functionType = GetFunctionType();
        // TODO the function name needs to be an input or derived from the input
        // return mlir::FuncOp::create(location, std::string("Prediction_Function"), functionType);
        return m_builder.create<mlir::func::FuncOp>(location, std::string("Prediction_Function"), functionType, m_builder.getStringAttr("public"));
    }
    virtual mlir::decisionforest::TreeEnsembleType GetEnsembleType() {
        // All trees have the default tiling to start with.
        int32_t tileSize = 1;
        auto treeType = mlir::decisionforest::TreeType::get(GetMLIRType(ReturnType(), m_builder), tileSize, 
                                                            GetMLIRType(ThresholdType(), m_builder), 
                                                            GetMLIRType(FeatureIndexType(), m_builder), GetMLIRType(int32_t(), m_builder),
                                                            mlir::decisionforest::UseSparseTreeRepresentation, 
                                                            m_builder.getIntegerType(m_childIndexBitWidth));

        auto forestType = mlir::decisionforest::TreeEnsembleType::get(GetMLIRType(ReturnType(), m_builder),
                                                                      m_forest->NumTrees(), GetInputRowType(), 
                                                                      mlir::decisionforest::ReductionType::kAdd, treeType);
        return forestType;
    }
public:
    static std::string ModelGlobalJSONFilePathFromJSONFilePath(const std::string& jsonFilePath) {
        return jsonFilePath + ".treebeard-globals.json";
    }

    ModelJSONParser(const std::string& jsonFilePath, const std::string& modelGlobalsJSONFilePath, mlir::MLIRContext& context, 
                    int32_t batchSize, double_t initialValue, const std::string& statsProfileCSV)
        : m_jsonFilePath(jsonFilePath), m_modelGlobalsJSONFilePath(modelGlobalsJSONFilePath),
          m_statsProfileCSV(statsProfileCSV), m_forest(new DecisionForestType(initialValue)), 
          m_currentTree(nullptr), m_schedule(nullptr), m_context(context), m_builder(&context),
          m_batchSize(batchSize), m_childIndexBitWidth(1) 
    {
        m_module = mlir::ModuleOp::create(m_builder.getUnknownLoc(), llvm::StringRef("MyModule"));
        // m_modelGlobalsJSONFilePath = ModelGlobalJSONFilePathFromJSONFilePath(jsonFilePath);
        mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(m_modelGlobalsJSONFilePath);
        mlir::decisionforest::ForestJSONReader::GetInstance().SetBatchSize(batchSize);
        mlir::decisionforest::ForestJSONReader::GetInstance().SetInputElementBitWidth(sizeof(InputElementType)*8);
        mlir::decisionforest::ForestJSONReader::GetInstance().SetReturnTypeBitWidth(sizeof(ReturnType)*8);
    }

    ModelJSONParser(const std::string& jsonFilePath, const std::string& modelGlobalsJSONFilePath, mlir::MLIRContext& context, 
                    int32_t batchSize, double_t initialValue)
        : ModelJSONParser(jsonFilePath, modelGlobalsJSONFilePath, context, batchSize, initialValue, "")
    { }


    ModelJSONParser(const std::string& jsonFilePath, const std::string& modelGlobalsJSONFilePath, mlir::MLIRContext& context, int32_t batchSize)
        : ModelJSONParser(jsonFilePath, modelGlobalsJSONFilePath, context, batchSize, 0.0, "")
    {
    }
    
    virtual ~ModelJSONParser() {
        delete m_schedule;
        delete m_forest;
    }

    mlir::decisionforest::Schedule* GetSchedule() { return m_schedule; }
    const std::string& GetModelGlobalsJSONFilePath() { return m_modelGlobalsJSONFilePath; }

    virtual void Parse() = 0;

    // Get the forest pointer
    DecisionForestType* GetForest() { return m_forest; }

    mlir::ModuleOp GetEvaluationFunction() { 
        if (m_statsProfileCSV != "")
            TreeBeard::Profile::ReadProbabilityProfile(*m_forest, m_statsProfileCSV);
        
        mlir::func::FuncOp function(GetFunctionPrototype());
        if (!function)
            return nullptr;

        // Function body. MLIR's entry block has the same arg list as function
        auto &entryBlock = *function.addEntryBlock();

        m_builder.setInsertionPointToStart(&entryBlock);

        auto forestType = GetEnsembleType();
        auto forestAttribute = mlir::decisionforest::DecisionForestAttribute::get(forestType, *m_forest);
        
        auto scheduleType = mlir::decisionforest::ScheduleType::get(&m_context);
        m_schedule = new mlir::decisionforest::Schedule(m_batchSize, m_forest->NumTrees());
        auto scheduleAttribute = mlir::decisionforest::ScheduleAttribute::get(scheduleType, m_schedule);
        
        auto predictOp = m_builder.create<mlir::decisionforest::PredictForestOp>(
            m_builder.getUnknownLoc(),
            GetFunctionResultType(),
            forestAttribute,
            static_cast<mlir::Value>(entryBlock.getArguments()[0]),
            entryBlock.getArguments()[1], scheduleAttribute);

        m_builder.create<mlir::func::ReturnOp>(m_builder.getUnknownLoc(), static_cast<mlir::Value>(predictOp));
        if (failed(mlir::verify(m_module))) {
            m_module.emitError("Module verification error");
            return nullptr;
        }
        m_module.push_back(function);

        return m_module;
    }

    void SetChildIndexBitWidth(int32_t value) { m_childIndexBitWidth = value; }
};
}

#endif //_MODEL_JSON_PARSER_H_