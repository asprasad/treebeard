#ifndef _MODEL_JSON_PARSER_H_
#define _MODEL_JSON_PARSER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "json.hpp"

#include "DecisionForest.h"
#include "Dialect.h"
#include "ModelSerializers.h"
#include "StatsUtils.h"
#include "TreeTilingUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"

#include "LIRLoweringHelpers.h"

using json = nlohmann::json;

/*
//++
// The abstract base class from which all JSON model parsers derive.
// Also implements callbacks that construct the actual MLIR structures.
//--
*/
namespace TreeBeard {

template <typename T>
mlir::Type GetMLIRType(const T &val, mlir::OpBuilder &builder) {
  assert(false);
  return mlir::Type();
}

template <>
inline mlir::Type GetMLIRType(const int8_t &val, mlir::OpBuilder &builder) {
  return builder.getIntegerType(8);
}

template <>
inline mlir::Type GetMLIRType(const int16_t &val, mlir::OpBuilder &builder) {
  return builder.getIntegerType(16);
}

template <>
inline mlir::Type GetMLIRType(const int32_t &val, mlir::OpBuilder &builder) {
  return builder.getI32Type();
}

template <>
inline mlir::Type GetMLIRType(const float &val, mlir::OpBuilder &builder) {
  return builder.getF32Type();
}

template <>
inline mlir::Type GetMLIRType(const double &val, mlir::OpBuilder &builder) {
  return builder.getF64Type();
}

template <typename T>
mlir::Type GetMLIRType(const T &val, mlir::MLIRContext &context) {
  assert(false);
  return mlir::Type();
}

template <>
inline mlir::Type GetMLIRType(const int8_t &val, mlir::MLIRContext &context) {
  return mlir::IntegerType::get(&context, 8);
}

template <>
inline mlir::Type GetMLIRType(const int16_t &val, mlir::MLIRContext &context) {
  return mlir::IntegerType::get(&context, 16);
}

template <>
inline mlir::Type GetMLIRType(const int32_t &val, mlir::MLIRContext &context) {
  return mlir::IntegerType::get(&context, 32);
}

template <>
inline mlir::Type GetMLIRType(const float &val, mlir::MLIRContext &context) {
  return mlir::Float32Type::get(&context);
}

template <>
inline mlir::Type GetMLIRType(const double &val, mlir::MLIRContext &context) {
  return mlir::Float64Type::get(&context);
}

inline mlir::Type GetMLIRTypeFromString(const std::string &typestr,
                                        mlir::OpBuilder &builder) {
  if (typestr == "float")
    return builder.getF64Type();
  else
    assert(false);
  return mlir::Type();
}

class ForestCreator {
protected:
  using DecisionForestType = mlir::decisionforest::DecisionForest;
  using DecisionTreeType = mlir::decisionforest::DecisionTree;

  std::string m_statsProfileCSV;
  DecisionForestType *m_forest;
  DecisionTreeType *m_currentTree;
  mlir::decisionforest::Schedule *m_schedule;
  mlir::MLIRContext &m_context;
  mlir::ModuleOp m_module;
  mlir::OpBuilder m_builder;
  int32_t m_batchSize;
  int32_t m_childIndexBitWidth;
  mlir::Type m_thresholdType;
  mlir::Type m_featureIndexType;
  mlir::Type m_nodeIndexType;
  mlir::Type m_returnType;
  mlir::Type m_inputElementType;
  mlir::arith::CmpFPredicate m_cmpPredicate;

  std::shared_ptr<mlir::decisionforest::IModelSerializer> m_serializer;

  void SetReductionType(mlir::decisionforest::ReductionType reductionType) {
    m_forest->SetReductionType(reductionType);
  }
  void AddFeature(const std::string &featureName, const std::string &type) {
    m_forest->AddFeature(featureName, type);
  }
  void NewTree() { m_currentTree = &(m_forest->NewTree()); }
  void EndTree() { m_currentTree = nullptr; }
  void SetTreeNumberOfFeatures(size_t numFeatures) {
    m_currentTree->SetNumberOfFeatures(numFeatures);
  }
  void SetTreeScalingFactor(double scale) {
    m_currentTree->SetTreeScalingFactor(scale);
  }
  void SetTreeClassId(int32_t classId) {
    if (m_forest->IsMultiClassClassifier())
      assert(classId < m_forest->GetNumClasses() &&
             "ClassId should be lesser than number of classes for multi-class "
             "classifiers.");

    m_currentTree->SetClassId(classId);
  }
  void SetInitialOffset(double val) { m_forest->SetInitialOffset(val); }
  void SetNumberOfClasses(int32_t numClasses) {
    m_forest->SetNumClasses(numClasses);
  }

  // Create a new node in the current tree
  int64_t NewNode(double threshold, int64_t featureIndex) {
    return m_currentTree->NewNode(threshold, featureIndex);
  }
  // Set the parent of a node
  void SetNodeParent(int64_t node, int64_t parent) {
    m_currentTree->SetNodeParent(node, parent);
  }
  // Set right child of a node
  void SetNodeRightChild(int64_t node, int64_t child) {
    m_currentTree->SetNodeRightChild(node, child);
  }
  // Set left child of a node
  void SetNodeLeftChild(int64_t node, int64_t child) {
    m_currentTree->SetNodeLeftChild(node, child);
  }
  void SetPredicateType(mlir::arith::CmpFPredicate value) {
    m_cmpPredicate = value;
  }
  mlir::Type GetInputRowType() {
    const auto &features = m_forest->GetFeatures();
    int64_t shape[] = {static_cast<int64_t>(features.size())};
    return mlir::MemRefType::get(shape, m_inputElementType);
  }
  mlir::Type GetFunctionArgumentType() {
    const auto &features = m_forest->GetFeatures();
    int64_t shape[] = {m_batchSize, static_cast<int64_t>(features.size())};
    return mlir::MemRefType::get(shape, m_inputElementType);
  }
  mlir::Type GetFunctionResultType() {
    return mlir::MemRefType::get(m_batchSize, m_returnType);
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
    // return mlir::FuncOp::create(location, std::string("Prediction_Function"),
    // functionType);
    auto predictionFunction = m_builder.create<mlir::func::FuncOp>(
        location, std::string("Prediction_Function"), functionType);
    predictionFunction.setPublic();
    return predictionFunction;
  }

  void AddConstIntegerGetFunction(const std::string &functionName,
                                  int32_t value) {
    auto location = m_builder.getUnknownLoc();
    auto int32Type = m_builder.getI32Type();
    auto functionType = m_builder.getFunctionType({}, int32Type);
    auto func = m_builder.create<mlir::func::FuncOp>(location, functionName,
                                                     functionType);
    func.setPublic();

    auto insertPoint = m_builder.saveInsertionPoint();

    auto &entryBlock = *func.addEntryBlock();
    m_builder.setInsertionPointToStart(&entryBlock);
    auto constVal = m_builder.create<mlir::arith::ConstantIntOp>(
        location, value, int32Type);
    m_builder.create<mlir::func::ReturnOp>(location, constVal.getResult());

    m_builder.restoreInsertionPoint(insertPoint);

    m_module.push_back(func);
  }

  virtual mlir::decisionforest::TreeEnsembleType GetEnsembleType() {
    // All trees have the default tiling to start with.
    int32_t tileSize = 1;
    auto treeType = mlir::decisionforest::TreeType::get(
        m_returnType, tileSize, m_thresholdType, m_featureIndexType,
        GetMLIRType(int32_t(), m_builder),
        m_builder.getIntegerType(m_childIndexBitWidth));

    auto forestType = mlir::decisionforest::TreeEnsembleType::get(
        m_returnType, m_forest->NumTrees(), GetInputRowType(),
        mlir::decisionforest::ReductionType::kAdd, treeType);
    return forestType;
  }

public:
  static std::string
  ModelGlobalJSONFilePathFromJSONFilePath(const std::string &jsonFilePath) {
    return jsonFilePath + ".treebeard-globals.json";
  }

  ForestCreator(
      std::shared_ptr<mlir::decisionforest::IModelSerializer> serializer,
      mlir::MLIRContext &context, int32_t batchSize, double_t initialValue,
      const std::string &statsProfileCSV, mlir::Type thresholdType,
      mlir::Type featureIndexType, mlir::Type nodeIndexType,
      mlir::Type returnType, mlir::Type inputElementType)
      : m_statsProfileCSV(statsProfileCSV),
        m_forest(new DecisionForestType(initialValue)), m_currentTree(nullptr),
        m_schedule(nullptr), m_context(context), m_builder(&context),
        m_batchSize(batchSize), m_childIndexBitWidth(1),
        m_thresholdType(thresholdType), m_featureIndexType(featureIndexType),
        m_nodeIndexType(nodeIndexType), m_returnType(returnType),
        m_inputElementType(inputElementType),
        m_cmpPredicate(mlir::arith::CmpFPredicate::ULT),
        m_serializer(std::move(serializer)) {
    m_module = mlir::ModuleOp::create(m_builder.getUnknownLoc(),
                                      llvm::StringRef("MyModule"));
  }

  ForestCreator(
      const std::shared_ptr<mlir::decisionforest::IModelSerializer> &serializer,
      mlir::MLIRContext &context, int32_t batchSize, double_t initialValue,
      mlir::Type thresholdType, mlir::Type featureIndexType,
      mlir::Type nodeIndexType, mlir::Type returnType,
      mlir::Type inputElementType)
      : ForestCreator(serializer, context, batchSize, initialValue, "",
                      thresholdType, featureIndexType, nodeIndexType,
                      returnType, inputElementType) {}

  ForestCreator(
      const std::shared_ptr<mlir::decisionforest::IModelSerializer> &serializer,
      mlir::MLIRContext &context, int32_t batchSize, mlir::Type thresholdType,
      mlir::Type featureIndexType, mlir::Type nodeIndexType,
      mlir::Type returnType, mlir::Type inputElementType)
      : ForestCreator(serializer, context, batchSize, 0.0, "", thresholdType,
                      featureIndexType, nodeIndexType, returnType,
                      inputElementType) {}

  virtual ~ForestCreator() {
    delete m_schedule;
    delete m_forest;
  }

  mlir::decisionforest::Schedule *GetSchedule() { return m_schedule; }
  const std::string &GetModelGlobalsJSONFilePath() {
    return m_serializer->GetFilePath();
  }

  virtual void ConstructForest() = 0;

  // Get the forest pointer
  DecisionForestType *GetForest() { return m_forest; }

  mlir::ModuleOp GetEvaluationFunction() {
    if (m_statsProfileCSV != "")
      TreeBeard::Profile::ReadProbabilityProfile(*m_forest, m_statsProfileCSV);

    // Add getters for some constants we rely on at runtime
    AddConstIntegerGetFunction("GetBatchSize", m_batchSize);
    AddConstIntegerGetFunction("GetRowSize", m_forest->GetFeatures().size());
    AddConstIntegerGetFunction("GetInputTypeBitWidth",
                               m_inputElementType.getIntOrFloatBitWidth());
    AddConstIntegerGetFunction("GetReturnTypeBitWidth",
                               m_returnType.getIntOrFloatBitWidth());

    mlir::func::FuncOp function(GetFunctionPrototype());
    if (!function)
      return nullptr;

    // Function body. MLIR's entry block has the same arg list as function
    auto &entryBlock = *function.addEntryBlock();

    m_builder.setInsertionPointToStart(&entryBlock);

    auto forestType = GetEnsembleType();
    auto forestAttribute = mlir::decisionforest::DecisionForestAttribute::get(
        forestType, *m_forest);

    auto scheduleType = mlir::decisionforest::ScheduleType::get(&m_context);
    m_schedule =
        new mlir::decisionforest::Schedule(m_batchSize, m_forest->NumTrees());
    auto scheduleAttribute =
        mlir::decisionforest::ScheduleAttribute::get(scheduleType, m_schedule);
    auto predicateAttribute =
        mlir::arith::CmpFPredicateAttr::get(&m_context, m_cmpPredicate);
    auto predictOp = m_builder.create<mlir::decisionforest::PredictForestOp>(
        m_builder.getUnknownLoc(), GetFunctionResultType(), forestAttribute,
        predicateAttribute,
        static_cast<mlir::Value>(entryBlock.getArguments()[0]),
        entryBlock.getArguments()[1], scheduleAttribute);

    m_builder.create<mlir::func::ReturnOp>(m_builder.getUnknownLoc(),
                                           static_cast<mlir::Value>(predictOp));
    if (failed(mlir::verify(m_module))) {
      m_module.emitError("Module verification error");
      return nullptr;
    }
    m_module.push_back(function);

    return m_module;
  }

  void SetChildIndexBitWidth(int32_t value) { m_childIndexBitWidth = value; }

  mlir::MLIRContext &GetContext() { return m_context; }
  mlir::ModuleOp GetModule() { return m_module; }
  std::shared_ptr<mlir::decisionforest::IModelSerializer> GetSerializer() {
    return m_serializer;
  }
};
} // namespace TreeBeard

#endif //_MODEL_JSON_PARSER_H_
