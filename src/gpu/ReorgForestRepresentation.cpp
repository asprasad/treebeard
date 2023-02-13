#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

#include "Representations.h"
#include "OpLoweringUtils.h"
#include "LIRLoweringHelpers.h"
#include "Logger.h"
#include "ReorgForestRepresentation.h"

using namespace mlir;
using namespace mlir::decisionforest::helpers;

namespace mlir
{
namespace decisionforest
{

ReorgForestSerializer::ReorgForestSerializer(const std::string& filename)
  :IModelSerializer(filename)
{ }

int32_t ComputeBufferSize(decisionforest::DecisionForest<>& forest) {
  int32_t maxDepth = -1;
  for (auto &tree: forest.GetTrees()) {
    auto depth = tree->GetTreeDepth();
    if (depth > maxDepth)
      depth = maxDepth;
  }
  return static_cast<int32_t>(forest.NumTrees()) * (std::pow(2, maxDepth) - 1);
}

void ReorgForestSerializer::WriteSingleTreeIntoReorgBuffer(mlir::decisionforest::DecisionForest<>& forest, int32_t treeIndex) {
  auto& tree = forest.GetTree(treeIndex);
  auto thresholds = tree.GetThresholdArray();
  auto featureIndices = tree.GetFeatureIndexArray();
  assert (thresholds.size() == featureIndices.size());
  for (size_t i=0 ; i<thresholds.size() ; ++i) {
    auto bufferIndex = i*forest.NumTrees() + treeIndex;
    m_thresholds.at(bufferIndex) = thresholds[i];
    m_featureIndices.at(bufferIndex) = featureIndices[i];
  }
}

void ReorgForestSerializer::Persist(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType) {
  m_numberOfTrees = forestType.getNumberOfTrees();
  m_numberOfClasses = forest.IsMultiClassClassifier() ? forest.GetNumClasses() : 1;
  m_rowSize = forestType.getRowType().cast<MemRefType>().getShape()[0];
  auto treeType = forestType.getTreeType(0).cast<decisionforest::TreeType>();
  m_thresholdBitWidth = treeType.getThresholdType().getIntOrFloatBitWidth();
  m_featureIndexBitWidth = treeType.getFeatureIndexType().getIntOrFloatBitWidth();
  
  // TODO Need to pass these in the constructor?
  // m_batchSize = ??
  // m_inputElementBitwidth = ?
  // m_returnTypeBitWidth = ?

  auto bufferSize = ComputeBufferSize(forest);
  m_thresholds.resize(bufferSize, NAN);
  m_featureIndices.resize(bufferSize, -1);

  for (int32_t i=0 ; i<(int32_t)forest.NumTrees() ; ++i)
    this->WriteSingleTreeIntoReorgBuffer(forest, i);

  // TODO Write out a JSON file
}

}
}