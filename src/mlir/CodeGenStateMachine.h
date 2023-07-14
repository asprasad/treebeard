#ifndef CODEGEN_STATE_MACHINE
#define CODEGEN_STATE_MACHINE

#include <iostream>
#include <memory>
#include "Dialect.h"
// #include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "Representations.h"

namespace mlir
{
namespace decisionforest
{
    class ICodeGeneratorStateMachine {
    public:
        // Returns false if there's no code to emit.
        virtual bool EmitNext(ConversionPatternRewriter& rewriter, Location& location) = 0;
        virtual std::vector<Value> GetResult() = 0;
        ~ICodeGeneratorStateMachine() {};
    };

    class InterleavedCodeGenStateMachine : public ICodeGeneratorStateMachine {
    private:
        std::vector<std::unique_ptr<ICodeGeneratorStateMachine>> m_codeGenStateMachines;
        bool m_startedEmitting = false;
        bool m_finishedEmitting = false;
    public:
        InterleavedCodeGenStateMachine() {};
        InterleavedCodeGenStateMachine(const InterleavedCodeGenStateMachine&) = delete;
        InterleavedCodeGenStateMachine& operator=(const InterleavedCodeGenStateMachine&) = delete;

        void AddStateMachine(std::unique_ptr<ICodeGeneratorStateMachine> codeGenerator) {
          assert(!m_startedEmitting);
          m_codeGenStateMachines.emplace_back(std::move(codeGenerator));
        }

        virtual bool EmitNext (ConversionPatternRewriter& rewriter, Location& location) override {
          m_startedEmitting = true;
          bool moreCodeToEmit = false;
          for (auto& codeGenerator : m_codeGenStateMachines) {
              moreCodeToEmit |= codeGenerator->EmitNext(rewriter, location);
          }
          m_finishedEmitting = !moreCodeToEmit;
          return moreCodeToEmit;
        }

        virtual std::vector<Value> GetResult () override {
          assert(m_finishedEmitting);
          std::vector<Value> results;

          for (auto& codeGenerator : m_codeGenStateMachines) {
              auto result = codeGenerator->GetResult();
              results.insert(
              results.end(),
              std::make_move_iterator(result.begin()),
              std::make_move_iterator(result.end())
              );
          }

          return results;
        }

        ~InterleavedCodeGenStateMachine() {
          m_codeGenStateMachines.clear();
        }
    };

class ScalarTraverseTileCodeGenerator : public ICodeGeneratorStateMachine {
  private:
    enum TraverseState { kLoadThreshold, kLoadFeatureIndex, kLoadFeature, kCompare, kNextNode, kDone };
    std::shared_ptr<IRepresentation> m_representation;
    TraverseState m_state;
    Value m_rowMemref;
    Type m_resultType;
    Value m_nodeToTraverse;
    decisionforest::NodeToIndexOp m_nodeIndex;
    decisionforest::LoadTileThresholdsOp m_loadThresholdOp;
    decisionforest::LoadTileFeatureIndicesOp m_loadFeatureIndexOp;
    memref::LoadOp m_loadFeatureOp;
    arith::ExtUIOp m_comparisonUnsigned;
    Value m_result;
    std::vector<mlir::Value> m_extraLoads;
    Value m_tree;
    mlir::arith::CmpFPredicateAttr m_cmpPredicateAttr;
  public:
    ScalarTraverseTileCodeGenerator(Value rowMemref, Value node, 
                                    Type resultType,
                                    std::shared_ptr<IRepresentation> representation,
                                    Value tree,
                                    mlir::arith::CmpFPredicateAttr cmpPredicateAttr);
    bool EmitNext(ConversionPatternRewriter& rewriter, Location& location) override;
    std::vector<Value> GetResult() override;
};

class VectorTraverseTileCodeGenerator : public ICodeGeneratorStateMachine {
  private:
    enum TraverseState { kLoadThreshold, kLoadFeatureIndex, kLoadTileShape, kLoadChildIndex, kLoadFeature, kCompare, kNextNode, kDone };
    std::shared_ptr<IRepresentation> m_representation;
    TraverseState m_state;
    Value m_tree;
    Value m_rowMemref;
    Type m_resultType;
    VectorType m_featureIndexVectorType;
    VectorType m_thresholdVectorType;
    Type m_tileShapeType;
    int32_t m_tileSize;

    Value m_nodeToTraverse;
    decisionforest::NodeToIndexOp m_nodeIndex;
    decisionforest::LoadTileThresholdsOp m_loadThresholdOp;
    decisionforest::LoadTileFeatureIndicesOp m_loadFeatureIndexOp;
    arith::IndexCastOp m_loadTileShapeIndexOp;
    arith::IndexCastOp m_leafBitMask;
    vector::GatherOp m_features;
    Value m_comparisonIndex;
    Value m_result;
    std::vector<mlir::Value> m_extraLoads;

    std::function<Value(Value)> m_getLutFunc;
    mlir::arith::CmpFPredicateAttr m_cmpPredicateAttr;
  public:
    VectorTraverseTileCodeGenerator(Value tree, 
                                    Value rowMemref,
                                    Value node,
                                    Type resultType, 
                                    std::shared_ptr<IRepresentation> representation,
                                    std::function<Value(Value)> getLutFunc,
                                    mlir::arith::CmpFPredicateAttr cmpPredicateAttr);
    bool EmitNext(ConversionPatternRewriter& rewriter, Location& location) override;
    std::vector<Value> GetResult() override;
};

#ifdef TREEBEARD_GPU_SUPPORT
struct GPUTraverseLoweringState {
  std::map<Operation*, Value> getRootOpToShMemNodeArrayMap;
  Value m_resultsSharedMemBuffer;

  Value GetSharedMemoryResultsMatrix(gpu::LaunchOp launchOp, 
                                   ConversionPatternRewriter &rewriter,
                                   Location location,
                                   int32_t tileSize);
};

class GPUVectorTraverseTileCodeGenerator : public ICodeGeneratorStateMachine {
  private:
    enum TraverseState { kInit, kLoadThreshold, kLoadFeatureIndex, kLoadTileShape, kLoadChildIndex, kLoadFeature, kCompare, kNextNode, kDone };
    TraverseState m_state;
    std::shared_ptr<IRepresentation> m_representation;
    decisionforest::CooperativeTraverseTreeTileOp m_traverseTileOp;

    int32_t m_currentTileElem;
    int32_t m_tileSize;

    Type m_resultType;
    Type m_featureIndexType;
    Type m_thresholdType;
    Type m_tileShapeType;

    Value m_tree;
    Value m_rowMemref;

    Value m_nodeToTraverse;
    Value m_nodeIndex;
    Value m_loadThresholdOp;
    Value m_loadFeatureIndexOp;
    arith::IndexCastOp m_loadTileShapeIndexOp;
    arith::IndexCastOp m_leafBitMask;
    Value m_features;
    Value m_comparisonIndex;
    Value m_result;
    std::vector<mlir::Value> m_extraLoads;
    Value m_threadBlockThreadId;
    Value m_threadId;

    std::function<Value(Value)> m_getLutFunc;
    mlir::arith::CmpFPredicateAttr m_cmpPredicateAttr;
    std::shared_ptr<GPUTraverseLoweringState> m_traverseLoweringState;
    Operation* m_getRootOp;
    Value m_nodeIndexShMemBuffer;
    Value m_cmpOutcomesShMemBuffer;
    Value m_threadBaseId;
    Value m_tileSizeConst;

  public:
    GPUVectorTraverseTileCodeGenerator(decisionforest::CooperativeTraverseTreeTileOp traverseTileOp,
                                    Type resultType, 
                                    std::shared_ptr<IRepresentation> representation,
                                    std::function<Value(Value)> getLutFunc,
                                    mlir::arith::CmpFPredicateAttr cmpPredicateAttr,
                                    std::shared_ptr<GPUTraverseLoweringState> traverseLoweringState);
    bool EmitNext(ConversionPatternRewriter& rewriter, Location& location) override;
    std::vector<Value> GetResult() override;
};
#endif

} // namespace decisionforest
} // namespace mlir

#endif // CODEGEN_STATE_MACHINE