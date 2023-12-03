#ifndef _SCHEDULE_H_
#define _SCHEDULE_H_

#include <cassert>
#include <fstream>
#include <functional>
#include <list>
#include <map>
#include <string>
#include <vector>

namespace mlir {
namespace decisionforest {

class IndexVariable;
class IndexDerivationTreeVisitor;
class Schedule;

class IndexDerivationTreeNode {
public:
  virtual void Visit(IndexDerivationTreeVisitor &visitor) = 0;
  virtual void Validate() = 0;
};

class IndexModifier : public IndexDerivationTreeNode {
protected:
  // TODO Should this just be an IndexDerivationTreeNode? Maybe a parent could
  // directly be an Index modifier
  IndexVariable *m_parent;

public:
  IndexModifier(IndexVariable *parent) : m_parent(parent) {}
  virtual ~IndexModifier() {}
  void Visit(IndexDerivationTreeVisitor &visitor) override = 0;
  void Validate() override = 0;
  IndexVariable *GetParent() const { return m_parent; }
};

class TileIndexModifier : public IndexModifier {
  IndexVariable *m_sourceIndex;
  IndexVariable *m_outerIndex;
  IndexVariable *m_innerIndex;
  int32_t m_tileSize;

public:
  TileIndexModifier(IndexVariable &source, IndexVariable &outer,
                    IndexVariable &inner, int32_t tileSize)
      : IndexModifier(&source), m_sourceIndex(&source), m_outerIndex(&outer),
        m_innerIndex(&inner), m_tileSize(tileSize) {}
  IndexVariable &OuterIndex() { return *m_outerIndex; }
  IndexVariable &InnerIndex() { return *m_innerIndex; }
  int32_t TileSize() { return m_tileSize; }
  IndexVariable &SourceIndex() { return *m_sourceIndex; }
  void Validate() override;
  void Visit(IndexDerivationTreeVisitor &visitor) override;
};

class SplitIndexModifier : public IndexModifier {
  IndexVariable *m_sourceIndex;
  IndexVariable *m_firstIndex;
  IndexVariable *m_secondIndex;
  int32_t m_splitIteration;

public:
  SplitIndexModifier(IndexVariable &source, IndexVariable &first,
                     IndexVariable &second, int32_t splitIteration)
      : IndexModifier(&source), m_sourceIndex(&source), m_firstIndex(&first),
        m_secondIndex(&second), m_splitIteration(splitIteration) {}
  IndexVariable &SourceIndex() { return *m_sourceIndex; }
  IndexVariable &FirstIndex() { return *m_firstIndex; }
  IndexVariable &SecondIndex() { return *m_secondIndex; }
  int32_t SplitIteration() { return m_splitIteration; }
  void Validate() override;
  void Visit(IndexDerivationTreeVisitor &visitor) override;
};

class DuplicateIndexModifier : public IndexModifier {
  IndexVariable *m_sourceIndex;
  IndexVariable *m_firstIndex;
  IndexVariable *m_secondIndex;

public:
  DuplicateIndexModifier(IndexVariable &source, IndexVariable &first,
                         IndexVariable &second)
      : IndexModifier(&source), m_sourceIndex(&source), m_firstIndex(&first),
        m_secondIndex(&second) {}
  IndexVariable &SourceIndex() { return *m_sourceIndex; }
  IndexVariable &FirstIndex() { return *m_firstIndex; }
  IndexVariable &SecondIndex() { return *m_secondIndex; }
  void Validate() override;
  void Visit(IndexDerivationTreeVisitor &visitor) override;
};

class SpecializeIndexModifier : public IndexModifier {
  IndexVariable *m_sourceIndex;
  std::vector<IndexVariable *> m_specializedIndices;

public:
  SpecializeIndexModifier(IndexVariable &source,
                          std::vector<IndexVariable *> &specializedIndices)
      : IndexModifier(&source), m_sourceIndex(&source),
        m_specializedIndices(specializedIndices) {}

  SpecializeIndexModifier(IndexVariable &source)
      : IndexModifier(&source), m_sourceIndex(&source) {}
  IndexVariable &SourceIndex() { return *m_sourceIndex; }
  std::vector<IndexVariable *> &SpecializedIndices() {
    return m_specializedIndices;
  }

  void AddSpecializedIndex(IndexVariable &index) {
    m_specializedIndices.push_back(&index);
  }

  void Validate() override;
  void Visit(IndexDerivationTreeVisitor &visitor) override;
};

class IndexVariable : public IndexDerivationTreeNode {
  friend class Schedule;

public:
  enum class IndexVariableType { kBatch, kTree, kUnknown };
  enum class GPUConstruct { Grid, ThreadBlock, None };
  enum class Dimension { X, Y, Z };

  struct IndexRange {
    int32_t m_start = -1;
    int32_t m_stop = -1;
    int32_t m_step = 0;
  };
  struct GPUDimension {
    GPUConstruct construct;
    Dimension dimension;
  };

protected:
  std::string m_name;
  IndexRange m_range;
  IndexVariable *m_containingLoop;
  std::vector<IndexVariable *> m_containedLoops;
  IndexVariableType m_type = IndexVariableType::kUnknown;
  GPUConstruct m_gpuConstruct = GPUConstruct::None;
  Dimension m_dimension = Dimension::X;

  // Fields for the index modifier tree
  IndexModifier
      *m_parentModifier; // The modifier that resulted in this index variable
  IndexModifier
      *m_modifier; // Modifier that derives new index variables from this one

  bool m_pipelined = false;
  // Only valid if m_pipelined is true
  int32_t m_unpipelinedStepSize = -1;

  bool m_simdized = false;
  bool m_parallel = false;
  bool m_unrolled = false;

  // Reduction optimizations
  bool m_atomicReduce = false;
  int32_t m_vectorReduceWidth = 1;
  bool m_sharedReduce = false;
  bool m_specializeIterations = false;

  int32_t m_treeWalkUnrollFactor = -1;

  int32_t m_iterationsToPeel = -1;
  bool m_peelWalk = false;

  bool m_cache = false;

  // Index variables can only be constructed through the Schedule object
  IndexVariable(const std::string &name)
      : m_name(name), m_containingLoop(nullptr), m_parentModifier(nullptr),
        m_modifier(nullptr), m_treeWalkUnrollFactor(-1) {}

  // Index variables can't be copied
  IndexVariable(const IndexVariable &other) = default;

public:
  IndexVariable *GetContainingLoop() const { return m_containingLoop; }
  const std::vector<IndexVariable *> &GetContainedLoops() const {
    return m_containedLoops;
  }
  IndexRange GetRange() const { return m_range; }
  void SetRange(IndexRange range) { m_range = range; }
  std::string GetName() const { return m_name; }

  bool Pipelined() const { return m_pipelined; }
  int32_t GetUnpipelinedStepSize() const {
    assert(m_pipelined);
    return m_unpipelinedStepSize;
  }
  bool Simdized() const { return m_simdized; }
  bool Parallel() const { return m_parallel; }
  bool Unroll() const { return m_unrolled; }
  bool UnrollTreeWalk() const { return m_treeWalkUnrollFactor > 0; }
  int32_t GetTreeWalkUnrollFactor() const { return m_treeWalkUnrollFactor; }
  void SetTreeWalkUnrollFactor(int32_t unrollFactor) {
    m_treeWalkUnrollFactor = unrollFactor;
  }

  bool AtomicReduce() const { return m_atomicReduce; }
  bool VectorReduce() const { return m_vectorReduceWidth != 1; }
  int32_t VectorReduceWidth() const { return m_vectorReduceWidth; }
  bool SharedReduce() const { return m_sharedReduce; }
  bool SpecializeIterations() const { return m_specializeIterations; }

  bool PeelWalk() const { return m_peelWalk; }
  int32_t IterationsToPeel() const { return m_iterationsToPeel; }

  bool Cache() const { return m_cache; }

  void Visit(IndexDerivationTreeVisitor &visitor) override;
  void Validate() override;

  IndexModifier *GetParentModifier() const { return m_parentModifier; }
  IndexModifier *GetIndexModifier() const { return m_modifier; }
  IndexVariableType GetType() const { return m_type; }

  // GPU Support
  void SetGPUDimension(GPUConstruct construct, Dimension dimension) {
    m_gpuConstruct = construct;
    m_dimension = dimension;
  }
  GPUDimension GetGPUDimension() const {
    return GPUDimension{m_gpuConstruct, m_dimension};
  }
};

class Schedule {
public:
  class IterationSpecializationInfo;

private:
  std::list<IndexVariable *> m_indexVars;
  std::list<IndexModifier *> m_indexModifiers;

  IndexVariable m_treeIndex;
  IndexVariable m_batchIndex;
  IndexVariable m_rootIndex;

  int32_t m_batchSize;
  int32_t m_forestSize;

  Schedule(const Schedule &) = delete;
  void DuplicateIndexVariables(
      IndexVariable &index,
      std::map<IndexVariable *, std::pair<IndexVariable *, IndexVariable *>>
          &indexMap);
  void DuplicateIndexVariables(IndexVariable &index, int32_t numCopies,
                               Schedule::IterationSpecializationInfo &info);
  void WriteIndexToDOTFile(IndexVariable *index, std::ofstream &fout);

public:
  class IterationSpecializationInfo {
  public:
    friend class Schedule;

    using SingleIterationMap = std::map<IndexVariable *, IndexVariable *>;
    std::vector<SingleIterationMap> m_iterationMaps;

    IndexVariable *GetCorrespodingIndex(IndexVariable *index,
                                        int32_t iteration) {
      auto iter = m_iterationMaps[iteration].find(index);
      assert(iter != m_iterationMaps[iteration].end());
      return iter->second;
    }
  };

  typedef std::map<IndexVariable *, std::pair<IndexVariable *, IndexVariable *>>
      IndexVariableMapType;
  Schedule(int32_t batchSize, int32_t forestSize);

  IndexVariable &NewIndexVariable(const std::string &name);
  IndexVariable &NewIndexVariable(const IndexVariable &indexVar);

  // Loop Modifiers
  Schedule &Tile(IndexVariable &index, IndexVariable &outer,
                 IndexVariable &inner, int32_t tileSize);
  Schedule &Reorder(const std::vector<IndexVariable *> &indices);
  Schedule &
  Split(IndexVariable &index, IndexVariable &first, IndexVariable &second,
        int32_t splitIteration,
        std::map<IndexVariable *, std::pair<IndexVariable *, IndexVariable *>>
            &indexMap);

  Schedule &SpecializeIterations(IndexVariable &index,
                                 IterationSpecializationInfo &info);

  // Optimizations
  Schedule &Pipeline(IndexVariable &index, int32_t stepSize);
  Schedule &Simdize(IndexVariable &index);
  Schedule &Parallel(IndexVariable &index);
  Schedule &Unroll(IndexVariable &index);
  Schedule &PeelWalk(IndexVariable &index, int32_t numberOfIterations);
  Schedule &Cache(IndexVariable &index);

  // Reductions
  Schedule &AtomicReduce(IndexVariable &index);
  Schedule &VectorReduce(IndexVariable &index, int32_t width);
  Schedule &SharedReduce(IndexVariable &index);

  const IndexVariable *GetRootIndex() const { return &m_rootIndex; }
  IndexVariable &GetBatchIndex() { return m_batchIndex; }
  IndexVariable &GetTreeIndex() { return m_treeIndex; }
  std::string PrintToString();

  int32_t GetBatchSize() const { return m_batchSize; }
  int32_t GetForestSize() const { return m_forestSize; }

  void WriteToDOTFile(const std::string &dotFile);
  bool IsDefaultSchedule();
  void Finalize();
};

class IndexDerivationTreeVisitor {
public:
  virtual void VisitTileIndexModifier(TileIndexModifier &tileIndexModifier) = 0;
  virtual void VisitIndexVariable(IndexVariable &indexVar) = 0;
  virtual void VisitSplitIndexModifier(SplitIndexModifier &indexModifier) = 0;
  virtual void
  VisitDuplicateIndexModifier(DuplicateIndexModifier &indexModifier) = 0;
  virtual void
  VisitSpecializeIndexModifier(SpecializeIndexModifier &indexModifier) = 0;
};

class ScheduleManipulator {
public:
  virtual void Run(Schedule *schedule) = 0;
  virtual ~ScheduleManipulator() {}
};

typedef void (*ScheduleManipulator_t)(mlir::decisionforest::Schedule *schedule);

class ScheduleManipulationFunctionWrapper
    : public mlir::decisionforest::ScheduleManipulator {
  std::function<void(decisionforest::Schedule &)> m_func;

public:
  ScheduleManipulationFunctionWrapper(ScheduleManipulator_t func) {
    m_func = [func](decisionforest::Schedule &schedule) { func(&schedule); };
  }

  ScheduleManipulationFunctionWrapper(
      std::function<void(decisionforest::Schedule &)> func)
      : m_func(func) {}

  void Run(mlir::decisionforest::Schedule *schedule) override {
    m_func(*schedule);
  }
};

void OneTreeAtATimeSchedule(mlir::decisionforest::Schedule *schedule);
void OneTreeAtATimePipelinedSchedule(mlir::decisionforest::Schedule *schedule);
void OneTreeAtATimeUnrolledSchedule(mlir::decisionforest::Schedule *schedule);
void UnrollTreeLoop(decisionforest::Schedule *schedule);

template <int32_t BatchTileSize, int32_t TreeTileSize>
void TiledSchedule(mlir::decisionforest::Schedule *schedule) {
  auto &batchIndexVar = schedule->GetBatchIndex();
  auto &treeIndexVar = schedule->GetTreeIndex();
  auto &b0 = schedule->NewIndexVariable("b0");
  auto &b1 = schedule->NewIndexVariable("b1");
  auto &t0 = schedule->NewIndexVariable("t0");
  auto &t1 = schedule->NewIndexVariable("t1");

  schedule->Tile(batchIndexVar, b0, b1, BatchTileSize);
  schedule->Tile(treeIndexVar, t0, t1, TreeTileSize);

  schedule->Reorder(
      std::vector<mlir::decisionforest::IndexVariable *>{&t0, &b0, &t1, &b1});
}

template <int32_t TreeTileSize>
void TileTreeDimensionSchedule(mlir::decisionforest::Schedule *schedule) {
  auto &batchIndexVar = schedule->GetBatchIndex();
  auto &treeIndexVar = schedule->GetTreeIndex();
  auto &t0 = schedule->NewIndexVariable("t0");
  auto &t1 = schedule->NewIndexVariable("t1");

  schedule->Tile(treeIndexVar, t0, t1, TreeTileSize);
  // t1.Unroll();
  schedule->Reorder(std::vector<mlir::decisionforest::IndexVariable *>{
      &t0, &batchIndexVar, &t1});
}

template <int32_t split>
void SplitTreeDimensionSchedule(mlir::decisionforest::Schedule *schedule) {
  auto &t0 = schedule->NewIndexVariable("t0");
  auto &t1 = schedule->NewIndexVariable("t1");
  mlir::decisionforest::Schedule::IndexVariableMapType indexMap;
  schedule->Split(schedule->GetTreeIndex(), t0, t1, split, indexMap);
}

template <int32_t split>
void SwapAndSplitTreeDimensionSchedule(
    mlir::decisionforest::Schedule *schedule) {
  auto &batchIndexVar = schedule->GetBatchIndex();
  auto &treeIndexVar = schedule->GetTreeIndex();
  schedule->Reorder({&batchIndexVar, &treeIndexVar});
  auto &t0 = schedule->NewIndexVariable("t0");
  auto &t1 = schedule->NewIndexVariable("t1");
  mlir::decisionforest::Schedule::IndexVariableMapType indexMap;
  schedule->Split(treeIndexVar, t0, t1, split, indexMap);
}

} // namespace decisionforest
} // namespace mlir

#endif // _SCHEDULE_H_
