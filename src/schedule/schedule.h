#ifndef _SCHEDULE_H_
#define _SCHEDULE_H_

#include <string>
#include <vector>
#include <list>

namespace mlir
{
namespace decisionforest
{

class IndexVariable;
class IndexDerivationTreeVisitor;
class Schedule;

class IndexDerivationTreeNode {
public:
  virtual void Visit(IndexDerivationTreeVisitor& visitor) = 0;
  virtual void Validate() = 0;
};

class IndexModifier : public IndexDerivationTreeNode {
  // TODO Should this just be an IndexDerivationTreeNode? Maybe a parent could directly be an Index modifier
  IndexVariable *m_parent;
public:
  virtual ~IndexModifier() { }
  void Visit(IndexDerivationTreeVisitor& visitor) override = 0;
  void Validate() override = 0;
};

class TileIndexModifier : public IndexModifier {
  IndexVariable* m_sourceIndex;
  IndexVariable* m_outerIndex;
  IndexVariable* m_innerIndex;
  int32_t m_tileSize;
public:
  TileIndexModifier(IndexVariable& source, IndexVariable& outer, IndexVariable& inner, int32_t tileSize)
    : m_sourceIndex(&source), m_outerIndex(&outer), m_innerIndex(&inner), m_tileSize(tileSize) { }
  IndexVariable& OuterIndex() { return *m_outerIndex; }
  IndexVariable& InnerIndex() { return *m_innerIndex; }
  int32_t TileSize() { return m_tileSize; }
  IndexVariable& SourceIndex() { return *m_sourceIndex; }
  void Validate() override;
  void Visit(IndexDerivationTreeVisitor& visitor) override;
};

class SplitIndexModifier : public IndexModifier {
  IndexVariable* m_sourceIndex;
  IndexVariable* m_firstIndex;
  IndexVariable* m_secondIndex;
  int32_t m_splitIteration;
public:
  SplitIndexModifier(IndexVariable& source, IndexVariable& first, IndexVariable& second, int32_t splitIteration)
    : m_sourceIndex(&source), m_firstIndex(&first), m_secondIndex(&second), m_splitIteration(splitIteration) { }
  IndexVariable& SourceIndex() { return *m_sourceIndex; }
  IndexVariable& FirstIndex() { return *m_firstIndex; }
  IndexVariable& SecondIndex() { return *m_secondIndex; }
  int32_t SplitIteration() { return m_splitIteration; }
};

class IndexVariable : public IndexDerivationTreeNode {
  friend class Schedule;
public:
  enum class IndexVariableType { kBatch, kTree, kUnknown };
  struct IndexRange {
    int32_t m_start = -1;
    int32_t m_stop = -1;
    int32_t m_step = 0;
  };
protected:
  std::string m_name;
  IndexRange m_range;
  IndexVariable* m_containingLoop;
  std::vector<IndexVariable*> m_containedLoops;
  IndexVariableType m_type = IndexVariableType::kUnknown; 

  // Fields for the index modifier tree
  IndexModifier *m_parentModifier; // The modifier that resulted in this index variable
  IndexModifier *m_modifier; // Modifier that derives new index variables from this one

  bool m_pipelined = false;
  bool m_simdized = false;
  bool m_parallel = false;
  bool m_unrolled = false;

  // Index variables can only be constructed through the Schedule object
  IndexVariable(const std::string& name)
    :m_name(name), m_containingLoop(nullptr), m_parentModifier(nullptr), m_modifier(nullptr)
  { }

  // Index variables can't be copied
  IndexVariable(const IndexVariable& other) = delete;
public:
  IndexVariable* GetContainingLoop() const { return m_containingLoop; }
  const std::vector<IndexVariable*>& GetContainedLoops() const { return m_containedLoops; }
  IndexRange GetRange() const { return m_range; }
  void SetRange(IndexRange range) { m_range = range; }
  
  bool Pipelined() const { return m_pipelined; }
  bool Simdized() const { return m_simdized; }
  bool Parallel() const { return m_parallel; }
  bool Unroll() const { return m_unrolled; }

  void Visit(IndexDerivationTreeVisitor& visitor) override;
  void Validate() override;

  IndexModifier* GetParentModifier() const { return m_parentModifier; }
  IndexModifier* GetIndexModifier() const { return m_modifier; }
  IndexVariableType GetType() const { return m_type; }
};

class Schedule {
  std::list<IndexVariable*> m_indexVars;
  std::list<IndexModifier*> m_indexModifiers;

  IndexVariable m_treeIndex;
  IndexVariable m_batchIndex;
  IndexVariable m_rootIndex;

  int32_t m_batchSize;
  int32_t m_forestSize;

  Schedule(const Schedule&) = delete;
public:
  Schedule(int32_t batchSize, int32_t forestSize); 
  IndexVariable& NewIndexVariable(const std::string& name);
  // Loop Modifiers
  Schedule& Tile(IndexVariable& index, IndexVariable& outer, IndexVariable& inner, int32_t tileSize);
  Schedule& Reorder(const std::vector<IndexVariable*>& indices);
  Schedule& Split(IndexVariable& index, IndexVariable& first, IndexVariable& second, int32_t splitIteration);
  
  // Optimizations
  Schedule& Pipeline(IndexVariable& index);
  Schedule& Simdize(IndexVariable& index);
  Schedule& Parallel(IndexVariable& index);
  Schedule& Unroll(IndexVariable& index);

  const IndexVariable* GetRootIndex() const { return &m_rootIndex; }
  IndexVariable& GetBatchIndex() { return m_batchIndex; }
  IndexVariable& GetTreeIndex() { return m_treeIndex; }
  std::string PrintToString();

  void Finalize();
};

class IndexDerivationTreeVisitor {
public:
  virtual void VisitTileIndexModifier(TileIndexModifier& tileIndexModifier) = 0;
  virtual void VisitIndexVariable(IndexVariable& indexVar) = 0;
};

} // decisionforest
} // mlir

#endif // _SCHEDULE_H_
