#include <cassert>
#include <algorithm>
#include <set>
#include "schedule.h"

namespace mlir
{
namespace decisionforest
{

Schedule::Schedule(int32_t batchSize, int32_t forestSize)
 :m_treeIndex("tree"), m_batchIndex("batch"), m_rootIndex("root"), m_batchSize(batchSize), m_forestSize(forestSize)
{
  m_treeIndex.m_range = IndexVariable::IndexRange{0, forestSize, 1};
  m_treeIndex.m_type = IndexVariable::IndexVariableType::kTree;

  m_batchIndex.m_range = IndexVariable::IndexRange{0, m_batchSize, 1};
  m_batchIndex.m_type = IndexVariable::IndexVariableType::kBatch;
  
  m_rootIndex.m_containedLoops.push_back(&m_batchIndex);
  m_batchIndex.m_containingLoop = &m_rootIndex;
  m_batchIndex.m_containedLoops.push_back(&m_treeIndex);
  m_treeIndex.m_containingLoop = &m_batchIndex;
}

IndexVariable& Schedule::NewIndexVariable(const std::string& name) {
  auto indexVarPtr = new IndexVariable(name);
  m_indexVars.push_back(indexVarPtr);
  return *indexVarPtr;
}

Schedule& Schedule::Tile(IndexVariable& index, IndexVariable& outer, IndexVariable& inner, int32_t tileSize) {
  auto sourceIndexRange = index.GetRange();
  // Since we don't handle generation of code for partial tiles yet, asserting that there should be no partial tiles
  assert (((sourceIndexRange.m_stop - sourceIndexRange.m_start) % tileSize) == 0);
  // Don't allow tiling of strided index variables (But shouldn't be a big problem to support)
  assert (sourceIndexRange.m_step == 1);
  
  auto tileModifierPtr = new TileIndexModifier(index, outer, inner, tileSize);
  m_indexModifiers.push_back(tileModifierPtr);

  // There should be no derived indices of the current index
  assert (index.m_modifier == nullptr && "This index already has an associated modifier!");
  index.m_modifier = tileModifierPtr;
  outer.m_parentModifier = tileModifierPtr;
  inner.m_parentModifier = tileModifierPtr;

  // The two new index variables have the same type as the source index variable
  outer.m_type = inner.m_type = index.m_type;
  
  // Make the inner loop contained in the outer loop
  outer.m_containedLoops = std::vector<IndexVariable*>{ &inner };
  inner.m_containingLoop = &outer;

  // Replace the index in the loop nest tree with the new tiled variables
  outer.m_containingLoop = index.m_containingLoop;
  std::replace(index.m_containingLoop->m_containedLoops.begin(), index.m_containingLoop->m_containedLoops.end(), &index, &outer);

  inner.m_containedLoops = index.m_containedLoops;
  for (auto nestedLoop : inner.m_containedLoops)
    nestedLoop->m_containingLoop = &inner;
  
  // Blank out the links in the old index
  index.m_containedLoops.resize(0);
  index.m_containingLoop = nullptr;
  
  // Set the bounds on the derived index variables
  // TODO this needs to take into account the actual bounds of the index variable when the original range was not a multiple of the step
  outer.SetRange(IndexVariable::IndexRange{sourceIndexRange.m_start, sourceIndexRange.m_stop, sourceIndexRange.m_step*tileSize});
  inner.SetRange(IndexVariable::IndexRange{0, tileSize*sourceIndexRange.m_step, sourceIndexRange.m_step});

  return *this;
}

Schedule& Schedule::Split(IndexVariable& index, IndexVariable& first, IndexVariable& second, int32_t splitIteration) {
  // For simplicity, we enforce that index must currently be an inner most loop. We will need to replicate all the nested
  // loops when there are some. However, the problem is, how do we communicate these newly generated copies to the caller? Maybe a map?
  assert (index.m_containedLoops.size() == 0 && "Specified loop must be the inner most loop");
  first.m_containingLoop = index.m_containingLoop;
  second.m_containingLoop = index.m_containingLoop;

  return *this;
}

Schedule& Schedule::Reorder(const std::vector<IndexVariable*>& indices) {
  assert (!indices.empty());
  // First, make sure that the specified indices are "contiguous"
  std::set<IndexVariable*> indexSet(indices.begin(), indices.end());

  IndexVariable* outermostIndex=nullptr;
  IndexVariable* innermostIndex=nullptr;
  for (auto indexPtr : indices) {
    if (indexSet.find(indexPtr->m_containingLoop) == indexSet.end()) {
      assert (outermostIndex == nullptr && "Only one index can have a containing loop outside the specified set of indices");
      outermostIndex = indexPtr;
    }
    if (indexPtr->m_containedLoops.empty() || 
        indexPtr->m_containedLoops.size() > 1 || 
        indexSet.find(indexPtr->m_containedLoops.front())==indexSet.end()) {
      assert (innermostIndex == nullptr && "Specified set of loops must be a perfectly nested set of loops");
      innermostIndex = indexPtr;
    }
  }
  assert (innermostIndex && outermostIndex);
  assert (outermostIndex->GetContainingLoop() != nullptr);

  auto outermostIndexContainingLoop = outermostIndex->m_containingLoop;
  auto innermostIndexContainedLoops = innermostIndex->m_containedLoops;
  
  std::replace(outermostIndexContainingLoop->m_containedLoops.begin(), outermostIndexContainingLoop->m_containedLoops.end(), outermostIndex, indices.front());
  indices.front()->m_containingLoop = outermostIndexContainingLoop;
  for (size_t i=0 ; i<indices.size() - 1 ; ++i) {
    indices.at(i)->m_containedLoops = { indices.at(i+1) };
    indices.at(i+1)->m_containingLoop = indices.at(i);
  }
  indices.back()->m_containedLoops = innermostIndexContainedLoops;

  return *this;
}

Schedule& Schedule::Unroll(IndexVariable& index) {
  index.m_unrolled = true;
  return *this;
}

Schedule& Schedule::Parallel(IndexVariable& index) {
  index.m_parallel = true;
  return *this;
}

Schedule& Schedule::Pipeline(IndexVariable& index) {
  assert (index.m_containedLoops.size() == 0 && "Pipeline must be called on an innermost loop");
  index.m_pipelined = true;
  return *this;
}

Schedule& Schedule::Simdize(IndexVariable& index) {
  assert (index.m_containedLoops.size() == 0 && "Pipeline must be called on an innermost loop");
  index.m_pipelined = true;
  return *this;
}

std::string Schedule::PrintToString() {
  return std::string("");
}

void Schedule::Finalize() {
  // TODO Go over all the nodes in the loop nest and check to see if any of them need to be peeled to handle partial tiles
  m_batchIndex.Validate();
  m_treeIndex.Validate();
}

void IndexVariable::Visit(IndexDerivationTreeVisitor& visitor) {
  visitor.VisitIndexVariable(*this);
}

void IndexVariable::Validate() {
  if (m_modifier)
    m_modifier->Validate();
}

void TileIndexModifier::Visit(IndexDerivationTreeVisitor& visitor) {
  visitor.VisitTileIndexModifier(*this);
}

void TileIndexModifier::Validate() {
  // auto sourceIndexRange = this->m_sourceIndex->GetRange();
  // m_outerIndex->SetRange(IndexVariable::IndexRange{sourceIndexRange.m_start, sourceIndexRange.m_stop, sourceIndexRange.m_step*m_tileSize});
  // // TODO this needs to take into account the actual bounds of the index variable when the original range was not a multiple of the step
  // m_innerIndex->SetRange(IndexVariable::IndexRange{0, m_tileSize*sourceIndexRange.m_step, sourceIndexRange.m_step});

  m_outerIndex->Validate();
  m_innerIndex->Validate();
}

} // decisionforest
} // mlir