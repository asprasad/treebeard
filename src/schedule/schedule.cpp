#include "schedule.h"
#include <algorithm>
#include <cassert>
#include <set>

namespace mlir {
namespace decisionforest {

Schedule::Schedule(int32_t batchSize, int32_t forestSize)
    : m_treeIndex("tree"), m_batchIndex("batch"), m_rootIndex("root"),
      m_batchSize(batchSize), m_forestSize(forestSize) {
  m_treeIndex.m_range = IndexVariable::IndexRange{0, forestSize, 1};
  m_treeIndex.m_type = IndexVariable::IndexVariableType::kTree;

  m_batchIndex.m_range = IndexVariable::IndexRange{0, m_batchSize, 1};
  m_batchIndex.m_type = IndexVariable::IndexVariableType::kBatch;

  m_rootIndex.m_containedLoops.push_back(&m_batchIndex);
  m_batchIndex.m_containingLoop = &m_rootIndex;
  m_batchIndex.m_containedLoops.push_back(&m_treeIndex);
  m_treeIndex.m_containingLoop = &m_batchIndex;
}

void Schedule::WriteIndexToDOTFile(IndexVariable *index, std::ofstream &fout) {
  auto parentIndex = index->m_containingLoop;
  fout << "\t\"node" << reinterpret_cast<int64_t>(index)
       << "\" [ label = \"Id:" << index->m_name
       << ", Start:" << index->m_range.m_start
       << ", Stop:" << index->m_range.m_stop
       << ", Step:" << index->m_range.m_step << "\"];\n";
  if (parentIndex != nullptr) {
    fout << "\t\"node" << reinterpret_cast<int64_t>(parentIndex)
         << "\" -> \"node" << reinterpret_cast<int64_t>(index) << "\"\n";
  }
  for (auto child : index->m_containedLoops) {
    WriteIndexToDOTFile(child, fout);
  }
}

void Schedule::WriteToDOTFile(const std::string &dotFile) {
  std::ofstream fout(dotFile);
  fout << "digraph {\n";
  WriteIndexToDOTFile(&m_rootIndex, fout);
  fout << "}\n";
}

IndexVariable &Schedule::NewIndexVariable(const std::string &name) {
  auto indexVarPtr = new IndexVariable(name);
  m_indexVars.push_back(indexVarPtr);
  return *indexVarPtr;
}

IndexVariable &Schedule::NewIndexVariable(const IndexVariable &indexVar) {
  auto indexVarPtr = new IndexVariable(indexVar);
  m_indexVars.push_back(indexVarPtr);
  return *indexVarPtr;
}

Schedule &Schedule::Tile(IndexVariable &index, IndexVariable &outer,
                         IndexVariable &inner, int32_t tileSize) {
  auto sourceIndexRange = index.GetRange();
  // Since we don't handle generation of code for partial tiles yet, asserting
  // that there should be no partial tiles
  assert(((sourceIndexRange.m_stop - sourceIndexRange.m_start) % tileSize) ==
         0);
  // Don't allow tiling of strided index variables (But shouldn't be a big
  // problem to support)
  assert(sourceIndexRange.m_step == 1);

  auto tileModifierPtr = new TileIndexModifier(index, outer, inner, tileSize);
  m_indexModifiers.push_back(tileModifierPtr);

  // There should be no derived indices of the current index
  assert(index.m_modifier == nullptr &&
         "This index already has an associated modifier!");
  index.m_modifier = tileModifierPtr;
  outer.m_parentModifier = tileModifierPtr;
  inner.m_parentModifier = tileModifierPtr;

  // The two new index variables have the same type as the source index variable
  outer.m_type = inner.m_type = index.m_type;

  // Make the inner loop contained in the outer loop
  outer.m_containedLoops = std::vector<IndexVariable *>{&inner};
  inner.m_containingLoop = &outer;

  // Replace the index in the loop nest tree with the new tiled variables
  outer.m_containingLoop = index.m_containingLoop;
  std::replace(index.m_containingLoop->m_containedLoops.begin(),
               index.m_containingLoop->m_containedLoops.end(), &index, &outer);

  inner.m_containedLoops = index.m_containedLoops;
  for (auto nestedLoop : inner.m_containedLoops)
    nestedLoop->m_containingLoop = &inner;

  // Blank out the links in the old index
  index.m_containedLoops.resize(0);
  index.m_containingLoop = nullptr;

  // Set the bounds on the derived index variables
  // TODO this needs to take into account the actual bounds of the index
  // variable when the original range was not a multiple of the step
  outer.SetRange(IndexVariable::IndexRange{sourceIndexRange.m_start,
                                           sourceIndexRange.m_stop,
                                           sourceIndexRange.m_step * tileSize});
  inner.SetRange(IndexVariable::IndexRange{
      0, tileSize * sourceIndexRange.m_step, sourceIndexRange.m_step});

  return *this;
}

void Schedule::DuplicateIndexVariables(
    IndexVariable &index,
    std::map<IndexVariable *, std::pair<IndexVariable *, IndexVariable *>>
        &indexMap) {
  assert(indexMap.find(&index) == indexMap.end());
  assert(index.m_modifier == nullptr);

  auto &indexCopy1 =
      NewIndexVariable(index); // TODO does this need to be unique?
  auto &indexCopy2 =
      NewIndexVariable(index); // TODO does this need to be unique?
  auto duplicateNode =
      new DuplicateIndexModifier(index, indexCopy1, indexCopy2);
  index.m_modifier = duplicateNode;
  indexCopy1.m_parentModifier = indexCopy2.m_parentModifier = duplicateNode;

  indexCopy1.m_containingLoop = indexCopy2.m_containingLoop = nullptr;
  indexCopy1.m_containedLoops.clear();
  indexCopy2.m_containedLoops.clear();

  auto iter = indexMap.find(index.m_containingLoop);
  // For the top level index being duplicated, we won't find
  // an entry in the map for the containing loop.
  if (iter != indexMap.end()) {
    // Here we are sure the containing loop was created by the Duplicate
    // So we can just add the new index vars to the containedLoops list.
    // TODO Does this maintain the order?
    indexCopy1.m_containingLoop = iter->second.first;
    indexCopy1.m_containingLoop->m_containedLoops.push_back(&indexCopy1);

    indexCopy2.m_containingLoop = iter->second.second;
    indexCopy2.m_containingLoop->m_containedLoops.push_back(&indexCopy2);
  }
  indexMap[&index] = std::make_pair(&indexCopy1, &indexCopy2);

  // Now recursively duplicate all the contained index variables
  for (auto containedIndex : index.m_containedLoops) {
    DuplicateIndexVariables(*containedIndex, indexMap);
  }
}

Schedule &Schedule::Split(
    IndexVariable &index, IndexVariable &first, IndexVariable &second,
    int32_t splitIteration,
    std::map<IndexVariable *, std::pair<IndexVariable *, IndexVariable *>>
        &indexMap) {
  // For simplicity, we enforce that index must currently be an inner most loop.
  // We will need to replicate all the nested loops when there are some.
  // However, the problem is, how do we communicate these newly generated copies
  // to the caller? Maybe a map?

  // Duplicate the source index variable into first and second
  auto duplicateNode = new DuplicateIndexModifier(index, first, second);
  assert(index.m_modifier == nullptr);
  index.m_modifier = duplicateNode;
  first.m_parentModifier = second.m_parentModifier = duplicateNode;
  first.m_modifier = second.m_modifier = nullptr;

  // Setup the limits
  first.m_range = {index.m_range.m_start, splitIteration, index.m_range.m_step};
  second.m_range = {splitIteration, index.m_range.m_stop, index.m_range.m_step};

  // Copy all fields from index into first and second
  first.m_type = second.m_type = index.m_type;
  first.m_pipelined = second.m_pipelined = index.m_pipelined;
  first.m_simdized = second.m_simdized = index.m_simdized;
  first.m_parallel = second.m_parallel = index.m_parallel;
  first.m_unrolled = second.m_unrolled = index.m_unrolled;
  first.m_peelWalk = second.m_peelWalk = index.m_peelWalk;
  first.m_iterationsToPeel = second.m_iterationsToPeel =
      index.m_iterationsToPeel;
  first.m_treeWalkUnrollFactor = second.m_treeWalkUnrollFactor =
      index.m_treeWalkUnrollFactor;

  // indexMap[&index] = std::make_pair(&first, &second);

  // Change the containment of index's containing loop
  if (index.m_containingLoop) {
    auto &containedLoops = index.m_containingLoop->m_containedLoops;
    auto iter = std::find(containedLoops.begin(), containedLoops.end(), &index);
    assert(iter != containedLoops.end());
    size_t indexPos = iter - containedLoops.begin();

    containedLoops.at(indexPos) = &first;
    containedLoops.insert(iter + 1, &second);

    first.m_containingLoop = second.m_containingLoop = index.m_containingLoop;

    index.m_containingLoop = nullptr;
  }

  // Duplicate all contained loops and add them to first and second's contained
  // loops
  for (auto containedLoop : index.m_containedLoops) {
    DuplicateIndexVariables(*containedLoop, indexMap);

    assert(indexMap.find(containedLoop) != indexMap.end());
    auto duplicateLoopPair = indexMap.find(containedLoop);
    first.m_containedLoops.push_back(duplicateLoopPair->second.first);
    duplicateLoopPair->second.first->m_containingLoop = &first;

    second.m_containedLoops.push_back(duplicateLoopPair->second.second);
    duplicateLoopPair->second.second->m_containingLoop = &second;
  }
  return *this;
}

void Schedule::DuplicateIndexVariables(
    IndexVariable &index, int32_t numCopies,
    Schedule::IterationSpecializationInfo &info) {

  assert(index.m_modifier == nullptr);
  auto *specializeNode = new SpecializeIndexModifier(index);
  index.m_modifier = specializeNode;

  for (auto iteration = 0; iteration < numCopies; ++iteration) {
    auto &indexCopy =
        NewIndexVariable(index); // TODO does this need to be unique?

    indexCopy.m_name = index.m_name + "_" + std::to_string(iteration);
    indexCopy.m_modifier = nullptr;

    indexCopy.m_parentModifier = specializeNode;
    specializeNode->AddSpecializedIndex(indexCopy);

    indexCopy.m_containingLoop = nullptr;
    indexCopy.m_containedLoops.clear();

    auto &indexMap = info.m_iterationMaps[iteration];
    auto iter = indexMap.find(index.m_containingLoop);
    // For the top level index being duplicated, we won't find
    // an entry in the map for the containing loop.
    if (iter != indexMap.end()) {
      // Here we are sure the containing loop was created by the Duplicate
      // So we can just add the new index vars to the containedLoops list.
      // TODO Does this maintain the order?
      indexCopy.m_containingLoop = iter->second;
      indexCopy.m_containingLoop->m_containedLoops.push_back(&indexCopy);
    }
    indexMap[&index] = &indexCopy;
  }
  // Now recursively duplicate all the contained index variables
  for (auto containedIndex : index.m_containedLoops) {
    DuplicateIndexVariables(*containedIndex, numCopies, info);
  }
}

Schedule &Schedule::SpecializeIterations(IndexVariable &index,
                                         IterationSpecializationInfo &info) {
  // Doesn't really change the existing index variable
  // Duplicate the children index variables into one new index var per iteration
  auto numIterations = (index.GetRange().m_stop - index.GetRange().m_start) /
                       index.GetRange().m_step;
  info.m_iterationMaps.resize(numIterations);
  for (auto containedLoop : index.m_containedLoops) {
    DuplicateIndexVariables(*containedLoop, numIterations, info);
  }
  std::vector<IndexVariable *> newContainedLoops;
  // Add one new index variable for each iteration
  // The original index variable contains the iteration index vars as children
  for (auto iteration = 0; iteration < numIterations; ++iteration) {
    auto &iterationIndexCopy = NewIndexVariable(index);
    newContainedLoops.push_back(&iterationIndexCopy);
    iterationIndexCopy.m_parentModifier = nullptr;
    iterationIndexCopy.m_containingLoop = &index;
    iterationIndexCopy.m_containedLoops.clear();
    for (auto containedLoop : index.m_containedLoops) {
      auto &containedLoopCopy = *info.m_iterationMaps[iteration][containedLoop];
      iterationIndexCopy.m_containedLoops.push_back(&containedLoopCopy);
      containedLoopCopy.m_containingLoop = &iterationIndexCopy;
    }
  }
  index.m_containedLoops = newContainedLoops;
  index.m_specializeIterations = true;
  return *this;
}

Schedule &Schedule::Reorder(const std::vector<IndexVariable *> &indices) {
  assert(!indices.empty());
  // First, make sure that the specified indices are "contiguous"
  std::set<IndexVariable *> indexSet(indices.begin(), indices.end());

  IndexVariable *outermostIndex = nullptr;
  IndexVariable *innermostIndex = nullptr;
  for (auto indexPtr : indices) {
    if (indexSet.find(indexPtr->m_containingLoop) == indexSet.end()) {
      assert(outermostIndex == nullptr &&
             "Only one index can have a containing loop outside the specified "
             "set of indices");
      outermostIndex = indexPtr;
    }
    if (indexPtr->m_containedLoops.empty() ||
        indexPtr->m_containedLoops.size() > 1 ||
        indexSet.find(indexPtr->m_containedLoops.front()) == indexSet.end()) {
      assert(innermostIndex == nullptr &&
             "Specified set of loops must be a perfectly nested set of loops");
      innermostIndex = indexPtr;
    }
  }
  assert(innermostIndex && outermostIndex);
  assert(outermostIndex->GetContainingLoop() != nullptr);

  auto outermostIndexContainingLoop = outermostIndex->m_containingLoop;
  auto innermostIndexContainedLoops = innermostIndex->m_containedLoops;

  std::replace(outermostIndexContainingLoop->m_containedLoops.begin(),
               outermostIndexContainingLoop->m_containedLoops.end(),
               outermostIndex, indices.front());
  indices.front()->m_containingLoop = outermostIndexContainingLoop;
  for (size_t i = 0; i < indices.size() - 1; ++i) {
    indices.at(i)->m_containedLoops = {indices.at(i + 1)};
    indices.at(i + 1)->m_containingLoop = indices.at(i);
  }
  indices.back()->m_containedLoops = innermostIndexContainedLoops;

  return *this;
}

Schedule &Schedule::Unroll(IndexVariable &index) {
  index.m_unrolled = true;
  return *this;
}

Schedule &Schedule::Parallel(IndexVariable &index) {
  index.m_parallel = true;
  return *this;
}

Schedule &Schedule::Cache(IndexVariable &index) {
  index.m_cache = true;
  return *this;
}

Schedule &Schedule::Pipeline(IndexVariable &index, int32_t stepSize) {
  assert(index.m_containedLoops.size() == 0 &&
         "Pipeline must be called on an innermost loop");
  // assert((index.m_range.m_stop - index.m_range.m_start) >= stepSize &&
  //        "Step size must be smaller than the range");
  auto iterCount =
      (index.m_range.m_stop - index.m_range.m_start) / (index.m_range.m_step);
  if (stepSize > iterCount)
    stepSize = iterCount;

  index.m_pipelined = true;
  index.m_unpipelinedStepSize = index.m_range.m_step;
  index.m_range.m_step = index.m_range.m_step * stepSize;
  return *this;
}

Schedule &Schedule::Simdize(IndexVariable &index) {
  assert(index.m_containedLoops.size() == 0 &&
         "Simdize must be called on an innermost loop");
  index.m_simdized = true;
  return *this;
}

bool Schedule::IsDefaultSchedule() {
  if (!(GetRootIndex()->GetContainedLoops().size() == 1 &&
        GetRootIndex()->GetContainedLoops().front() == &GetBatchIndex()))
    return false;
  if (!(GetBatchIndex().GetContainedLoops().size() == 1 &&
        GetBatchIndex().GetContainedLoops().front() == &GetTreeIndex()))
    return false;
  return true;
}

std::string Schedule::PrintToString() { return std::string(""); }

void Schedule::Finalize() {
  // TODO Go over all the nodes in the loop nest and check to see if any of them
  // need to be peeled to handle partial tiles
  m_batchIndex.Validate();
  m_treeIndex.Validate();
}

Schedule &Schedule::PeelWalk(IndexVariable &index, int32_t numberOfIterations) {
  index.m_peelWalk = true;
  index.m_iterationsToPeel = numberOfIterations;
  return *this;
}

Schedule &Schedule::AtomicReduce(IndexVariable &index) {
  assert(index.m_type == IndexVariable::IndexVariableType::kTree);
  index.m_atomicReduce = true;
  return *this;
}

Schedule &Schedule::VectorReduce(IndexVariable &index, int32_t width) {
  assert(index.m_type == IndexVariable::IndexVariableType::kTree);
  index.m_vectorReduceWidth = width;
  return *this;
}

Schedule &Schedule::SharedReduce(IndexVariable &index) {
  assert(index.m_type == IndexVariable::IndexVariableType::kTree);
  index.m_sharedReduce = true;
  return *this;
}

void IndexVariable::Visit(IndexDerivationTreeVisitor &visitor) {
  visitor.VisitIndexVariable(*this);
}

void IndexVariable::Validate() {
  if (m_modifier)
    m_modifier->Validate();
}

void TileIndexModifier::Visit(IndexDerivationTreeVisitor &visitor) {
  visitor.VisitTileIndexModifier(*this);
}

void TileIndexModifier::Validate() {
  // auto sourceIndexRange = this->m_sourceIndex->GetRange();
  // m_outerIndex->SetRange(IndexVariable::IndexRange{sourceIndexRange.m_start,
  // sourceIndexRange.m_stop, sourceIndexRange.m_step*m_tileSize});
  // // TODO this needs to take into account the actual bounds of the index
  // variable when the original range was not a multiple of the step
  // m_innerIndex->SetRange(IndexVariable::IndexRange{0,
  // m_tileSize*sourceIndexRange.m_step, sourceIndexRange.m_step});

  m_outerIndex->Validate();
  m_innerIndex->Validate();
}

void SplitIndexModifier::Validate() {
  m_firstIndex->Validate();
  m_secondIndex->Validate();
}

void SplitIndexModifier::Visit(IndexDerivationTreeVisitor &visitor) {
  visitor.VisitSplitIndexModifier(*this);
}

void DuplicateIndexModifier::Validate() {
  m_firstIndex->Validate();
  m_secondIndex->Validate();
}

void DuplicateIndexModifier::Visit(IndexDerivationTreeVisitor &visitor) {
  visitor.VisitDuplicateIndexModifier(*this);
}

void SpecializeIndexModifier::Validate() {
  for (auto &index : m_specializedIndices)
    index->Validate();
}

void SpecializeIndexModifier::Visit(IndexDerivationTreeVisitor &visitor) {
  visitor.VisitSpecializeIndexModifier(*this);
}

void OneTreeAtATimeSchedule(decisionforest::Schedule *schedule) {
  auto &batchIndexVar = schedule->GetBatchIndex();
  auto &treeIndexVar = schedule->GetTreeIndex();
  schedule->Reorder(std::vector<mlir::decisionforest::IndexVariable *>{
      &treeIndexVar, &batchIndexVar});
}

void OneTreeAtATimePipelinedSchedule(decisionforest::Schedule *schedule) {
  auto &batchIndexVar = schedule->GetBatchIndex();
  auto &treeIndexVar = schedule->GetTreeIndex();

  schedule->Reorder(std::vector<mlir::decisionforest::IndexVariable *>{
      &treeIndexVar, &batchIndexVar});
  schedule->Pipeline(batchIndexVar, 4);
}

void OneTreeAtATimeUnrolledSchedule(decisionforest::Schedule *schedule) {
  auto &batchIndexVar = schedule->GetBatchIndex();
  auto &treeIndexVar = schedule->GetTreeIndex();
  schedule->Reorder(std::vector<mlir::decisionforest::IndexVariable *>{
      &treeIndexVar, &batchIndexVar});
  schedule->Unroll(treeIndexVar);
}

void UnrollTreeLoop(decisionforest::Schedule *schedule) {
  auto &treeIndexVar = schedule->GetTreeIndex();
  schedule->Unroll(treeIndexVar);
}

} // namespace decisionforest
} // namespace mlir