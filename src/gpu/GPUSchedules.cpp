#include "GPUSchedules.h"

namespace mlir {
namespace decisionforest {

void GPUBasicSchedule(decisionforest::Schedule &schedule,
                      int32_t rowsPerThreadBlock) {
  auto &batchIndex = schedule.GetBatchIndex();
  auto &blockIndex = schedule.NewIndexVariable("gridX");
  auto &threadIndex = schedule.NewIndexVariable("blockX");

  schedule.Tile(batchIndex, blockIndex, threadIndex, rowsPerThreadBlock);
  blockIndex.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid,
                             decisionforest::IndexVariable::Dimension::X);
  threadIndex.SetGPUDimension(
      decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
      decisionforest::IndexVariable::Dimension::X);

  // auto& treeIndex = schedule->GetTreeIndex();
  // treeIndex.SetTreeWalkUnrollFactor(2);
}

// Divides the rows among threads (one thread processes one row fully)
// Processes one tree at a time -- all threads pick the same tree and
// walk that tree using all their rows.
void OneTreeAtATimeGPUSchedule(decisionforest::Schedule &schedule,
                               int32_t rowsPerThreadBlock,
                               int32_t rowsPerThread) {
  auto &batchIndex = schedule.GetBatchIndex();
  auto &treeIndex = schedule.GetTreeIndex();
  auto &threadBlockIndex = schedule.NewIndexVariable("b0");
  auto &threadIndexTemp = schedule.NewIndexVariable("b1_temp");
  auto &threadIndex = schedule.NewIndexVariable("b1_outer");
  auto &perThreadIndex = schedule.NewIndexVariable("b1_inner");

  schedule.Tile(batchIndex, threadBlockIndex, threadIndexTemp,
                rowsPerThreadBlock);
  schedule.Tile(threadIndexTemp, threadIndex, perThreadIndex, rowsPerThread);
  schedule.Reorder(
      {&threadBlockIndex, &threadIndex, &treeIndex, &perThreadIndex});
  threadBlockIndex.SetGPUDimension(
      decisionforest::IndexVariable::GPUConstruct::Grid,
      decisionforest::IndexVariable::Dimension::X);
  threadIndex.SetGPUDimension(
      decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
      decisionforest::IndexVariable::Dimension::X);
}

void OneTreeAtATimeCacheRowsGPUSchedule(decisionforest::Schedule &schedule,
                                        int32_t rowsPerThreadBlock,
                                        int32_t rowsPerThread) {
  auto &batchIndex = schedule.GetBatchIndex();
  auto &treeIndex = schedule.GetTreeIndex();
  auto &threadBlockIndex = schedule.NewIndexVariable("b0");
  auto &threadIndexTemp = schedule.NewIndexVariable("b1_temp");
  auto &threadIndex = schedule.NewIndexVariable("b1_outer");
  auto &perThreadIndex = schedule.NewIndexVariable("b1_inner");

  schedule.Tile(batchIndex, threadBlockIndex, threadIndexTemp,
                rowsPerThreadBlock);
  schedule.Tile(threadIndexTemp, threadIndex, perThreadIndex, rowsPerThread);
  schedule.Reorder(
      {&threadBlockIndex, &threadIndex, &treeIndex, &perThreadIndex});
  threadBlockIndex.SetGPUDimension(
      decisionforest::IndexVariable::GPUConstruct::Grid,
      decisionforest::IndexVariable::Dimension::X);
  threadIndex.SetGPUDimension(
      decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
      decisionforest::IndexVariable::Dimension::X);

  schedule.Cache(threadIndex);
}

void OneTreeAtATimeCacheTreeGPUSchedule(decisionforest::Schedule &schedule,
                                        int32_t rowsPerThreadBlock,
                                        int32_t rowsPerThread) {
  auto &batchIndex = schedule.GetBatchIndex();
  auto &treeIndex = schedule.GetTreeIndex();
  auto &threadBlockIndex = schedule.NewIndexVariable("b0");
  auto &threadIndexTemp = schedule.NewIndexVariable("b1_temp");
  auto &threadIndex = schedule.NewIndexVariable("b1_outer");
  auto &perThreadIndex = schedule.NewIndexVariable("b1_inner");

  schedule.Tile(batchIndex, threadBlockIndex, threadIndexTemp,
                rowsPerThreadBlock);
  schedule.Tile(threadIndexTemp, threadIndex, perThreadIndex, rowsPerThread);
  schedule.Reorder(
      {&threadBlockIndex, &threadIndex, &treeIndex, &perThreadIndex});
  threadBlockIndex.SetGPUDimension(
      decisionforest::IndexVariable::GPUConstruct::Grid,
      decisionforest::IndexVariable::Dimension::X);
  threadIndex.SetGPUDimension(
      decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
      decisionforest::IndexVariable::Dimension::X);

  schedule.Cache(treeIndex);
}

void SplitTreesAcrossThreadsGPUSchedule(decisionforest::Schedule &schedule,
                                        int32_t rowsPerThreadBlock,
                                        int32_t rowsPerThread,
                                        int32_t numParallelTreeGroups) {
  auto &batchIndex = schedule.GetBatchIndex();
  auto &treeIndex = schedule.GetTreeIndex();
  auto &threadBlockIndex = schedule.NewIndexVariable("b0");
  auto &threadIndexTemp = schedule.NewIndexVariable("b1_temp");
  auto &threadIndex = schedule.NewIndexVariable("b1_outer");
  auto &perThreadIndex = schedule.NewIndexVariable("b1_inner");

  schedule.Tile(batchIndex, threadBlockIndex, threadIndexTemp,
                rowsPerThreadBlock);
  schedule.Tile(threadIndexTemp, threadIndex, perThreadIndex, rowsPerThread);

  auto &treeIndexParallel = schedule.NewIndexVariable("t0_parallel");
  auto &treeIndexSerial = schedule.NewIndexVariable("t0_serial");

  schedule.Tile(treeIndex, treeIndexSerial, treeIndexParallel,
                numParallelTreeGroups);

  schedule.Reorder({&threadBlockIndex, &threadIndex, &treeIndexParallel,
                    &treeIndexSerial, &perThreadIndex});

  threadBlockIndex.SetGPUDimension(
      decisionforest::IndexVariable::GPUConstruct::Grid,
      decisionforest::IndexVariable::Dimension::X);
  threadIndex.SetGPUDimension(
      decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
      decisionforest::IndexVariable::Dimension::X);
  treeIndexParallel.SetGPUDimension(
      decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
      decisionforest::IndexVariable::Dimension::Y);
}

void SplitTreesAcrossThreadsAndCacheRowsGPUSchedule(
    decisionforest::Schedule &schedule, int32_t rowsPerThreadBlock,
    int32_t rowsPerThread, int32_t numParallelTreeGroups) {
  auto &batchIndex = schedule.GetBatchIndex();
  auto &treeIndex = schedule.GetTreeIndex();
  auto &threadBlockIndex = schedule.NewIndexVariable("b0");
  auto &threadIndexTemp = schedule.NewIndexVariable("b1_temp");
  auto &threadIndex = schedule.NewIndexVariable("b1_outer");
  auto &perThreadIndex = schedule.NewIndexVariable("b1_inner");

  schedule.Tile(batchIndex, threadBlockIndex, threadIndexTemp,
                rowsPerThreadBlock);
  schedule.Tile(threadIndexTemp, threadIndex, perThreadIndex, rowsPerThread);

  auto &treeIndexParallel = schedule.NewIndexVariable("t0_parallel");
  auto &treeIndexSerial = schedule.NewIndexVariable("t0_serial");

  schedule.Tile(treeIndex, treeIndexSerial, treeIndexParallel,
                numParallelTreeGroups);

  schedule.Reorder({&threadBlockIndex, &threadIndex, &treeIndexParallel,
                    &treeIndexSerial, &perThreadIndex});

  threadBlockIndex.SetGPUDimension(
      decisionforest::IndexVariable::GPUConstruct::Grid,
      decisionforest::IndexVariable::Dimension::X);
  threadIndex.SetGPUDimension(
      decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
      decisionforest::IndexVariable::Dimension::X);
  treeIndexParallel.SetGPUDimension(
      decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
      decisionforest::IndexVariable::Dimension::Y);

  schedule.Cache(threadIndex);
}

void TahoeSharedForestStrategy(decisionforest::Schedule &schedule,
                               int32_t rowsPerThreadBlock) {
  auto &batchIndex = schedule.GetBatchIndex();
  auto &treeIndex = schedule.GetTreeIndex();

  auto &b0 = schedule.NewIndexVariable("b0");
  auto &b1 = schedule.NewIndexVariable("b1");

  auto &t0 = schedule.NewIndexVariable("t0");
  auto &t1 = schedule.NewIndexVariable("t1");

  schedule.Tile(batchIndex, b0, b1, rowsPerThreadBlock);
  b0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid,
                     decisionforest::IndexVariable::Dimension::X);
  b1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
                     decisionforest::IndexVariable::Dimension::X);

  schedule.Tile(treeIndex, t0, t1, schedule.GetForestSize());
  schedule.Cache(t0);
}

// Here, each thread processes one input fully. However, in Tahoe, each thread
// processes one tree and one thread block process one row.
void TahoeSharedDataStrategy_Modified(decisionforest::Schedule &schedule,
                                      int32_t rowsPerThreadBlock) {
  auto &batchIndex = schedule.GetBatchIndex();

  auto &b0 = schedule.NewIndexVariable("b0");
  auto &b1 = schedule.NewIndexVariable("b1");

  schedule.Tile(batchIndex, b0, b1, rowsPerThreadBlock);
  b0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid,
                     decisionforest::IndexVariable::Dimension::X);
  b1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
                     decisionforest::IndexVariable::Dimension::X);

  schedule.Cache(b0);
}

void TahoeSharedDataStrategy(decisionforest::Schedule &schedule) {
  /*
    for b0 = 1:N_rows step 1 <Grid>
      for tree = 1:N_trees step 1 <Block.x>
        for b1 = 1:1 step 1 -- CacheRow
          WalkDecisionTree
  */

  auto &batchIndex = schedule.GetBatchIndex();
  auto &treeIndex = schedule.GetTreeIndex();

  auto &b0 = schedule.NewIndexVariable("b0");
  auto &b1 = schedule.NewIndexVariable("b1");

  schedule.Tile(batchIndex, b0, b1, 1);
  b0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid,
                     decisionforest::IndexVariable::Dimension::X);
  treeIndex.SetGPUDimension(
      decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
      decisionforest::IndexVariable::Dimension::X);
  schedule.Cache(b1);
  schedule.Reorder({&b0, &treeIndex, &b1});
}

void tahoeSharedDataStrategy_MultipleRowsPerBlock(
    decisionforest::Schedule &schedule, int32_t numRowsPerBlock) {
  /*
    for b0 = 1:N_rows step numRowsPerBlock <Grid.x>
      for b1 = 1:numRowsPerBlock step 1 <Block.x>
        for tree = 1:N_trees step 1 <Block.y>
          for b2 = 1:1 step 1 -- CacheRow
            WalkDecisionTree
  */

  auto &batchIndex = schedule.GetBatchIndex();
  auto &treeIndex = schedule.GetTreeIndex();

  auto &b0 = schedule.NewIndexVariable("b0");
  auto &b1_initial = schedule.NewIndexVariable("b1_initial");
  auto &b1 = schedule.NewIndexVariable("b1");
  auto &b2 = schedule.NewIndexVariable("b2");

  schedule.Tile(batchIndex, b0, b1_initial, numRowsPerBlock);
  schedule.Tile(b1_initial, b1, b2, 1);
  b0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid,
                     decisionforest::IndexVariable::Dimension::X);
  b1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
                     decisionforest::IndexVariable::Dimension::X);
  treeIndex.SetGPUDimension(
      decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
      decisionforest::IndexVariable::Dimension::Y);
  schedule.Reorder({&b0, &b1, &treeIndex, &b2});
  schedule.Cache(b0);
}

void TahoeSharedPartialForestStrategy(decisionforest::Schedule &schedule,
                                      int32_t treesPerThreadBlock,
                                      int32_t rowsPerThreadBlock) {
  auto &batchIndex = schedule.GetBatchIndex();
  auto &treeIndex = schedule.GetTreeIndex();

  auto &b0 = schedule.NewIndexVariable("b0");
  auto &b1 = schedule.NewIndexVariable("b1");

  auto &t0 = schedule.NewIndexVariable("t0");
  auto &t0Inner = schedule.NewIndexVariable("t0Inner");
  auto &t1 = schedule.NewIndexVariable("t1");
  auto &t2 = schedule.NewIndexVariable("t2");

  schedule.Tile(batchIndex, b0, b1, rowsPerThreadBlock);

  schedule.Tile(treeIndex, t0, t1, treesPerThreadBlock);
  schedule.Tile(t0Inner, t1, t2, treesPerThreadBlock);
  schedule.Cache(t1);
  schedule.Reorder(
      std::vector<decisionforest::IndexVariable *>{&b0, &t0, &b1, &t1, &t2});

  b0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid,
                     decisionforest::IndexVariable::Dimension::X);
  t0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid,
                     decisionforest::IndexVariable::Dimension::Y);
  b1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
                     decisionforest::IndexVariable::Dimension::X);
}

// Every thread block processes multiple rows and processes
// several trees at a time. It also caches the trees in shared memory.
void iterativeCachedPartialForestStrategy(decisionforest::Schedule &schedule,
                                          int32_t treesPerIteration,
                                          int32_t rowsPerThreadBlock) {
  /*
    for b0 = 1:N_rows step rowsPerTB <Grid.x>
      for t1 = 1:treesPerIteration step 1 <Block.y>
        for b1 = 1:rowsPerTB step 1 <Block.x>
          for t0 = 1:N_trees step treesPerIteration
            start = t0
            CacheTrees(start, treesPerIteration)
            WalkDecisionTree(t0+t1, b0+b1)
  */
  auto &batchIndex = schedule.GetBatchIndex();
  auto &treeIndex = schedule.GetTreeIndex();

  auto &b0 = schedule.NewIndexVariable("b0");
  auto &b1 = schedule.NewIndexVariable("b1");

  auto &t0 = schedule.NewIndexVariable("t0");
  auto &t1 = schedule.NewIndexVariable("t1");

  schedule.Tile(batchIndex, b0, b1, rowsPerThreadBlock);

  schedule.Tile(treeIndex, t0, t1, treesPerIteration);
  schedule.Cache(t0);
  schedule.Reorder(
      std::vector<decisionforest::IndexVariable *>{&b0, &b1, &t1, &t0});

  b0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid,
                     decisionforest::IndexVariable::Dimension::X);
  b1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
                     decisionforest::IndexVariable::Dimension::X);
  t1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
                     decisionforest::IndexVariable::Dimension::Y);
}

// Every thread block processes multiple rows and processes
// several trees at a time.
void iterativeCachedPartialForestStrategy_NoCache(
    decisionforest::Schedule &schedule, int32_t treesPerIteration,
    int32_t rowsPerThreadBlock) {
  /*
    for b0 = 1:N_rows step rowsPerTB <Grid.x>
      for t1 = 1:treesPerIteration step 1 <Block.y>
        for b1 = 1:rowsPerTB step 1 <Block.x>
          for t0 = 1:N_trees step treesPerIteration
            start = t0
            WalkDecisionTree
  */
  auto &batchIndex = schedule.GetBatchIndex();
  auto &treeIndex = schedule.GetTreeIndex();

  auto &b0 = schedule.NewIndexVariable("b0");
  auto &b1 = schedule.NewIndexVariable("b1");

  auto &t0 = schedule.NewIndexVariable("t0");
  auto &t1 = schedule.NewIndexVariable("t1");

  schedule.Tile(batchIndex, b0, b1, rowsPerThreadBlock);
  schedule.Tile(treeIndex, t0, t1, treesPerIteration);

  schedule.Reorder(
      std::vector<decisionforest::IndexVariable *>{&b0, &b1, &t1, &t0});

  b0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid,
                     decisionforest::IndexVariable::Dimension::X);
  b1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
                     decisionforest::IndexVariable::Dimension::X);
  t1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
                     decisionforest::IndexVariable::Dimension::Y);
}

// Every thread block processes multiple rows and processes
// several trees at a time.
// Reduction is performed in shared memory.
void iterativeCachedPartialForestStrategy_NoCache_SharedReduce(
    decisionforest::Schedule &schedule, int32_t treesPerIteration,
    int32_t rowsPerThreadBlock) {
  /*
    for b0 = 1:N_rows step rowsPerTB <Grid.x>
      for t1 = 1:treesPerIteration step 1 <Block.y>
        for b1 = 1:rowsPerTB step 1 <Block.x>
          for t0 = 1:N_trees step treesPerIteration
            start = t0
            WalkDecisionTree
  */
  auto &batchIndex = schedule.GetBatchIndex();
  auto &treeIndex = schedule.GetTreeIndex();

  auto &b0 = schedule.NewIndexVariable("b0");
  auto &b1 = schedule.NewIndexVariable("b1");

  auto &t0 = schedule.NewIndexVariable("t0");
  auto &t1 = schedule.NewIndexVariable("t1");

  schedule.Tile(batchIndex, b0, b1, rowsPerThreadBlock);
  schedule.Tile(treeIndex, t0, t1, treesPerIteration);

  schedule.Reorder(
      std::vector<decisionforest::IndexVariable *>{&b0, &b1, &t1, &t0});

  b0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid,
                     decisionforest::IndexVariable::Dimension::X);
  b1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
                     decisionforest::IndexVariable::Dimension::X);
  t1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
                     decisionforest::IndexVariable::Dimension::Y);
  schedule.SharedReduce(t1);
}

void iterativeCachedPartialForestStrategy_NoCache_SpecializeTreeLoop(
    decisionforest::Schedule &schedule, int32_t treesPerIteration,
    int32_t rowsPerThreadBlock) {
  /*
    for b0 = 1:N_rows step rowsPerTB <Grid.x>
      for t1 = 1:treesPerIteration step 1 <Block.y>
        for b1 = 1:rowsPerTB step 1 <Block.x>
          for t0 = 1:N_trees step treesPerIteration
            start = t0
            WalkDecisionTree
  */
  auto &batchIndex = schedule.GetBatchIndex();
  auto &treeIndex = schedule.GetTreeIndex();

  auto &b0 = schedule.NewIndexVariable("b0");
  auto &b1 = schedule.NewIndexVariable("b1");

  auto &t0 = schedule.NewIndexVariable("t0");
  auto &t1 = schedule.NewIndexVariable("t1");

  schedule.Tile(batchIndex, b0, b1, rowsPerThreadBlock);
  schedule.Tile(treeIndex, t0, t1, treesPerIteration);

  schedule.Reorder(
      std::vector<decisionforest::IndexVariable *>{&b0, &b1, &t1, &t0});

  b0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid,
                     decisionforest::IndexVariable::Dimension::X);
  b1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
                     decisionforest::IndexVariable::Dimension::X);
  t1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
                     decisionforest::IndexVariable::Dimension::Y);
  decisionforest::Schedule::IterationSpecializationInfo info;
  schedule.SpecializeIterations(t1, info);
}

void CachePartialForestStrategy(decisionforest::Schedule &schedule,
                                int32_t treesToCache,
                                int32_t rowsPerThreadBlock) {
  /*
    for b0 = 1:N_rows step rowsPerTB <Grid>
      for b1 = 1:rowsPerTB step 1 <Block.x>
        for t0 = 1:N_trees step NTreesToCache
          CacheTrees(t0, NTreesToCache)
          for t1 = 1:NTreesToCache step 1
            WalkDecisionTree
  */
  auto &batchIndex = schedule.GetBatchIndex();
  auto &treeIndex = schedule.GetTreeIndex();

  auto &b0 = schedule.NewIndexVariable("b0");
  auto &b1 = schedule.NewIndexVariable("b1");

  auto &t0 = schedule.NewIndexVariable("t0");
  auto &t1 = schedule.NewIndexVariable("t1");

  schedule.Tile(batchIndex, b0, b1, rowsPerThreadBlock);

  schedule.Tile(treeIndex, t0, t1, treesToCache);
  schedule.Cache(t0);
  schedule.Reorder(
      std::vector<decisionforest::IndexVariable *>{&b0, &b1, &t0, &t1});

  b0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid,
                     decisionforest::IndexVariable::Dimension::X);
  b1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
                     decisionforest::IndexVariable::Dimension::X);
}

} // namespace decisionforest
} // namespace mlir
