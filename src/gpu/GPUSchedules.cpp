#include "GPUSchedules.h"

namespace mlir
{
namespace decisionforest
{

void GPUBasicSchedule(decisionforest::Schedule& schedule, int32_t gridXSize) {
  auto& batchIndex = schedule.GetBatchIndex();
  auto& blockIndex = schedule.NewIndexVariable("gridX");
  auto& threadIndex = schedule.NewIndexVariable("blockX");
  
  schedule.Tile(batchIndex, blockIndex, threadIndex, gridXSize);
  blockIndex.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid, decisionforest::IndexVariable::Dimension::X);
  threadIndex.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock, decisionforest::IndexVariable::Dimension::X);
  
  // auto& treeIndex = schedule->GetTreeIndex();
  // treeIndex.SetTreeWalkUnrollFactor(2);
}

void TahoeSharedForestStrategy(decisionforest::Schedule& schedule, int32_t rowsPerThreadBlock) {
  auto& batchIndex = schedule.GetBatchIndex();
  auto& treeIndex = schedule.GetTreeIndex();
  
  auto& b0 = schedule.NewIndexVariable("b0");
  auto& b1 = schedule.NewIndexVariable("b1");

  auto& t0 = schedule.NewIndexVariable("t0");
  auto& t1 = schedule.NewIndexVariable("t1");
  
  schedule.Tile(batchIndex, b0, b1, rowsPerThreadBlock);
  b0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid, decisionforest::IndexVariable::Dimension::X);
  b1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock, decisionforest::IndexVariable::Dimension::X);
  
  schedule.Tile(treeIndex, t0, t1, schedule.GetForestSize());
  schedule.Cache(t0);
}

// Here, each thread processes one input fully. However, in Tahoe, each thread processes one tree
// and one thread block process one row.
void TahoeSharedDataStrategy_Modified(decisionforest::Schedule& schedule, int32_t rowsPerThreadBlock) {
  auto& batchIndex = schedule.GetBatchIndex();

  auto& b0 = schedule.NewIndexVariable("b0");
  auto& b1 = schedule.NewIndexVariable("b1");

  schedule.Tile(batchIndex, b0, b1, rowsPerThreadBlock);
  b0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid, decisionforest::IndexVariable::Dimension::X);
  b1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock, decisionforest::IndexVariable::Dimension::X);
  
  schedule.Cache(b0);
}

void TahoeSharedDataStrategy(decisionforest::Schedule& schedule) {
  /*
    for b0 = 1:N_rows step 1 <Grid>
      for tree = 1:N_trees step 1 <Block.x>
        for b1 = 1:1 step 1 -- CacheRow
          WalkDecisionTree
  */  
  
  auto& batchIndex = schedule.GetBatchIndex();
  auto& treeIndex = schedule.GetTreeIndex();
  
  auto& b0 = schedule.NewIndexVariable("b0");
  auto& b1 = schedule.NewIndexVariable("b1");

  schedule.Tile(batchIndex, b0, b1, 1);
  b0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid, decisionforest::IndexVariable::Dimension::X);
  treeIndex.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock, decisionforest::IndexVariable::Dimension::X);
  
  schedule.Cache(b1);
}

void TahoeSharedPartialForestStrategy(decisionforest::Schedule& schedule,
                                      int32_t treesPerThreadBlock,
                                      int32_t rowsPerThreadBlock) {
  auto& batchIndex = schedule.GetBatchIndex();
  auto& treeIndex = schedule.GetTreeIndex();
  
  auto& b0 = schedule.NewIndexVariable("b0");
  auto& b1 = schedule.NewIndexVariable("b1");

  auto& t0 = schedule.NewIndexVariable("t0");
  auto& t0Inner = schedule.NewIndexVariable("t0Inner");
  auto& t1 = schedule.NewIndexVariable("t1");
  auto& t2 = schedule.NewIndexVariable("t2");
  
  schedule.Tile(batchIndex, b0, b1, rowsPerThreadBlock);
  
  schedule.Tile(treeIndex, t0, t1, treesPerThreadBlock);
  schedule.Tile(t0Inner, t1, t2, treesPerThreadBlock);
  schedule.Cache(t1);
  schedule.Reorder(std::vector<decisionforest::IndexVariable*>{&b0, &t0, &b1, &t1, &t2});

  b0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid, decisionforest::IndexVariable::Dimension::X);
  t0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid, decisionforest::IndexVariable::Dimension::Y);
  b1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock, decisionforest::IndexVariable::Dimension::X);
}

void CachePartialForestStrategy(decisionforest::Schedule& schedule,
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
  auto& batchIndex = schedule.GetBatchIndex();
  auto& treeIndex = schedule.GetTreeIndex();
  
  auto& b0 = schedule.NewIndexVariable("b0");
  auto& b1 = schedule.NewIndexVariable("b1");

  auto& t0 = schedule.NewIndexVariable("t0");
  auto& t1 = schedule.NewIndexVariable("t1");
  
  schedule.Tile(batchIndex, b0, b1, rowsPerThreadBlock);
  
  schedule.Tile(treeIndex, t0, t1, treesToCache);
  schedule.Cache(t0);
  schedule.Reorder(std::vector<decisionforest::IndexVariable*>{&b0, &b1, &t0, &t1});

  b0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid, decisionforest::IndexVariable::Dimension::X);
  b1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock, decisionforest::IndexVariable::Dimension::X);
}



} // namespace decisionforest
} // namespace mlir
