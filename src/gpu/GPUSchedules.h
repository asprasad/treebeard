#ifndef _GPUSCHEDULES_H_
#define _GPUSCHEDULES_H_

#include "schedule.h"

namespace mlir {
namespace decisionforest {

void GPUBasicSchedule(decisionforest::Schedule &schedule,
                      int32_t rowsPerThreadBlock);

void GPUBasicScheduleCacheRows(decisionforest::Schedule &schedule,
                               int32_t rowsPerThreadBlock);

void OneTreeAtATimeGPUSchedule(decisionforest::Schedule &schedule,
                               int32_t rowsPerThreadBlock,
                               int32_t rowsPerThread);

void OneTreeAtATimeCacheRowsGPUSchedule(decisionforest::Schedule &schedule,
                                        int32_t rowsPerThreadBlock,
                                        int32_t rowsPerThread);

void OneTreeAtATimeCacheTreeGPUSchedule(decisionforest::Schedule &schedule,
                                        int32_t rowsPerThreadBlock,
                                        int32_t rowsPerThread);

void OneTreeAtATimeCacheRowsAndTreesGPUSchedule(
    decisionforest::Schedule &schedule, int32_t rowsPerThreadBlock,
    int32_t rowsPerThread);

void SplitTreesAcrossThreadsGPUSchedule(decisionforest::Schedule &schedule,
                                        int32_t rowsPerThreadBlock,
                                        int32_t rowsPerThread,
                                        int32_t numParallelTreeGroups);

void SplitTreesAcrossThreadsGPUSchedule_Tiling(
    decisionforest::Schedule &schedule, int32_t rowsPerThreadBlock,
    int32_t rowsPerThread, int32_t numParallelTreeGroups);

void SplitTreesAcrossThreadsAndCacheRowsGPUSchedule(
    decisionforest::Schedule &schedule, int32_t rowsPerThreadBlock,
    int32_t rowsPerThread, int32_t numParallelTreeGroups);

void SplitTreesAcrossThreadsAndCacheRowsGPUSchedule_Reorg(
    decisionforest::Schedule &schedule, int32_t rowsPerThreadBlock,
    int32_t rowsPerThread, int32_t numParallelTreeGroups);

void SplitTreesAcrossThreadsCacheRowsAndInterleaveTreesGPUSchedule(
    decisionforest::Schedule &schedule, int32_t rowsPerThreadBlock,
    int32_t rowsPerThread, int32_t numParallelTreeGroups,
    int32_t numTreesToInterleave);

void SplitTreesAcrossThreadsCacheRowsAndInterleaveRowsGPUSchedule(
    decisionforest::Schedule &schedule, int32_t rowsPerThreadBlock,
    int32_t rowsPerThread, int32_t numParallelTreeGroups);

void SplitTreesAcrossThreadsAndCacheRowsGPUSchedule_Tiling(
    decisionforest::Schedule &schedule, int32_t rowsPerThreadBlock,
    int32_t rowsPerThread, int32_t numParallelTreeGroups);

void SplitTreesAcrossThreadsAndCacheTreesGPUSchedule(
    decisionforest::Schedule &schedule, int32_t rowsPerThreadBlock,
    int32_t rowsPerThread, int32_t numParallelTreeGroups);

void SplitTreesAcrossThreadsAndCacheTreesAndRowsGPUSchedule(
    decisionforest::Schedule &schedule, int32_t rowsPerThreadBlock,
    int32_t rowsPerThread, int32_t numParallelTreeGroups);

// ===---------------------------------------------------=== //
// Tahoe schedules
// ===---------------------------------------------------=== //

void TahoeSharedForestStrategy(decisionforest::Schedule &schedule,
                               int32_t rowsPerThreadBlock);

void TahoeSharedDataStrategy_Modified(decisionforest::Schedule &schedule,
                                      int32_t rowsPerThreadBlock);

void TahoeSharedDataStrategy(decisionforest::Schedule &schedule);

void TahoeSharedPartialForestStrategy(decisionforest::Schedule &schedule,
                                      int32_t treesPerThreadBlock,
                                      int32_t rowsPerThreadBlock);

void tahoeSharedDataStrategy_MultipleRowsPerBlock(
    decisionforest::Schedule &schedule, int32_t numRowsPerBlock);

// ===---------------------------------------------------=== //
// Iterative Strategies (tile rows and trees)
// ===---------------------------------------------------=== //

void iterativeCachedPartialForestStrategy(decisionforest::Schedule &schedule,
                                          int32_t treesPerIteration,
                                          int32_t rowsPerThreadBlock);

void iterativeCachedPartialForestStrategy_NoCache(
    decisionforest::Schedule &schedule, int32_t treesPerIteration,
    int32_t rowsPerThreadBlock);

void iterativeCachedPartialForestStrategy_NoCache_SharedReduce(
    decisionforest::Schedule &schedule, int32_t treesPerIteration,
    int32_t rowsPerThreadBlock);

void iterativeCachedPartialForestStrategy_NoCache_SpecializeTreeLoop(
    decisionforest::Schedule &schedule, int32_t treesPerIteration,
    int32_t rowsPerThreadBlock);

void CachePartialForestStrategy(decisionforest::Schedule &schedule,
                                int32_t treesToCache,
                                int32_t rowsPerThreadBlock);

} // end namespace decisionforest
} // end namespace mlir
#endif // _GPUSCHEDULES_H_