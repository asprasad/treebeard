#ifndef _GPUSCHEDULES_H_
#define _GPUSCHEDULES_H_

#include "schedule.h"

namespace mlir {
namespace decisionforest {

void GPUBasicSchedule(decisionforest::Schedule &schedule,
                      int32_t rowsPerThreadBlock);

void OneTreeAtATimeGPUSchedule(decisionforest::Schedule &schedule,
                               int32_t rowsPerThreadBlock,
                               int32_t rowsPerThread);

void SplitTreesAcrossThreadsGPUSchedule(decisionforest::Schedule &schedule,
                                        int32_t rowsPerThreadBlock,
                                        int32_t rowsPerThread,
                                        int32_t numParallelTreeGroups);

void TahoeSharedForestStrategy(decisionforest::Schedule &schedule,
                               int32_t rowsPerThreadBlock);

void TahoeSharedDataStrategy_Modified(decisionforest::Schedule &schedule,
                                      int32_t rowsPerThreadBlock);

void TahoeSharedDataStrategy(decisionforest::Schedule &schedule);

void tahoeSharedDataStrategy_MultipleRowsPerBlock(
    decisionforest::Schedule &schedule, int32_t numRowsPerBlock);

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

void TahoeSharedPartialForestStrategy(decisionforest::Schedule &schedule,
                                      int32_t treesPerThreadBlock,
                                      int32_t rowsPerThreadBlock);

void CachePartialForestStrategy(decisionforest::Schedule &schedule,
                                int32_t treesToCache,
                                int32_t rowsPerThreadBlock);

} // end namespace decisionforest
} // end namespace mlir
#endif // _GPUSCHEDULES_H_