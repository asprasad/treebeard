#ifndef _GPUSCHEDULES_H_
#define _GPUSCHEDULES_H_

#include <functional>
#include "schedule.h"

namespace mlir
{
namespace decisionforest
{

void GPUBasicSchedule(decisionforest::Schedule& schedule, int32_t gridXSize);
void TahoeSharedForestStrategy(decisionforest::Schedule& schedule, int32_t rowsPerThreadBlock);
void TahoeSharedDataStrategy_Modified(decisionforest::Schedule& schedule, int32_t rowsPerThreadBlock);
void TahoeSharedDataStrategy(decisionforest::Schedule& schedule);
void TahoeSharedPartialForestStrategy(decisionforest::Schedule& schedule,
                                      int32_t treesPerThreadBlock,
                                      int32_t rowsPerThreadBlock);

} // end namespace decisionforest
} // end namespace mlir
#endif // _GPUSCHEDULES_H_