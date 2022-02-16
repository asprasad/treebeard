#ifndef _STATSUTILS_H_
#define _STATSUTILS_H_

#include <string>

namespace TreeBeard
{
namespace Profile
{

void ComputeForestInferenceStats(const std::string& modelJSONPath, const std::string& csvPath, int32_t numRows);
void ComputeForestInferenceStatsOnSampledTestInput(const std::string& model, int32_t numRows);
void ComputeForestInferenceStatsOnModel(const std::string& model, const std::string& csvPath, int32_t numRows);

}
}

#endif // _STATSUTILS_H_