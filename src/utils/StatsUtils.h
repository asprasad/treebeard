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
void ComputeForestProbabilityProfile(const std::string& modelJSONPath, const std::string& csvPath, const std::string& statsCSVPath, int32_t numRows);
void ComputeForestProbabilityProfileForXGBoostModel(const std::string& modelName, const std::string& csvPath, const std::string& statsCSVPath, int32_t numRows);

void ReadProbabilityProfile(mlir::decisionforest::DecisionForest<>& decisionForest, const std::string& statsCSVFile);
}
}

#endif // _STATSUTILS_H_