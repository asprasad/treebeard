#ifndef RUNTIME_H
#define RUNTIME_H

#include <cstdint>
#define TREEBEARD_RUNTIME_EXPORT __attribute__((visibility("default")))

#define COMPILER_OPTION_SETTER_DECLARATION(propName, propType) TREEBEARD_RUNTIME_EXPORT void Set_##propName(intptr_t options, propType val);

extern "C"
{
    TREEBEARD_RUNTIME_EXPORT intptr_t CreateInferenceRunnerForONNXModelInputs(
    int32_t numFeatures, int64_t inputAndThresholdSize, int64_t numNodes, const char *predTransform,
    double baseValue, const int64_t *treeIds, const int64_t *nodeIds,
    const int64_t *featureIds, void *thresholds, const int64_t *leftChildIds,
    const int64_t *rightChildIds, int64_t numberOfClasses,
    const int64_t *targetClassIds, const int64_t *targetClassTreeId,
    const int64_t *targetClassNodeId, const float *targetWeights,
    int64_t numWeights, int64_t batchSize, intptr_t options);
    
    TREEBEARD_RUNTIME_EXPORT void RunInference(intptr_t inferenceRunnerInt, void *inputs, void *results);

    TREEBEARD_RUNTIME_EXPORT void DeleteInferenceRunner(intptr_t inferenceRunnerInt);
    TREEBEARD_RUNTIME_EXPORT intptr_t CreateCompilerOptions();
    TREEBEARD_RUNTIME_EXPORT void DeleteCompilerOptions(intptr_t options);

    COMPILER_OPTION_SETTER_DECLARATION(batchSize, int32_t)
    COMPILER_OPTION_SETTER_DECLARATION(tileSize, int32_t)

    COMPILER_OPTION_SETTER_DECLARATION(thresholdTypeWidth, int32_t)
    COMPILER_OPTION_SETTER_DECLARATION(returnTypeWidth, int32_t)
    COMPILER_OPTION_SETTER_DECLARATION(returnTypeFloatType, int32_t)
    COMPILER_OPTION_SETTER_DECLARATION(featureIndexTypeWidth, int32_t)
    COMPILER_OPTION_SETTER_DECLARATION(nodeIndexTypeWidth, int32_t)
    COMPILER_OPTION_SETTER_DECLARATION(inputElementTypeWidth, int32_t)
    COMPILER_OPTION_SETTER_DECLARATION(tileShapeBitWidth, int32_t)
    COMPILER_OPTION_SETTER_DECLARATION(childIndexBitWidth, int32_t)
    COMPILER_OPTION_SETTER_DECLARATION(makeAllLeavesSameDepth, int32_t)
    COMPILER_OPTION_SETTER_DECLARATION(reorderTreesByDepth, int32_t)
    COMPILER_OPTION_SETTER_DECLARATION(statsProfileCSVPath,  const char*)
    COMPILER_OPTION_SETTER_DECLARATION(pipelineSize, int32_t)
    COMPILER_OPTION_SETTER_DECLARATION(numberOfCores, int32_t)


    TREEBEARD_RUNTIME_EXPORT void Set_tilingType(intptr_t options, int32_t val);
    TREEBEARD_RUNTIME_EXPORT void SetEnableSparseRepresentation(int32_t val);
    TREEBEARD_RUNTIME_EXPORT int32_t IsSparseRepresentationEnabled();
    TREEBEARD_RUNTIME_EXPORT void SetPeeledCodeGenForProbabilityBasedTiling(int32_t val);
    TREEBEARD_RUNTIME_EXPORT int32_t IsPeeledCodeGenForProbabilityBasedTilingEnabled();
}

#endif // RUNTIME_H