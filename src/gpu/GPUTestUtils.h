#ifndef GPU_TEST_UTILS_H
#define GPU_TEST_UTILS_H

#include "dlfcn.h"

#include "ForestTestUtils.h"
#include "GPUExecutionHelper.h"
#include "GPUModelSerializers.h"

using namespace mlir;
namespace TreeBeard {
namespace test {
class GPUInferenceRunnerForTest : public InferenceRunnerForTestTemplate<
                                      decisionforest::GPUInferenceRunner> {
public:
  GPUInferenceRunnerForTest(
      std::shared_ptr<decisionforest::IModelSerializer> serializer,
      mlir::ModuleOp module, int32_t tileSize, int32_t thresholdSize,
      int32_t featureIndexSize)
      : InferenceRunnerForTestTemplate<decisionforest::GPUInferenceRunner>(
            serializer, module, tileSize, thresholdSize, featureIndexSize) {
    if (decisionforest::measureGpuKernelTime) {
      typedef void (*ResetFunc_t)();
      auto resetFunc =
          reinterpret_cast<ResetFunc_t>(dlsym(NULL, "resetTotalKernelTime"));
      assert(resetFunc);
      resetFunc();
    }
  }

  //   using InferenceRunnerForTestTemplate<
  //       decisionforest::GPUInferenceRunner>::InferenceRunnerForTestTemplate;

  inline decisionforest::ModelMemrefType GetModelMemref() {
    return reinterpret_cast<decisionforest::GPUArraySparseSerializerBase *>(
               m_serializer.get())
        ->GetModelMemref();
  }

  inline int64_t GetKernelExecutionTime() {
    if (!decisionforest::measureGpuKernelTime)
      return 0;
    typedef int64_t (*GetKernelTimeFunc_t)();
    auto getKernelTimeFunc = reinterpret_cast<GetKernelTimeFunc_t>(
        dlsym(NULL, "getTotalKernelTime"));
    assert(getKernelTimeFunc);
    return getKernelTimeFunc();
  }
};

struct NoOpDeleter {
  void operator()(ForestCreator *ptr) const {
    // Do nothing
  }
};

} // namespace test
} // namespace TreeBeard

#endif // GPU_TEST_UTILS_H
