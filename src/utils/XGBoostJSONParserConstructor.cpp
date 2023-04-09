#include "xgboostparser.h"
#include "TreebeardContext.h"

using namespace TreeBeard;

namespace 
{

template<typename ThresholdType, typename ReturnType, typename FeatureIndexType, typename NodeIndexType>
std::shared_ptr<ForestCreator> SpecializeInputElementType(mlir::MLIRContext& context, TreebeardContext& tbContext) {
  auto& options = tbContext.options;
  auto& modelJsonPath = tbContext.modelPath;

  if (options.inputElementTypeWidth == 32) {
    auto parser = std::make_shared<XGBoostJSONParser<ThresholdType, 
                                                      ReturnType,
                                                      FeatureIndexType,
                                                      NodeIndexType,
                                                      float>>(context, 
                                                            modelJsonPath,
                                                            tbContext.serializer,
                                                            options.statsProfileCSVPath,
                                                            options.batchSize);
    return parser;                                                            
  }
  else if (options.inputElementTypeWidth == 64) {
    auto parser = std::make_shared<XGBoostJSONParser<ThresholdType, 
                                                      ReturnType,
                                                      FeatureIndexType,
                                                      NodeIndexType,
                                                      float>>(context, 
                                                            modelJsonPath,
                                                            tbContext.serializer,
                                                            options.statsProfileCSVPath,
                                                            options.batchSize);
    return parser;                                                            
  }
  else {
    assert (false && "Unknown input element type");
  }
  return nullptr;  
}

template<typename ThresholdType, typename ReturnType, typename FeatureIndexType>
std::shared_ptr<ForestCreator> SpecializeNodeIndexType(mlir::MLIRContext& context, TreebeardContext& tbContext) {
  auto& options = tbContext.options;
  if (options.nodeIndexTypeWidth == 8) {
    return SpecializeInputElementType<ThresholdType, ReturnType, FeatureIndexType, int8_t>(context, tbContext);
  } 
  else if (options.nodeIndexTypeWidth == 16) {
    return SpecializeInputElementType<ThresholdType, ReturnType, FeatureIndexType, int16_t>(context, tbContext);
  } 
  else if (options.nodeIndexTypeWidth == 32) {
    return SpecializeInputElementType<ThresholdType, ReturnType, FeatureIndexType, int32_t>(context, tbContext);
  }
  else if (options.nodeIndexTypeWidth == 64) {
    return SpecializeInputElementType<ThresholdType, ReturnType, FeatureIndexType, int64_t>(context, tbContext);
  } 
  else {
    assert (false && "Unknown feature index type");
  }
  return nullptr;
}

template<typename ThresholdType, typename ReturnType>
std::shared_ptr<ForestCreator> SpecializeFeatureIndexType(mlir::MLIRContext& context, TreebeardContext& tbContext) {
  auto& options = tbContext.options;
  if (options.featureIndexTypeWidth == 8) {
    return SpecializeNodeIndexType<ThresholdType, ReturnType, int8_t>(context, tbContext);
  } 
  else if (options.featureIndexTypeWidth == 16) {
    return SpecializeNodeIndexType<ThresholdType, ReturnType, int16_t>(context, tbContext);
  } 
  else if (options.featureIndexTypeWidth == 32) {
    return SpecializeNodeIndexType<ThresholdType, ReturnType, int32_t>(context, tbContext);
  }
  else if (options.featureIndexTypeWidth == 64) {
    return SpecializeNodeIndexType<ThresholdType, ReturnType, int64_t>(context, tbContext);
  } 
  else {
    assert (false && "Unknown feature index type");
  }
  return nullptr;
}

template<typename ThresholdType>
std::shared_ptr<ForestCreator> SpecializeReturnType(mlir::MLIRContext& context, TreebeardContext& tbContext) {
  auto& options = tbContext.options;
  if (options.returnTypeFloatType) {
    if (options.returnTypeWidth == 32) {
      return SpecializeFeatureIndexType<ThresholdType, float>(context, tbContext);
    }
    else if (options.returnTypeWidth == 64) {
      return SpecializeFeatureIndexType<ThresholdType, double>(context, tbContext);
    } 
    else {
      assert (false && "Unknown return type");
    }
  }
  else {
    if (options.returnTypeWidth == 8) {
      return SpecializeFeatureIndexType<ThresholdType, int8_t>(context, tbContext);
    }
    else {
      assert (false && "Unknown return type");
    }
  }
  return nullptr;
}

} // anonymous namespace

namespace TreeBeard
{

std::shared_ptr<ForestCreator> ConstructXGBoostJSONParser(mlir::MLIRContext& context, TreebeardContext& tbContext) {
  auto& options = tbContext.options;
  if (options.thresholdTypeWidth == 32) {
    return SpecializeReturnType<float>(context, tbContext);
  }
  else if (options.thresholdTypeWidth == 64) {
    return SpecializeReturnType<double>(context, tbContext);
  }
  else {
    assert (false && "Unknown threshold type");
  }
  return nullptr;
}

}