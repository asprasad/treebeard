#ifndef _TYPEDEFINITIONS_H_
#define _TYPEDEFINITIONS_H_

#include <cstdint>
namespace mlir 
{
namespace decisionforest
{

template<typename T, int32_t Rank>
struct Memref {
  T *bufferPtr;
  T *alignedPtr;
  int64_t offset;
  int64_t lengths[Rank];
  int64_t strides[Rank];
};

}
}
#endif // _TYPEDEFINITIONS_H_