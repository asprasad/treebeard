#ifndef _GPUCOMPILEUTILS_H_
#define _GPUCOMPILEUTILS_H_

#include "Dialect.h"
#include "forestcreator.h"
#include "TreebeardContext.h"

namespace TreeBeard
{

// Construct a GPU module from a TreebeardContext
mlir::ModuleOp ConstructGPUModuleFromTreebeardContext(TreebeardContext& tbContext);

} // TreeBeard

#endif // _GPUCOMPILEUTILS_H_