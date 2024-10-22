#!/bin/bash
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
LLVM_DIR=`dirname $SCRIPTPATH`
LLVM_DIR=`dirname $LLVM_DIR`
LLVM_DIR=`dirname $LLVM_DIR`
LLVM_DIR=`dirname $LLVM_DIR`
echo "Using LLVM rooted at : $LLVM_DIR"

if [ "$BUILD_MODE" = "debug" ]; then
    MLIR_BUILD="debug_build"
    CONFIG="Debug"
else
    MLIR_BUILD="build"
    CONFIG="Release"
fi

# Default cmake command
CMAKE="cmake"

# Process options
while getopts "d:m:c:" opt
do
   case "$opt" in
      d ) CMAKE="$OPTARG" ;;
      m ) MLIR_BUILD="$OPTARG" ;;
      c ) CONFIG="$OPTARG" ;;
   esac
done

echo "Using cmake command : $CMAKE"
echo "Using MLIR_BUILD : $MLIR_BUILD"
echo "Using configuration : $CONFIG"

# Run the cmake command from the build directory
$CMAKE -G Ninja .. -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
       -DMLIR_DIR=$LLVM_DIR/$MLIR_BUILD/lib/cmake/mlir \
       -DLLVM_BUILD_DIRECTORY=$LLVM_DIR/$MLIR_BUILD/ \
       -DCMAKE_BUILD_TYPE=$CONFIG \
       -DNV_GPU_SUPPORT=ON
#      -DAMD_GPU_SUPPORT=ON
#      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
#      -DLLVM_ENABLE_LLD=ON
#      -DCMAKE_CXX_FLAGS="-std=c++17"