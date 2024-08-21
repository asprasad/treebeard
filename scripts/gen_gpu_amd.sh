#!/bin/bash
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
LLVM_DIR=`dirname $SCRIPTPATH`
LLVM_DIR=`dirname $LLVM_DIR`
LLVM_DIR=`dirname $LLVM_DIR`
LLVM_DIR=`dirname $LLVM_DIR`
echo "Using LLVM rooted at : $LLVM_DIR"

CMAKE="cmake"
MLIR_BUILD="build"
CONFIG="Release"

while getopts "d:m:c:" opt
do
   case "$opt" in
      d ) CMAKE="$OPTARG" ;;
      m ) MLIR_BUILD="$OPTARG" ;;
      c ) CONFIG="$OPTARG" ;;
   esac
done

# check if rocminfo is available.
if ! command -v rocminfo > /dev/null 2>&1; then
    echo "rocminfo not found. Please install ROCm and make sure it is in your PATH."
    exit 1
fi

CHIPSETS=`rocminfo | grep amdgcn-amd* | awk -F'--|:' '{print $3}'`
AMD_CHIPSET=$(echo "$CHIPSETS" | awk 'NR==1 {print $1}')

echo "Using cmake command : $CMAKE"
echo "Using MLIR_BUILD : $MLIR_BUILD"
echo "Using configuration : $CONFIG"
echo "Using AMD GPU architecture : $AMD_CHIPSET"

if command -v ninja > /dev/null 2>&1; then
    # run this from the build directory
    $CMAKE  -G Ninja .. -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
        -DMLIR_DIR=$LLVM_DIR/$MLIR_BUILD/lib/cmake/mlir \
        -DLLVM_BUILD_DIRECTORY=$LLVM_DIR/$MLIR_BUILD/ \
        -DCMAKE_BUILD_TYPE=$CONFIG \
        -DAMD_GPU_SUPPORT=ON \
        -DAMD_GPU_CHIPSET=$AMD_CHIPSET
    #      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
    #      -DLLVM_ENABLE_LLD=ON
    #      -DCMAKE_CXX_FLAGS="-std=c++17"
else
    # run this from the build directory
    $CMAKE  .. -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
        -DMLIR_DIR=$LLVM_DIR/$MLIR_BUILD/lib/cmake/mlir \
        -DLLVM_BUILD_DIRECTORY=$LLVM_DIR/$MLIR_BUILD/ \
        -DCMAKE_BUILD_TYPE=$CONFIG \
        -DAMD_GPU_SUPPORT=ON \
        -DAMD_GPU_CHIPSET=$AMD_CHIPSET
    #      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
    #      -DLLVM_ENABLE_LLD=ON
    #      -DCMAKE_CXX_FLAGS="-std=c++17"
fi