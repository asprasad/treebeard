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
CONFIG="Debug"

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

# run this from the build directory
$CMAKE -G Ninja .. -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DMLIR_DIR=$LLVM_DIR/$MLIR_BUILD/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$LLVM_DIR/$MLIR_BUILD/llvm-lit -DCMAKE_BUILD_TYPE=$CONFIG -DCMAKE_POLICY_DEFAULT_CMP0116=OLD
