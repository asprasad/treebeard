#!/bin/bash
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
LLVM_DIR=`dirname $SCRIPTPATH`
LLVM_DIR=`dirname $LLVM_DIR`
LLVM_DIR=`dirname $LLVM_DIR`
LLVM_DIR=`dirname $LLVM_DIR`
echo "Using LLVM rooted at : $LLVM_DIR"

CMAKE=$1
CMAKE=${CMAKE:="cmake"}
echo "Using cmake command : $CMAKE"

MLIR_BUILD=$2
MLIR_BUILD=${MLIR_BUILD:="build"}
echo "Using MLIR_BUILD : $MLIR_BUILD"

# run this from the build directory
$CMAKE -G Ninja .. -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DMLIR_DIR=$LLVM_DIR/$MLIR_BUILD/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$LLVM_DIR/$MLIR_BUILD/llvm-lit -DCMAKE_BUILD_TYPE=Debug -DCMAKE_POLICY_DEFAULT_CMP0116=OLD
