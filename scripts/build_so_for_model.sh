#!/bin/bash
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
LLVM_DIR=`dirname $SCRIPTPATH`
LLVM_DIR=`dirname $LLVM_DIR`
LLVM_DIR=`dirname $LLVM_DIR`
LLVM_DIR=`dirname $LLVM_DIR`
echo "Using LLVM rooted at : $LLVM_DIR"

LLVM_BUILD_DIR=${LLVM_BUILD_DIR:-'build_gpu'}
LLVM_BUILD_PATH="$LLVM_DIR/$LLVM_BUILD_DIR"
echo "Using LLVM build at : $LLVM_BUILD_PATH"

TREEBEARD_DIR=`dirname $SCRIPTPATH`
echo "TreeBeard directory : $TREEBEARD_DIR"

TREEBEARD_BUILD_DIR=${TREEBEARD_BUILD_DIR:-'build'}
TREEBEARD_BUILD_PATH="$TREEBEARD_DIR/$TREEBEARD_BUILD_DIR"
echo "Using TreeBeard build at : $TREEBEARD_BUILD_PATH"

while getopts "x:t:b:m:o:sin:" opt
do
   case "$opt" in
      t ) TILE_SIZE="$OPTARG" ;;
      b ) BATCH_SIZE="$OPTARG" ;;
      m ) MODEL="$OPTARG" ;;
      x ) MODEL_TYPE="-$OPTARG" ;;
      o ) OUTPUT_DIR="$OPTARG" ;;
      s ) SPARSE_FLAG="--sparse"
          BASE_NAME_SPARSE_EXT="_sparse"
          ;;
      i ) INVERT_FLAG="--invertLoops"
          BASE_NAME_INVERT_EXT="_invert"
          ;;
      n ) RET_TYPE_WIDTH="-returnIntBitWidth $OPTARG"
      # ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

echo "Output directory : $OUTPUT_DIR"

TREEBEARD_EXEC="$TREEBEARD_BUILD_PATH/bin/treebeard"
BASE_NAME="$temp_t${TILE_SIZE}_b${BATCH_SIZE}_f_i16${BASE_NAME_SPARSE_EXT}${BASE_NAME_INVERT_EXT}"
LLVM_IR_FILE="$OUTPUT_DIR/$BASE_NAME.ll"
ASM_FILE="$OUTPUT_DIR/$BASE_NAME.s"
SO_FILE="$OUTPUT_DIR/$BASE_NAME.so"
MODEL_GLOBALS_JSON="$SO_FILE.treebeard-globals.json"

DUMP_LLVM_CMD="$TREEBEARD_EXEC --dumpLLVM $SPARSE_FLAG $INVERT_FLAG $MODEL_TYPE $MODEL -globalValuesJSON $MODEL_GLOBALS_JSON -o $LLVM_IR_FILE -batchSize $BATCH_SIZE -tileSize $TILE_SIZE $RET_TYPE_WIDTH"
echo "$DUMP_LLVM_CMD"
$DUMP_LLVM_CMD

LLC="$LLVM_BUILD_PATH/bin/llc"
LLC_CMD="$LLC $LLVM_IR_FILE -O3 -march=x86-64 -mcpu=native -o $ASM_FILE --relocation-model=pic"
echo "$LLC_CMD"
$LLC_CMD

CLANG="$LLVM_BUILD_PATH/bin/clang"
CLANG_CMD="$CLANG $ASM_FILE -shared -o $SO_FILE"
$CLANG_CMD
