#!/bin/bash
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
LLVM_DIR=`dirname $SCRIPTPATH`
LLVM_DIR=`dirname $LLVM_DIR`
LLVM_DIR=`dirname $LLVM_DIR`
LLVM_DIR=`dirname $LLVM_DIR`
echo "Using LLVM rooted at : $LLVM_DIR"

LLVM_BUILD_DIR=${LLVM_BUILD_DIR:-'build'}
LLVM_BUILD_PATH="$LLVM_DIR/$LLVM_BUILD_DIR"
echo "Using LLVM build at : $LLVM_BUILD_PATH"

TREEBEARD_DIR=`dirname $SCRIPTPATH`
echo "TreeBeard directory : $TREEBEARD_DIR"

TREEBEARD_BUILD_DIR=${TREEBEARD_BUILD_DIR:-'build'}
TREEBEARD_BUILD_PATH="$TREEBEARD_DIR/$TREEBEARD_BUILD_DIR"
echo "Using TreeBeard build at : $TREEBEARD_BUILD_PATH"

while getopts "t:b:m:o:si" opt
do
   case "$opt" in
      t ) TILE_SIZE="$OPTARG" ;;
      b ) BATCH_SIZE="$OPTARG" ;;
      m ) MODEL="$OPTARG" ;;
      o ) OUTPUT_DIR="$OPTARG" ;;
      s ) SPARSE_FLAG="--sparse"
          BASE_NAME_SPARSE_EXT="_sparse"
          ;;
      i ) INVERT_FLAG="--invertLoops"
          BASE_NAME_INVERT_EXT="_invert"
          ;;

      # ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# echo "Output directory : $OUTPUT_DIR"

TREEBEARD_EXEC="$TREEBEARD_BUILD_PATH/bin/tree-heavy"
MODEL_JSON="$TREEBEARD_DIR/xgb_models/${MODEL}_xgb_model_save.json"
BASE_NAME="${MODEL}_t${TILE_SIZE}_b${BATCH_SIZE}_f_i16${BASE_NAME_SPARSE_EXT}${BASE_NAME_INVERT_EXT}"
LLVM_IR_FILE="$OUTPUT_DIR/$BASE_NAME.ll"
ASM_FILE="$OUTPUT_DIR/$BASE_NAME.s"
SO_FILE="$OUTPUT_DIR/$BASE_NAME.so"
MODEL_GLOBALS_JSON="$SO_FILE.treebeard-globals.json"

# bin/tree-heavy --dumpLLVM -json ~/mlir-build/llvm-project/mlir/examples/tree-heavy/xgb_models/abalone_xgb_model_save.json 
# -o ~/mlir-build/llvm-project/mlir/examples/tree-heavy/debug/bin/abalone_b4_t8_f.ll -batchSize 4 -tileSize 8
DUMP_LLVM_CMD="$TREEBEARD_EXEC --dumpLLVM $SPARSE_FLAG $INVERT_FLAG -json $MODEL_JSON -globalValuesJSON $MODEL_GLOBALS_JSON -o $LLVM_IR_FILE -batchSize $BATCH_SIZE -tileSize $TILE_SIZE"
echo "$DUMP_LLVM_CMD"
$DUMP_LLVM_CMD

LLC="$LLVM_BUILD_PATH/bin/llc"
# ~/mlir-build/llvm-project/full_release/bin/llc abalone_b4_t8_f.ll -O3 -march=x86-64 -mcpu=rocketlake -o abalone_b4_t8_f.s --relocation-model=pic
# NOTE : The architecture and CPU parameters below are to match the JIT behavior on the machine "holmes"
LLC_CMD="$LLC $LLVM_IR_FILE -O3 -march=x86-64 -mcpu=rocketlake -o $ASM_FILE --relocation-model=pic"
echo "$LLC_CMD"
$LLC_CMD

CLANG="$LLVM_BUILD_PATH/bin/clang"
# ~/mlir-build/llvm-project/full_release/bin/clang abalone_b4_t8_f.s -shared -o abalone_b4_t8_f.so
CLANG_CMD="$CLANG $ASM_FILE -shared -o $SO_FILE"
$CLANG_CMD
