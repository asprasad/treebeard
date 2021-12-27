#!/bin/bash
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

TREEBEARD_DIR=`dirname $SCRIPTPATH`
echo "TreeBeard directory : $TREEBEARD_DIR"

TREEBEARD_BUILD_DIR=${TREEBEARD_BUILD_DIR:-'build'}
TREEBEARD_BUILD_PATH="$TREEBEARD_DIR/$TREEBEARD_BUILD_DIR"
echo "Using TreeBeard build at : $TREEBEARD_BUILD_PATH"

while getopts "t:b:m:o:s" opt
do
   case "$opt" in
      t ) TILE_SIZE="$OPTARG" ;;
      b ) BATCH_SIZE="$OPTARG" ;;
      m ) MODEL="$OPTARG" ;;
      o ) OUTPUT_DIR="$OPTARG" ;;
      s ) SPARSE_FLAG="--sparse"
          BASE_NAME_SPARSE_EXT="_sparse"
          ;;
      # ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

TREEBEARD_EXEC="$TREEBEARD_BUILD_PATH/bin/tree-heavy"
MODEL_JSON="$TREEBEARD_DIR/xgb_models/${MODEL}_xgb_model_save.json"
BASE_NAME="${MODEL}_t${TILE_SIZE}_b${BATCH_SIZE}_f_i16${BASE_NAME_SPARSE_EXT}"
SO_FILE="$OUTPUT_DIR/$BASE_NAME.so"
INPUT_FILE="$TREEBEARD_DIR/xgb_models/${MODEL}_xgb_model_save.json.csv"

# bin/tree-heavy --loadSO -json ~/mlir-build/llvm-project/mlir/examples/tree-heavy/xgb_models/abalone_xgb_model_save.json 
# -so ~/mlir-build/llvm-project/mlir/examples/tree-heavy/debug/bin/abalone_b4_t8_f.so -batchSize 4 -tileSize 8 -i ~/mlir-build/llvm-project/mlir/examples/tree-heavy/xgb_models/abalone_xgb_model_save.json.csv
RUN_CMD="$TREEBEARD_EXEC --loadSO $SPARSE_FLAG -json $MODEL_JSON -so $SO_FILE -batchSize $BATCH_SIZE -tileSize $TILE_SIZE -i $INPUT_FILE"
echo "$RUN_CMD"
$RUN_CMD
