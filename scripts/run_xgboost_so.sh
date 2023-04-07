#!/bin/bash
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

TREEBEARD_DIR=`dirname $SCRIPTPATH`
echo "TreeBeard directory : $TREEBEARD_DIR"

TREEBEARD_BUILD_DIR=${TREEBEARD_BUILD_DIR:-'build'}
TREEBEARD_BUILD_PATH="$TREEBEARD_DIR/$TREEBEARD_BUILD_DIR"
echo "Using TreeBeard build at : $TREEBEARD_BUILD_PATH"

while getopts "f:p:t:b:m:o:sin:" opt
do
   case "$opt" in
      t ) TILE_SIZE="$OPTARG" ;;
      p ) SO_FILE="$OPTARG" ;;
      f ) INPUT_FILE="$OPTARG" ;;
      b ) BATCH_SIZE="$OPTARG" ;;
      m ) MODEL="$OPTARG" ;;
      s ) SPARSE_FLAG="--sparse"
        BASE_NAME_SPARSE_EXT="_sparse"
          ;;
      n ) RET_TYPE_WIDTH="-returnIntBitWidth $OPTARG"
      # ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

TREEBEARD_EXEC="$TREEBEARD_BUILD_PATH/bin/treebeard"
#BASE_NAME="${MODEL}_t${TILE_SIZE}_b${BATCH_SIZE}_f_i16${BASE_NAME_SPARSE_EXT}${BASE_NAME_INVERT_EXT}"
#SO_FILE="$OUTPUT_DIR/$BASE_NAME.so"
#INPUT_FILE="$TREEBEARD_DIR/xgb_models/${MODEL}_xgb_model_save.json.csv"
MODEL_GLOBALS_JSON="$SO_FILE.treebeard-globals.json"

# bin/treebeard --loadSO so ~/mlir-build/llvm-project/mlir/examples/treebeard/debug/bin/abalone_b4_t8_f.so -batchSize 4 -tileSize 8 -i ~/mlir-build/llvm-project/mlir/examples/treebeard/xgb_models/abalone_xgb_model_save.json.csv
RUN_CMD="$TREEBEARD_EXEC --loadSO $SPARSE_FLAG -globalValuesJSON $MODEL_GLOBALS_JSON -so $SO_FILE -batchSize $BATCH_SIZE -tileSize $TILE_SIZE -i $INPUT_FILE $RET_TYPE_WIDTH -featIndexBitWidth 32"
echo "$RUN_CMD"
$RUN_CMD
