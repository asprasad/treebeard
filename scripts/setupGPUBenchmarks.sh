CURRENT_DIR=$(pwd)

# Create a conda environment
ENV_NAME="silvanforge"
cd $CURRENT_DIR/../test/python/RAPIDs/
PACKAGE_FILE=packageList.txt
# Check if the environment already exists
if conda info --envs | grep -q "^$ENV_NAME\s"; then
    echo "Environment '$ENV_NAME' already exists."
else
    echo "Creating environment '$ENV_NAME' using package file '$PACKAGE_FILE'..."
    conda create --name "$ENV_NAME" --file $PACKAGE_FILE
fi
conda activate $ENV_NAME

# Get the Treebeard directory
cd $CURRENT_DIR/../
TREEBEARD_DIR=$(pwd)
echo "Treebeard directory: $TREEBEARD_DIR"

# Convert Treebeard models to Tahoe models
cd $TREEBEARD_DIR/../Tahoe/Tahoe_expts
TAHOE_EXPTS_DIR=$(pwd)
echo "Tahoe experiments directory: $TAHOE_EXPTS_DIR"

# If TAHOE_EXPTS_DIR does not exist, create it
if [ ! -d "$TAHOE_EXPTS_DIR/treebeard_models" ]; then
    mkdir $TAHOE_EXPTS_DIR/treebeard_models
fi
python convert_xgboost_to_tahoe.py --model_dir $TREEBEARD_DIR/xgb_models --output_dir $TAHOE_EXPTS_DIR/treebeard_models

cd $CURRENT_DIR