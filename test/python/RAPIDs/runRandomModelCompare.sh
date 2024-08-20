while :; do
    RAPIDS_DIR=$(pwd)
    cd $RAPIDS_DIR/../../..
    TREEBEARD_DIR=$(pwd)
    cd $TREEBEARD_DIR/xgb_models/test
    MODEL_ZIP=$(pwd)/random-models.tar.gz
    if [ ! -e "$MODEL_ZIP" ]; then
        echo "Error: $MODEL_ZIP does not exist."
        break
    fi
    echo "Unzipping $MODEL_ZIP"
    tar -xzf random-models.tar.gz

    cd $RAPIDS_DIR

    conda activate silvanforge
    python -u random_compare_runner.py --batch_size 512 --num_trials 3 2>random_models_512_stderr.txt | tee random_models_512.txt
    python -u random_compare_runner.py --batch_size 4096 --num_trials 3 2>random_models_4k_stderr.txt | tee random_models_4k.txt
    python plot_random_model_figures.py 
    break
done