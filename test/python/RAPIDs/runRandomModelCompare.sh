# TODO store and download the random models!
conda activate silvanforge
python random_compare_runner.py --batch_size 512 --num_trials 3 2>random_models_512_stderr.txt | tee random_models_512.txt
python random_compare_runner.py --batch_size 4096 --num_trials 3 2>random_models_4k_stderr.txt | tee random_models_4k.txt
python plot_random_model_figures.py 