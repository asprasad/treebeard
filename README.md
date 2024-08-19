# SilvanForge 
SilvanForge is an optimizing compiler for decision tree ensemble inference. Please feel free to file a github issue if you have trouble setting up SilvanForge.

# Setting up SilvanForge
1. **[Dependencies]** Install git, clang (v14 or newer), lld (v14 or newer), cmake (v3.22 or newer), ninja (v1.10.0 or newer), gcc (v9.4.0 or newer), g++ (v9.4.0 or newer), CUDA 11.8 (for NVIDIA GPUs) or ROCm 6.0.2 (AMD GPUs), Anaconda.
    1. Run the following command on Ubuntu 22.04 (The packages default to the right versions on Ubuntu 22.04. Specific versions may need to be specified in other versions of Linux).
        ```bash
        sudo apt install git gcc g++ clang lld cmake ninja-build
        ```
    2. Install Anaconda as described here : https://docs.anaconda.com/anaconda/install/linux/.
2. **[Setup and Build]** In a bash terminal, run the following steps to build SilvanForge.
    1. First, create a directory inside which all SilvanForge code and binaries will be downloaded and built.
        ```bash
        mkdir silvanforge-setup
        cd silvanforge-setup
        ```
    2. **[NVIDIA GPUs]** Download script to setup LLVM, SilvanForge and Tahoe and run it.
        ```bash
        wget https://raw.githubusercontent.com/asprasad/treebeard/master/scripts/setupSilvanForge.sh
        bash setupSilvanForge.sh
        ```
## Setting up Python Support
1. We use <silvanforge_home> to denote the directory silvanforge-setup/llvm-project/mlir/examples/treebeard/. Run the following commands to setup the python environment (using conda). The last two lines use ". script_name" in order to run the scripts in the current shell. A conda environment called "silvanforge" will be created and activated in the current shell. **Please note that these steps assume you are running in a bash shell.**
    ```bash
    cd <silvanforge_home>/scripts
    . setupGPUBenchmarks.sh
    . setupPythonEnv.sh
    ```
2.  The script setupGPUBenchmarks.sh should only be run once as it creates a new conda environment.
To setup SilvanForge's python support in a new shell (after the previous step has been performed once), run the following.
    ```bash
    . setupPythonEnv.sh
    conda activate silvanforge
    ```
## Functional Tests
Use the following steps to run functionality tests. Run these steps in the same shell as above. Running these will print a pass/fail result on the console. All of these tests should pass after the previous steps have been completed successfully. Please raise a github issue if any of these tests fail.
1. To run python tests, execute the following commands.
    ```bash
    cd <silvanforge_home>/test/python
    python run_python_tests.py
    ```
2. To run all SilvanForge tests, run the following (These tests will take about 15 minutes to run).
    ```bash
    cd <silvanforge_home>/build/bin
    ./treebeard
    ```

## Performance Tests
We consider figures 7 and 8 to be the key results of the paper _"SilvanForge: A Schedule Guided Retargetable Compiler for Decision Tree Inference"_. The following are the steps to replicate these experiments. 
1. **Figures 7a, 7b, 7c:** Activate the ```silvanforge``` conda environment and run the following commands.
    ```bash
    cd <silvanforge_home>/test/python
    python run_gpu_benchmarks.py
    ```
    This will take 3-4 hours to run. Three PNG files (```figure7a.png```, ```figure7b.png```, ```figure7c.png```) will be generated in the same directory.
2. **Figure 8:** Activate the ```silvanforge``` conda environment and run the following commands.
    ```bash
    cd <silvanforge_home>/test/python/RAPIDs
    [TODO]
    ```
## MLIR Version
The current version of Treebeard is tested with LLVM 16 (branch release/16.x of the LLVM github repo).

