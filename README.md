# Treebeard 
An optimizing compiler for decision tree ensemble inference.

# Setting up Treebeard
1. **[Dependencies]** Install git, clang (v10 or newer), lld (v10 or newer), cmake (v3.16.3 or newer), ninja (v1.10.0 or newer), gcc (v9.3.0 or newer), g++ (v9.3.0 or newer), Anaconda.
    1. Run the following command on Ubuntu 20.04 (The packages default to the right versions on Ubuntu 20.04. Specific versions may need to be specified in other versions of Linux).
        ```bash
        sudo apt install git gcc g++ clang lld cmake ninja-build
        ```
    2. Install Anaconda as described here : https://docs.anaconda.com/anaconda/install/linux/.
2. **[Setup and Build]** In a bash terminal, run the following steps to build Treebeard.
    1. First, create a directory inside which all Treebeard code and binaries will be downloaded and built.
        ```bash
        mkdir treebeard-setup
        cd treebeard-setup
        ```
    2. Download scripts to setup LLVM and Treebeard and run them.
        ```bash
        wget https://raw.githubusercontent.com/asprasad/treebeard/master/scripts/setupLLVM.sh
        wget https://raw.githubusercontent.com/asprasad/treebeard/master/scripts/setupTreebeard.sh
        bash setupLLVM.sh
        bash setupTreebeard.sh
        ```
    3. We use <treebeard_home> to denote the directory treebeard-setup/llvm-project/mlir/examples/treebeard/. Run the following commands to setup the python environment (using conda).
        ```bash
        cd <treebeard_home>/scripts
        . createCondaEnvironment.sh
        . setupPythonEnv.h
        ```
    Note that the last two lines use ". script_name" in order to run the scripts in the current shell. A conda environment called treebeard should have been created and activated in the current shell.
3. **[Functionality Tests]** Use the following steps to run functionality tests. Run these steps in the same shell as above.
    1. To run python tests, execute the following commands.
        ```bash
        cd <treebeard_home>/test/python
        python run_python_tests.py
        ```
    2. To run basic sanity tests, run the following.
        ```bash
        cd <treebeard_home>/build/bin
        ./treebeard --sanityTests
        ```
    3. [Optional] To run all Treebeard tests, run the following (These tests will take more than an hour to run).
        ```bash
        cd <treebeard_home>/build/bin
        ./treebeard
        ```
4. **[Performance Tests]** To run the comparison with XGBoost and Treelite, run the following commands.
    ```bash
    cd <treebeard_home>/test/python
    python treebeard_serial_benchmarks.py
    python treebeard_parallel_benchmarks.py
    ```
