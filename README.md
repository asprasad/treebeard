# Treebeard 
Treebeard is an optimizing compiler for decision tree ensemble inference. Please feel free to file a github issue if you have trouble setting up Treebeard.

# Setting up Treebeard
1. **[Dependencies]** Install git, clang (v15 or newer), lld (v15 or newer), cmake (v3.20 or newer), ninja (v1.10.0 or newer), gcc (v9.3.0 or newer), g++ (v9.3.0 or newer), Anaconda.
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
## Setting up Python Support

1. We use <treebeard_home> to denote the directory treebeard-setup/llvm-project/mlir/examples/treebeard/. Run the following commands to setup the python environment (using conda). The last two lines use ". script_name" in order to run the scripts in the current shell. A conda environment called "treebeard" will be created and activated in the current shell.
    ```bash
    cd <treebeard_home>/scripts
    . createCondaEnvironment.sh
    . setupPythonEnv.sh
    ```
2.  The script createCondaEnvironment.sh should only be run once as it creates a new conda environment.
To setup Treebeard's python support in a new shell (after the previous step has been performed once), run the following.
    ```bash
    . setupPythonEnv.sh
    conda activate treebeard
    ```
## Running Tests

1. **[Functionality Tests]** Use the following steps to run functionality tests. Run these steps in the same shell as above. Running these will print a pass/fail result on the console. All of these tests should pass after the previous steps have been completed successfully. Please raise a github issue if any of these tests fail.
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
2. **[Performance Tests]** To run the comparison with XGBoost and Treelite, run the following commands. This will print a table in CSV format to stdout containing the per sample inference times (in seconds) for Treebeard, XGBoost and Treelite and the speedup of Treebeard relative to XGBoost and Treelite. 
    ```bash
    cd <treebeard_home>/test/python
    python treebeard_serial_benchmarks.py
    python treebeard_parallel_benchmarks.py
    ```
3. **[Customizing Performance Tests]** The benchmark scripts above are hard-coded to use a configuration
that is tuned for the Intel processor on which we ran our experiments (Intel Core i9-11900K). Running the benchmark
scripts with the "--explore" switch will explore a few other predefined configurations 
and find the best one among these for the machine on which code is being executed. However, this will mean 
that the python script will take significantly longer to complete.

# Customizing the build
1. Setup a build of [MLIR](https://mlir.llvm.org/getting_started/).
```bash    
    git clone https://github.com/llvm/llvm-project.git
    mkdir llvm-project/build.release
    cd llvm-project/build.release
    git checkout release/16.x
    cmake -G Ninja ../llvm \
        -DLLVM_ENABLE_PROJECTS="llvm;clang;lld;mlir;openmp" \
        -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
        -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
        -DLLVM_ENABLE_LLD=ON
    cmake --build .
    cmake --build . --target check-mlir
```
2. Clone this repository into <path-to-llvm-repo>/llvm-project/mlir/examples/
3. Open a terminal and change directory into <path-to-llvm-repo>/llvm-project/mlir/examples/treebeard.
```bash    
    mkdir build && cd build
    bash ../scripts/gen.sh [-b "cmake path"] [-m "mlir build directory name"][-c "Debug|Release"] 
    # Eg : bash ../scripts/gen.sh -m build.release -c Release 
    # If your mlir build is in a directory called "build.release" and
    # you're building Treebeard in Release
    cmake --build .
```
4. All command line arguments to gen.sh are optional. If cmake path is not specified above, the "cmake" binary in the path is used. The default mlir build directory name is "build". The default configuration is "Release".

## MLIR Version
The current version of Treebeard is tested with LLVM 16 (branch release/16.x of the LLVM github repo).

# Profiling Generated Code with VTune

Firstly, build MLIR with LLVM_USE_INTEL_JITEVENTS enabled. Add the following option to the cmake configuration command while building MLIR.
```bash
-DLLVM_USE_INTEL_JITEVENTS=1
```
You may need to fix some build errors in LLVM when you do this. In the commit referenced above, you will need to add the following line to IntelJITEventListener.cpp.
```C++
#include "llvm/Object/ELFObjectFile.h"
```
Build Treebeard (as described above), linking it to the build of MLIR with LLVM_USE_INTEL_JITEVENTS=1.

Set the following environment variables in the shell where you will run the Treebeard executable and in the shell from which you will launch VTune.
```bash
export ENABLE_JIT_PROFILING=1
export INTEL_LIBITTNOTIFY64=/opt/intel/oneapi/vtune/latest/lib64/runtime/libittnotify_collector.so
export INTEL_JIT_PROFILER64=/opt/intel/oneapi/vtune/latest/lib64/runtime/libittnotify_collector.so
```
The paths above are the default VTune installation paths. These maybe different if you've installed VTune in a different directory. Consider adding these variables to your .bashrc file.

Run the Treebeard executable with JIT profiler events enabled.
```bash
./treebeard --xgboostBench --enablePerfNotificationListener
```

TODO : You will need to modify the benchmark code to run only the test you want to profile.
