# Treebeard 
An optimizing compiler for decision tree ensemble inference.

# To build the Treebeard project
1. Setup a build of [MLIR](https://mlir.llvm.org/getting_started/). See section below for MLIR commit to use.
2. Clone this repository into <path-to-llvm-repo>/llvm-project/mlir/examples/
3. Open a terminal and change directory into <path-to-llvm-repo>/llvm-project/mlir/examples/tree-heavy.
```bash    
    mkdir build && cd build
    bash ../scripts/gen.sh [-b "cmake path"] [-m "mlir build directory name"][-c "Debug|Release"] 
    # Eg : bash ../scripts/gen.sh -b /snap/bin/cmake -m build (if your mlir build is in a directory called "build")
    cmake --build .
```
4. All command line arguments to gen.sh are optional. If cmake path is not specified above, the "cmake" binary in the path is used. The default mlir build directory name is "build". The default configuration is "Debug".

# MLIR Version
The current version of Treebeard is tested with the following LLVM commit:
```
commit a698790c51ec2804c3a7ba4c59438e7816690ea2 (HEAD -> main, origin/main, origin/HEAD)
Author: Noah Goldstein <goldstein.w.n@gmail.com>
Date:   Fri Jan 6 14:08:25 2023 +0800

    [X86] Add additional tests to no-shift.ll
    
    Reviewed By: pengfei
    
    Differential Revision: https://reviews.llvm.org/D141076
```

# Python API
Building Treebeard will generate a library <build_folder>/lib/libtreebeard-runtime.so.*. Copy this file into <Treebeard_Root>/src/python/libtreebeard-runtime.so . Now run <Treebeard_Root>/test/python/run_python_tests.py and verify that tests pass. See the file run_python_tests.py for an example of how the Treebeard API is imported.

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
./tree-heavy --xgboostBench --enablePerfNotificationListener
```

TODO : You will need to modify the benchmark code to run only the test you want to profile.