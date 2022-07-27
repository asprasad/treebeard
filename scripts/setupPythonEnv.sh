#!/bin/bash
SCRIPT=`realpath "$BASH_SOURCE"`
SCRIPTPATH=`dirname $SCRIPT`
TREEBEARD_DIR=`dirname $SCRIPTPATH`
echo "Using Treebeard rooted at : $TREEBEARD_DIR"

BUILD_DIR="build"

while getopts "m:" opt
do
   case "$opt" in
      m ) BUILD_DIR="$OPTARG" ;;
   esac
done

echo "Using build at : $BUILD_DIR"
LIB_BUILD_PATH="$TREEBEARD_DIR/$BUILD_DIR/lib"
PYTHON_SUPPORT_DIR="$TREEBEARD_DIR/src/python"

# copy the built runtime libraries into the python support folder
cp $LIB_BUILD_PATH/* $PYTHON_SUPPORT_DIR

# set up python path so that the treebeard module can be imported
export PYTHONPATH=$PYTHON_SUPPORT_DIR:${PYTHONPATH}