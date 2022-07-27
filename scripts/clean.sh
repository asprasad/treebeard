#!/bin/bash
CMAKE=$1
CMAKE=${CMAKE:="cmake"}
echo "Using cmake command : $CMAKE"
$CMAKE --build . --target clean