#!/usr/bin/env bash

if [[ $# -eq 0 ]]; then
    mkdir -p build
    ./build.sh release Release &
    # ./build.sh debug Debug &
    wait
    exit 0
else
    mkdir -p build/"$1"
    cd build/"$1"
    cmake -DCMAKE_BUILD_TYPE="$2" ../..
    make
    # make test
fi
