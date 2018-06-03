#!/bin/bash

unamestr=`uname`
if [[ "$unamestr" == "Darwin" ]]; then
    patch -l -p0 -i _patchfiles/patches
else
    patch -l -p0 -i _patchfiles/patches_win
fi
