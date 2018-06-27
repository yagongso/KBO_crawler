#!/bin/bash

unamestr=`uname`
if [[ "$unamestr" == "Darwin" ]]; then
    FILES=_patchfiles/mac/*
    for f in $FILES
    do
        patch -N -l -p0 --dry-run -s < $f 2>/dev/null
        if [ $? -eq 0];
        then
            patch -N -l -p0 < $f
        fi
    done
else
    FILES=_patchfiles/win/*
    for f in $FILES
    do
        patch -N -l -p0 --dry-run -s < $f 2>nul
        if [ $? -eq 0];
        then
            patch -N -l -p0 < $f
        fi
    done
fi
