#!/bin/bash

ls -l _patchfiles/* | grep 'txt' | awk '{print $9}' | xargs patch -p0 -i
