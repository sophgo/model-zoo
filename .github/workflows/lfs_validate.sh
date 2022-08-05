#!/bin/bash

DIR="$(dirname "$(realpath "$0")")"
DIR="$(realpath $DIR/../..)"

function check()
{
    local fn=$1
    git check-attr --all -- $fn  | grep lfs > /dev/null || {
        echo $fn should upload via lfs
        exit 1
    }
}

while read fn; do check $fn; done < <(find -name *.caffemodel)
while read fn; do check $fn; done < <(find -name *.prototxt)
while read fn; do check $fn; done < <(find -name *.pt)
while read fn; do check $fn; done < <(find -name *.onnx)
while read fn; do check $fn; done < <(find -name *.pb)
while read fn; do check $fn; done < <(find $DIR -size +1M ! -path *.git*)
