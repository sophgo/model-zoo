#!/bin/bash

echo "REMOTE=\"${REMOTE}\""

top=$(realpath $(dirname -- "$0")/../../)

set -eE

[[ $top -ef ./ ]] || pushd $top

mkdir -p data
mkdir -p output

revision=$(date +%Y%m%d)_$(git rev-parse --short HEAD)

which pbzip2 &> /dev/null || {
    sudo apt-get update
    sudo apt-get install -y pbzip2
}

echo "Save whole model-zoo..."
tar -I pbzip2 \
    -cO \
    --transform 's#^#model-zoo/#' \
    --exclude ILSVRC2012_img_val/* \
    --exclude .git \
    --exclude 'output/*' \
    --exclude 'data/*' \
    --exclude .github \
    --exclude __pycache__ \
    --exclude 'venv' * | curl \
    -T - $REMOTE/model-zoo_${revision}.tar.bz2

[[ $top -ef ./ ]] || popd
