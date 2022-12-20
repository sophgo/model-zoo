#!/bin/bash

DATA_SET="squad"
DATA_ROOT_DIR="./dataset"
DATA_DIR="${DATA_ROOT_DIR}/${DATA_SET}"
rm -rf $DATA_DIR
mkdir $DATA_DIR
wget -O $(DATA_DIR)/dev-v1.1.json https://github.com/rajpurkar/SQuAD-explorer/blob/master/dataset/dev-v1.1.json?raw=true;