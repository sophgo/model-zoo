#!/bin/bash

python3 run.py --model=/mnt/onager/my/bert-tianjin/compilation/int8.bmodel \
                --data=/mnt/hussar/sophon_mlperf/dataset/squad/dev-v1.1.json \
                --count=100 \
                --accuracy \

