#!/bin/bash
for ((i=1; i < 10; i++))
do
    echo $i
    python3 main.py show net --dataset even --dataset-config fill --save -n $i
done