#!/bin/bash
for ((i=1; i < 16; i++))
do
    echo $i
    python3 main.py learn cnn_flat_$i even
done