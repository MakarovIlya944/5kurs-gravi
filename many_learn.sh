#!/bin/bash
for ((i=30; i > 25; i--))
do
    echo $i
    python3 main.py learn even_$i fill_2
    # python3 main.py learn cnn_flat_$i even
done