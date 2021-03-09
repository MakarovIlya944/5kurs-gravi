#!/bin/bash
for ((i=1; i < 16; i++))
do
    echo $i
    python3 main.py learn even_$i even
done