#!/bin/bash
for ((i=23; i < 29; i++))
do
    echo $i
    # python3 main.py show loss --save --file even_${i}.log
    # python3 main.py show pred --dataset-config fill --dataset even  --model even_18-even --model-config even_18 --save -n $i
    # python3 main.py show net --dataset-config fill --dataset even  --model even_18-even --model-config even_18 --save -n $i
    # python3 main.py show response --dataset-config fill --dataset even  --model even_18-even --model-config even_18 --save -n $i
done
for ((i=20; i < 21; i++))
do
    echo $i
    # python3 main.py show pred --dataset-config fill --dataset even  --model even_25-even --model-config even_25 --save -n $i
    python3 main.py show net --dataset-config fill_2 --dataset fill_2  --model even_18-even --model-config even_18 --save -n $i
    python3 main.py show net --dataset-config fill_2 --dataset fill_2  --model even_18-even --model-config even_18 --save -n $i --type reverse --alpha 1.0
    # python3 main.py show response --dataset-config fill --dataset even  --model even_18-even --model-config even_18 --save -n $i
done