#!/bin/bash
for ((i=23; i < 29; i++))
do
    echo $i
    python3 main.py show loss --save --file even_${i}.log
    # python3 main.py show pred --dataset-config fill --dataset even  --model even_18-even --model-config even_18 --save -n $i
    # python3 main.py show net --dataset-config fill --dataset even  --model even_18-even --model-config even_18 --save -n $i
    # python3 main.py show response --dataset-config fill --dataset even  --model even_18-even --model-config even_18 --save -n $i
done
# for ((i=1; i < 20; i++))
# do
#     echo $i
#     # python3 main.py show loss --save --file even_${i}.log
#     python3 main.py show pred --dataset-config fill --dataset even  --model even_25-even --model-config even_25 --save -n $i
#     # python3 main.py show net --dataset-config fill --dataset even  --model even_18-even --model-config even_18 --save -n $i
#     # python3 main.py show response --dataset-config fill --dataset even  --model even_18-even --model-config even_18 --save -n $i
# done