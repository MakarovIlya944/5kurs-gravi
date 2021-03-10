for($i=23; $i -le 31; $i++)
{
    python main.py show loss --save --file even_${i}.log;
    for($j=10; $j -le 20; $j++)
    {
        Write-Host $i-$j
        python main.py show pred --dataset-config fill --dataset even  --model even_${i}-even --model-config even_${i} --save -n $j
        python main.py show net --dataset-config fill --dataset even  --model even_${i}-even --model-config even_${i} --save -n $j
    }
}
for($j=10; $j -le 20; $j++)
{
    python main.py show response --dataset-config fill --dataset even --save -n $j
}