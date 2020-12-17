#/bin/bash
echo "start learning"
for i in {1..4};
do
# echo learn big_1000 big_${i}
python3.8 main.py learn big_1000 big_${i} &
done;
echo "end learning"