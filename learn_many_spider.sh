#/bin/bash
echo "start learning"
for i in {1..6};
do
# echo learn big_1000 big_${i}
./Mnist learn big_1000 big_${i} &
done;
echo "end learning"