#! /bin/bash

sum=0
for((i=1;i<30;i++))
do
	echo "请输入分数:"
	read score
	if [ "$score" -lt 101 ]; then
		sum=$(($sum + $score))
	else
		average=$(($sum / $((i - 1)))) 
    echo "$((i - 1))门课程的平均分为:"
    echo "scale=4;$sum / $((i - 1))" | bc
    sum=0
	fi
done


