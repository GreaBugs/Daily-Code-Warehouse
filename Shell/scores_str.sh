#! /bin/bash

scores="现代优化方法(专硕):92 多媒体技术:90 模式识别:91 新时代中国特色社会主义理论与实践: 97 硕士研究生英语:76 自然辩证法概论:89 随机过程:85  矩阵论:100  算法设计与分析:92"
i=0
sum=0
echo "请输入密码: "
read password
if [[ "$password" = "521212YM" ]];then
	for score in 92 90 91 97 76 89 85 100 92
	do
  		sum=$((sum + score))
 		i=$((i + 1))
	done
	echo $scores
	echo "$i门课程的平均分为:"
	echo "scale=4;$sum / $i" | bc
else 
	echo "密码输入错误！"
fi
