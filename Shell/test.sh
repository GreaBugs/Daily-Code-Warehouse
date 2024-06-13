#! /bin/bash


#整数测试
echo "请输入n1的数值："
read n1
echo "请输入n2的数值："
read n2

test $n1 -eq $n2
test $n1 -lt $n2
test $n1 -gt $n2

#字符串测试
test -z $str_a
test -n $str_a
test $str_a=$str_b

#文件测试
test -e /dmt && echo "exist"
test -f /usr/bin/npm && echo "file exist"
