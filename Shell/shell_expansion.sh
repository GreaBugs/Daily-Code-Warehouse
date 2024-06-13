#! /bin/bash

# 大括号展开（Brace Expansion）{...}
# 字符串序列
a{b,c,d}e

# 表达式序列，（数字可以使用incr调整增量，字母不行）
{1..5}
{1..5..2}
{a..e}

echo a{b,c,d}e

# 参数展开 ${}
# 1. 间接参数扩展 ${!parameter}，其中引用的参数并不是parameter而是parameter的实际的值
parameter="var"
var="hello"
echo ${!parameter}

# 2. 参数长度 ${#parameter}
par=shen
echo ${#par}

# 3. 空参数处理
# ${parameter:-word}  为空替换
# ${parameter:=word}  为空替换，并将值赋给$parameter变量
# ${parameter:?word}  为空报错
# ${parameter:+word}  不为空替换
a=1
echo ${a:word}
echo ${b:-word}

echo ${par:=word}
echo ${par:-hello}
echo ${par:+foo}

