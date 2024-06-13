#! /bin/bash

#语法一
#funcName(){echo "abc";}

#语法二
#function funcName(){echo "abc"}

printName()
{
if [ $# -lt 2 ]; then
	echo "illegal parameter."
	exit 1
fi

echo "firstname is:$1" # 第一个变量
echo "lastname is:$2" # 第二个变量
}

printName jacky chen

#例2
function test()
{
	local word="hello world"
	echo $word

	unset word  #撤销变量
}
content=`test`

echo "状态码: $?"

echo "执行结果: $content"
