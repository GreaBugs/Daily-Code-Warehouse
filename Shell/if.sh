#! /bin/bash

echo "请为level赋值:"
read level

if [ -n "$level" ]; then  # 如果level非空
	if [ "$level" == 0 ]; then
		prefix=ERROR
	elif [ "$level" == 1 ]; then
		prefix=INFO
	elif test $level = "2"; then
		prefix=GOOD
		echo "整挺好！"
	else
		echo "log level not supported"
	fi
fi

echo "[${prefix}] $message"

read -p "please input (Y/N):" yn

if [ "$yn" == "y" -o "$yn" == "Y" ]; then
	echo "ok continue"
fi

if [ "$yn" == "y" || "$yn" == "Y" ]; then
	echo "ok continue too"
fi
