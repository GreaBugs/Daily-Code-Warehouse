#! /bin/bash

name=john
case $name in 
	"nick")
		echo "hi nick"
	;;
	"john")
		echo "my name is john"
	;;
	*)
		echo "404"
	;;
esac
