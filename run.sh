#!/bin/bash

# g++ -Wall -std=c++14 -o kmeans kmeans.cpp -l pthread
# for number in {1..16}
# do
# 	echo $number
# 	./kmeans --input random-n65536-d32-c16.txt --threshold 0.0000001 --iterations 20 --clusters 16 --workers $number >> out
# 	# echo $output >> part2_spin.out
# done
# ./kmeans --input "in" --threshold 0.0000001 --iterations 20 --clusters 2 --workers 4

nvcc -std=c++11 -arch=sm_61 -rdc=true -O3 -o kmeans kmeans.cu

./kmeans --input random-n2048-d16-c16.txt --threshold 0.0000001 --iterations 20 --clusters 16 --workers 1
./kmeans --input random-n16384-d24-c16.txt --threshold 0.0000001 --iterations 20 --clusters 16 --workers 1
./kmeans --input random-n65536-d32-c16.txt --threshold 0.0000001 --iterations 20 --clusters 16 --workers 1