#!/bin/bash
max=1000
N=100
echo "Dataset Collection Started"
for((i=1; i<=max; i++))
	do
		now=$(date +"%T")
		echo "Start $i at : $now"
		
		mkdir ./Internal_Energy
		mkdir ./Internal_Energy/$i
		
		./run --NBEAD $N --SWEEP 100000000 --SAMPLE 1000000 --SAVE_LOC ./Internal_Energy/$i/
		
		echo "RAN $i with Sweep $sweep Sample $sample"
		ls ./Internal_Energy/$i | wc -l
		now=$(date +"%T")
		echo "Done $i at : $now"
		
	done