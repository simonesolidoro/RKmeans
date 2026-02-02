#!/bin/bash

#$1 runs
#$2 nodi 
#$3 n thread
#$4 granularity

output_file="test_calcolo_triple_nodi"$2"_thread"$3".txt"
touch "$output_file"

for ((i=1; i<=${1}; i++)); do
    ./"test_calcolo_triple_parallel" "$2" "$3" "$4" >> "$output_file" 
    sleep 0.1
done


