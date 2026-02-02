#!/bin/bash

# $1 = numero di run
# $2 = numero di nodi

output_file="test_calcolo_triple_nodi${2}.txt"
touch "$output_file"

for ((i=1; i<=${1}; i++)); do
    ./test_calcolo_triple "${2}" >> "$output_file"
    sleep 0.1
done
