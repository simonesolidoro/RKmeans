#!/bin/bash

n_threads=(1 2 4 8 12 16) # 1 lasciato per vedere overhead threadpool rispetto seq
output_file="test_assemble.txt"
for i in "${n_threads[@]}"; do
    ./test_assemble "$i" >> "$output_file"
done
