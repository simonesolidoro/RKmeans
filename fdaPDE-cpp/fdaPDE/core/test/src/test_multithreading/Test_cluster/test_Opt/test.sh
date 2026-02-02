#!/bin/bash

n_threads=(1 2 4 8 16 32 64 96) # 1 lasciato per vedere overhead threadpool rispetto seq

for i in "${n_threads[@]}"; do
    output_file="test_Opt_thread_${i}_.txt"
    ./test "$i" >> "$output_file"
done
