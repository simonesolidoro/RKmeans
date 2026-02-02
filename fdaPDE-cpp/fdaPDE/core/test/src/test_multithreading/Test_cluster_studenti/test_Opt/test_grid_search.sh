#!/bin/bash

n_threads=(1 2 3 4 5 6 7 8) # 1 lasciato per vedere overhead threadpool rispetto seq
output_file="test_grid_search.txt"
for i in "${n_threads[@]}"; do
    ./test_grid_search "$i" >> "$output_file"
done
