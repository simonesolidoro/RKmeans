#!/bin/bash
#"$1" tot elem 
#"$2" runs

n_threads=(1 2 3 4 5 6 7 8) # 1 lasciato per vedere overhead threadpool rispetto seq
output_file="test_sq_Opt_totel_${1}_run_${2}.txt"
for i in "${n_threads[@]}"; do
    ./test_sq "$i" "$1" "$2" >> "$output_file"
done
