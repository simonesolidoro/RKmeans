#!/bin/bash

#$1 is the number of times the file is ran

#$2 size matrix 
#$3 n thread
#$4 size queue of worker
#$5 granularity


granularity=($5)

# Per ogni eseguibile nella lista
for nb in "${granularity[@]}"; do
    # Crea (o sovrascrive) il file di output
    output_file="test_parallel_for_matrix_gr${nb}.txt"
    #rm -f "$output_file"
    touch "$output_file"

    # Esegui il programma N volte e salva l'output
    for i in $(seq 1 "$1"); do
        ./"test_parallel_for_matrix" "$2" "$3" "$4" "$nb">> "$output_file" 
        sleep 0.1  # evita congestione
    done
    #sleep 0.1  # evita congestione
done
