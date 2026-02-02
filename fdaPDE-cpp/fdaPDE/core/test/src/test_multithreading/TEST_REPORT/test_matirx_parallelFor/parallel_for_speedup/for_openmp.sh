#!/bin/bash

#$1 is the number of times the file is ran

#$2 size matrix 
#$3 n thread


# n_blocchi = []
threads=($3)

# Per ogni eseguibile nella lista
for nb in "${threads[@]}"; do
    # Crea (o sovrascrive) il file di output
    output_file="test_for_openmp_th${nb}.txt"
    #rm -f "$output_file"
    touch "$output_file"

    # Esegui il programma N volte e salva l'output
    for i in $(seq 1 "$1"); do
        ./"test_for_openmp" "$2" "$nb">> "$output_file" 
        sleep 0.1  # evita congestione
    done
    #sleep 0.1  # evita congestione
done
