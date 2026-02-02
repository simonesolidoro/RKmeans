#!/bin/bash


#$1 is the number of times the file is ran
#$2 is the number of elements to be handled

#####ES ./multi.sh pop_back 10 100

# n_thread= []
threads=(1 2 4 8)

exe="test_relax_random_accuracy"
for th in "${threads[@]}"; do
    # Crea (o sovrascrive) il file di output
    output_file="${exe}_numthread${th}.txt"
    #rm -f "$output_file"
    touch "$output_file"

    # Esegui il programma N volte e salva l'output
    for i in $(seq 1 "$1"); do
        ./"$exe" "$2" "${th}" >> "$output_file" 
        sleep 0.1  # evita congestione
    done
done
