#!/bin/bash

#$1 is the name of the file to be ran
#$2 is the number of times the file is ran
#$3 is the number of elements to be handled

#####ES ./multi.sh pop_back 10 100

# n_thread= []
threads=(2 4 6 8)
# Lista di eseguibili 

tag="$1"
executables=(
    "test_tbbqueue_push_multi"
    "test_relax_${tag}_multi"
    "test_hold_${tag}_multi"
    "test_hold_wait_${tag}_multi"
)

for th in "${threads[@]}"; do
    # Per ogni eseguibile nella lista
    for exe in "${executables[@]}"; do
        # Crea (o sovrascrive) il file di output
        output_file="${exe}_numthread${th}.txt"
        #rm -f "$output_file"
        touch "$output_file"

        # Esegui il programma N volte e salva l'output
        for i in $(seq 1 "$2"); do
            ./"$exe" "$3" "${th}" >> "$output_file" 
            sleep 0.1  # evita congestione
        done
    done
done
