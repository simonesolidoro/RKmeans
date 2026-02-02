#!/bin/bash


#$1 is the number of times the file is ran
#$2 is the number of elements to be handled

#####ES ./multi.sh pop_back 10 100

# n_thread= []
threads=(2 4 6 8)
# Lista di eseguibili 

executables=(
    "test_deque_pop_push_rand"
    "test_hold_pop_push_rand"
    "test_hold_wait_pop_push_rand"
    "test_relax_pop_push_rand"
)

for th in "${threads[@]}"; do
    # Per ogni eseguibile nella lista
    for exe in "${executables[@]}"; do
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
done

