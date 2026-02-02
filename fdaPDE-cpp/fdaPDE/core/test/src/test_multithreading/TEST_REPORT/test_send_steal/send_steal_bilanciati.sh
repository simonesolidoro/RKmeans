#!/bin/bash

#$1 is the number of times the file is ran

#$2 size worker_queue
#$3 n_cicli in singolo job (ogni ciclo 4 operation)
#$4 n_job totali

#job eseguiti in single thread per poi confronto in speedup
rm -f "test_job_single_thread.txt"
touch "test_job_single_thread.txt"

# Esegui il programma N volte e salva l'output
for i in $(seq 1 "$1"); do
    ./"test_job_single_thread" "1" "$2" "$3" "$4" >> "test_job_single_thread.txt" 
done

# n_thread= []
threads=(8 6 4 2)
# Lista di eseguibili (modifica questa parte se serve)
executables=("test_sendMostfree_stealMostbusy" "test_sendRound_stealMostbusy" "test_sendMostfree_stealRandom" "test_sendRound_stealRandom" )

for th in "${threads[@]}"; do
    # Per ogni eseguibile nella lista
    for exe in "${executables[@]}"; do
        # Crea (o sovrascrive) il file di output
        output_file="${exe}_numthread${th}.txt"
        rm -f "$output_file"
        touch "$output_file"

        # Esegui il programma N volte e salva l'output
        for i in $(seq 1 "$1"); do
            ./"$exe" "$th" "$2" "$3" "$4" >> "$output_file" 
        done
    done
done
