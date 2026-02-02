#!/bin/bash

#$1 is the number of times the file is ran

#$2 grid size 

output_file="test_optimization_sizegrid"$2".txt"
touch "$output_file"

# Esegui il programma N volte e salva l'output
for i in $(seq 1 "$1"); do
    ./"test_optimization" "$2" >> "$output_file" 
    sleep 0.1  # evita congestione
done

