#!/bin/bash

#$1 is the number of times the file is ran

#$2 size of nxn matrix


# test_for.txt 
#rm -f "test_for.txt"
touch "test_for_matrix.txt"

# Esegui il programma N volte e salva l'output
for i in $(seq 1 "$1"); do
    ./"test_for_matrix" "$2" >> "test_for_matrix.txt" 
    sleep 0.1
done
