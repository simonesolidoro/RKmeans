#!/bin/bash

#$1 is the name of the file to be ran
#$2 type deque,hold,hold_nowait, relax
#$3 is the number of times the file is ran
#$4 is the number of elements to be handled
#$5 number thread

# Lista di eseguibili 

tag="$1"
type="$2"
executable="test_${type}_${tag}_multi"

# Crea (o sovrascrive) il file di output
output_file="${executable}_numthread"$5".txt"
#rm -f "$output_file"
touch "$output_file"

# Esegui il programma N volte e salva l'output
for i in $(seq 1 "$3"); do
    ./"$executable" "$4" "$5" >> "$output_file" 
    sleep 0.1  # evita congestione
done

