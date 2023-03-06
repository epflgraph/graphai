#!/bin/bash

if [ "$#" -lt 1 ]; then
    printf "Missing parameters. Usage: wikify-local.sh (file.json)+\n";
    exit
fi

for file in "$@"
do
    if [ "$#" -ne 1 ]; then
        printf "$file\n"
    fi
    curl -X POST http://localhost:28800/wikify -H 'Content-Type: application/json' --data-binary "@$file";
    printf "\n"
done
