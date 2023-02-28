#!/bin/bash

if [ "$#" -lt 1 ]; then
    printf "Missing parameters. Usage: strip-test.sh (file.json)+\n";
    exit
fi

for file in "$@"
do
    printf "$file\n"
    curl -X POST http://86.119.27.90:30010/strip -H 'Content-Type: application/json' --data-binary "@$file";
    printf "\n"
done
