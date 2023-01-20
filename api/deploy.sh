#!/bin/bash
if [ "$#" -ge 2 ]; then
    printf "Too many parameters. Usage: deploy.sh [host]. Example: deploy.sh 0.0.0.0\n";
    exit
fi

if [ "$#" -ge 1 ]; then
    uvicorn main:app --host $1 --port 28800 --reload
else
    uvicorn main:app --host 0.0.0.0 --port 28800 --reload
fi
