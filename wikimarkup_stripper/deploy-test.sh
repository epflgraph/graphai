#!/bin/bash
if [ "$#" -ge 2 ]; then
    printf "Too many parameters. Usage: deploy-test.sh [host]. Example: deploy.sh 192.168.142.120\n";
    exit
fi

if [ "$#" -ge 1 ]; then
    uvicorn main:app --host $1 --port 30010 --app-dir src --reload
else
    uvicorn main:app --host 192.168.142.120 --port 30010 --app-dir src --reload
fi
