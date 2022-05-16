#!/bin/bash
if [ "$#" -ge 2 ]; then
    printf "Too many parameters. Usage: deploy-prod.sh [host]. Example: deploy.sh 192.168.140.120\n";
    exit
fi

if [ "$#" -ge 1 ]; then
    uvicorn main:app --host $1 --port 28800 --reload
    # gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind $1:28800
else
    uvicorn main:app --host 192.168.140.120 --port 28800 --reload
    # gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 192.168.140.120:28800
fi
