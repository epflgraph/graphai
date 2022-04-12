#!/bin/bash
if [ "$#" -ge 2 ]; then
    printf "Too many parameters. Usage: deploy-prod.sh [host]. Example: deploy.sh 0.0.0.0\n";
    exit
fi

if [ "$#" -ge 1 ]; then
    gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind $1:28900
else
    gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:28900
fi
