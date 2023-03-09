#!/bin/bash
if [ "$#" -ge 2 ]; then
    printf "Too many parameters. Usage: deploy-local.sh [host]. Example: deploy.sh 0.0.0.0\n";
    exit
fi

# If you want to see celery logs, run the following without --detach in a separate command line and then deploy
celery -A main.celery_instance worker --loglevel=info -Q text,video,ontology --detach

if [ "$#" -ge 1 ]; then
    uvicorn main:app --host $1 --port 28800
else
    uvicorn main:app --host 0.0.0.0 --port 28800
fi
