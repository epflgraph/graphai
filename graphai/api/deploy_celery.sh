#!/bin/bash

celery -A main.celery_instance worker --loglevel=info --concurrency=16 -Q text,video,ontology