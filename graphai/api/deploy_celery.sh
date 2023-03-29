#!/bin/bash

celery -A main.celery_instance worker --loglevel=debug --concurrency=16 -Q text,video,ontology