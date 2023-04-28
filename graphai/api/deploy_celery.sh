#!/bin/bash

celery -A main.celery_instance worker --hostname w1@%h -l debug -P threads --prefetch-multiplier 1 -c 16 -Q common,video,ontology -D
celery -A main.celery_instance worker --hostname w2@%h -l debug -P prefork --prefetch-multiplier 4 -c 16 -Q text