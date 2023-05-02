#!/bin/bash

nice -n 20 celery -A main.celery_instance worker --hostname w1@%h -l debug -P threads --prefetch-multiplier 1 -c 16 -Q celery,video,ontology -D
nice -n 0 celery -A main.celery_instance worker --hostname w2@%h -l debug -P threads --prefetch-multiplier 4 -c 16 -Q text