#!/bin/bash

nice -n 20 celery -A main.celery_instance worker --hostname w1@%h -l debug -P threads --prefetch-multiplier 1 -c 16 -Q celery,video_2,ontology_6,text_6 -D
nice -n 0 celery -A main.celery_instance worker --hostname w2@%h -l debug -P threads --prefetch-multiplier 20 -c 16 -Q text_10 -D
celery -A main.celery_instance flower --port=5555
