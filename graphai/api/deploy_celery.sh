#!/bin/bash

nice -n 0 celery -A main.celery_instance beat --detach
nice -n 0 celery -A main.celery_instance worker --hostname workerHigh@%h -l info -P threads --prefetch-multiplier 20 -c 16 -Q text_10 -D
nice -n 20 celery -A main.celery_instance worker --hostname workerLow@%h -l info -P threads --prefetch-multiplier 1 -c 16 -Q celery,video_2,ontology_6,text_6 -D
nice -n 20 celery -A main.celery_instance worker --hostname workerMid@%h -l info -P prefork --prefetch-multiplier 1 -c 20 -Q scraping_6 -D
nice -n 10 celery -A main.celery_instance worker --hostname workerCache@%h -l info -P threads --prefetch-multiplier 10 -c 16 -Q caching_6 -D
FLOWER_UNAUTHENTICATED_API=true celery -A main.celery_instance flower --port=5555
