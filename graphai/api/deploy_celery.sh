#!/bin/bash

celery -A main.celery_instance worker --loglevel=debug -Q common,text,video,ontology