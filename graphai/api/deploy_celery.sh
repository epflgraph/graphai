#!/bin/bash

celery -A main.celery_instance worker --pool threads --loglevel=debug -Q common,text,video,ontology