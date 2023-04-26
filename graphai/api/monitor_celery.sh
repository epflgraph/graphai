#!/bin/bash

celery -A main.celery_instance flower --port=5555