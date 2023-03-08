#!/bin/bash
host="0.0.0.0"
port=28800

while getopts ":h:p:" opt; do
  case $opt in
    h) host="$OPTARG"
    ;;
    p) port="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG. Usage: deploy.sh -h 0.0.0.0 -p 28800" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

uvicorn main:app --host $host --port $port
