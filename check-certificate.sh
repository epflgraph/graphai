#!/bin/bash

CERT_FILE="/etc/letsencrypt/live/graphcert.cede-apps.ch/fullchain.pem"
HOST_NAME="graphai.cede-apps.ch"
PORT=443

echo
echo "GraphAI certificate check"
echo

loaded=$(echo | openssl s_client \
    -connect localhost:"$PORT" \
    -servername "$HOST_NAME" 2>/dev/null |
    openssl x509 -noout -enddate 2>/dev/null |
    cut -d= -f2-)

disk=$(openssl x509 -in "$CERT_FILE" -noout -enddate | cut -d= -f2-)

echo "Loaded: $loaded"
echo "Disk:   $disk"
echo

if [ "$loaded" = "$disk" ]; then
    echo "✅ GraphAI is using the current certificate."
    echo
    exit 0
else
    echo "❌ GraphAI is NOT using the current certificate."
    echo
    exit 1
fi
