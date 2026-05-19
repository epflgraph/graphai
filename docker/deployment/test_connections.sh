#!/usr/bin/env bash
set -euo pipefail

echo ""
echo "🚀 Starting service connectivity tests..."

if [[ ! -f ".env" ]]; then
    echo "❌ .env file not found!"
    exit 1
fi

echo ""
echo "📦 Loading environment variables..."
set -a
source .env
set +a

echo ""
echo "🐬 Testing MySQL connection..."
if docker run --rm mysql:8 \
    mysql --protocol=TCP \
    -h "$DB_HOST" \
    -P "$DB_PORT" \
    -u "$DB_USER" \
    -p"$DB_PASSWORD" \
    -e "SELECT 1;" >/dev/null; then
    echo "✅ MySQL connection OK!"
else
    echo "❌ MySQL connection failed!"
    exit 1
fi

echo ""
echo "🔎 Testing ElasticSearch health..."
if docker run --rm \
    -v "$HOST_CERTS_DIR:/app/certs:ro" \
    curlimages/curl:latest \
    -f \
    -k \
    -u "$ES_USERNAME:$ES_PASSWORD" \
    "https://$ES_HOST:$ES_PORT/_cluster/health";
then
    echo ""
    echo "✅ ElasticSearch connection OK!"
else
    echo "❌ ElasticSearch connection failed!"
    exit 1
fi

echo ""
echo "All tests completed."
echo ""
