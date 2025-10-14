#!/usr/bin/env bash
# info.sh
# Run with: bash info.sh

# Print which services are running and where (with links)
echo ""
echo "🐳 Services are running here:"
echo " - ElasticSearch ....... https://localhost:9203"
echo " - Kibana .............. http://localhost:5604"
echo " - RabbitMQ ............ http://localhost:15672"
echo " - Redis Commander ..... http://localhost:8081"
echo " - Flower .............. http://localhost:5555"
echo " - Grafana ............. http://localhost:3000"
echo " - FastAPI ............. http://localhost:28800"
echo ""
echo "📊 Metrics/exporter services are running here:"
echo " - Prometheus .......... http://localhost:9090"
echo " - Celery exporter ..... http://localhost:9808/metrics"
echo " - Telegraf ............ http://localhost:9273/metrics"
echo ""
