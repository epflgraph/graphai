echo "Stopping all services..."
systemctl --user stop promtail.service
systemctl --user stop celery-cpu-cache.service
systemctl --user stop celery-cpu-embedding.service
systemctl --user stop celery-cpu-image.service
systemctl --user stop celery-cpu-ontl_scrp_celery.service
systemctl --user stop celery-cpu-rag.service
systemctl --user stop celery-cpu-text.service
systemctl --user stop celery-cpu-translate.service
systemctl --user stop celery-cpu-video.service
systemctl --user stop celery-cpu-voice.service
systemctl --user stop celery-gpu0-embedding.service
systemctl --user stop celery-gpu1-voice.service
systemctl --user stop celery-gpu2-voice.service
systemctl --user stop celery-gpu3-translation.service
systemctl --user stop uvicorn-api-graphai.service
systemctl --user daemon-reload
echo "All services stopped."