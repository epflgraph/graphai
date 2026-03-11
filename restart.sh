echo "Restarting all services..."
systemctl --user daemon-reload
systemctl --user restart promtail.service
systemctl --user restart celery-cpu-cache.service
systemctl --user restart celery-cpu-rag.service
systemctl --user restart celery-cpu-img.service
systemctl --user restart celery-cpu-ontl_scrp_celery.service
systemctl --user restart celery-cpu-text.service
systemctl --user restart celery-cpu-video_voice_transl.service
systemctl --user restart celery-gpu0-embedding.service
systemctl --user restart celery-gpu1-voice.service
systemctl --user restart celery-gpu2-voice.service
systemctl --user restart celery-gpu3-translation.service
systemctl --user restart uvicorn-api-graphai.service
echo "All services restarted."