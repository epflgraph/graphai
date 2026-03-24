echo "Starting all services..."
systemctl --user daemon-reload
systemctl --user start promtail.service
systemctl --user start celery-cpu-cache.service
systemctl --user start celery-cpu-rag.service
systemctl --user start celery-cpu-img.service
systemctl --user start celery-cpu-ontl_scrp_celery.service
systemctl --user start celery-cpu-text.service
systemctl --user start celery-cpu-embedding.service
systemctl --user start celery-cpu-video_voice_transl.service
systemctl --user start celery-gpu0-embedding.service
systemctl --user start celery-gpu1-voice.service
systemctl --user start celery-gpu2-voice.service
systemctl --user start celery-gpu3-translation.service
systemctl --user start uvicorn-api-graphai.service
echo "All services started."