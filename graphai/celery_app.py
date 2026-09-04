# graphai/celery_app.py
# Worker entry point for the GraphAI Celery application.
# Read the queue-specific task imports from the environment so workers can
# avoid loading unrelated heavy subsystems such as whisper/torch, presidio, or
# transformers.
import os
from graphai.celery.common.celery_tools import celery_instance

# Apply the configured imports to the Celery application instance.
_worker_imports = os.environ.get("GRAPHAI_CELERY_IMPORTS", "")
if _worker_imports:
    celery_instance.conf.update(  # type: ignore[attr-defined]
        imports=[m.strip() for m in _worker_imports.split(",") if m.strip()]
    )
