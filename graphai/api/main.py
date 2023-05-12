from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from graphai.api.common.celery_tools import celery_instance

from graphai.api.common.log import log

import graphai.api.routers.image as image_router
import graphai.api.routers.ontology as ontology_router
import graphai.api.routers.text as text_router
import graphai.api.routers.video as video_router
import graphai.api.routers.voice as voice_router
import graphai.api.routers.translation as translation_router

from graphai.api.celery_tasks.text import text_init_task
from graphai.api.celery_tasks.video import video_init_task


# Initialise FastAPI
app = FastAPI(
    title="EPFL Graph AI API",
    description="This API offers several tools related with AI in the context of the EPFL Graph project, "
                "such as automatized concept detection from a given text.",
    version="0.2.1"
)

# Include all routers in the app
app.include_router(image_router.router)
app.include_router(ontology_router.router)
app.include_router(text_router.router)
app.include_router(video_router.router)
app.include_router(voice_router.router)
app.include_router(translation_router.router)
app.celery_app = celery_instance


# On startup, we spawn tasks to initialise services and variables in the memory space of the celery workers
@app.on_event('startup')
async def init():
    log(f'Loading big objects and models into the memory space of the celery workers...')

    # Spawn tasks
    text_job = text_init_task.apply_async(priority=10)
    video_job = video_init_task.apply_async(priority=2)

    # Wait for results
    text_ok = text_job.get()
    video_ok = video_job.get()

    # Print status message
    if text_ok and video_ok:
        log(f'Loaded')
    else:
        log(f'ERROR: Loading unsuccessful, check celery logs')


# Root endpoint redirects to docs
@app.get("/")
async def redirect_docs():
    return RedirectResponse(url='/docs')
