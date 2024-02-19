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
import graphai.api.routers.completion as summarization_router
import graphai.api.routers.scraping as scraping_router

from graphai.api.routers.auth import (
    unauthenticated_router,
    authenticated_router
)

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
authenticated_router.include_router(image_router.router)
authenticated_router.include_router(ontology_router.router)
authenticated_router.include_router(text_router.router)
authenticated_router.include_router(video_router.router)
authenticated_router.include_router(voice_router.router)
authenticated_router.include_router(translation_router.router)
authenticated_router.include_router(summarization_router.router)
authenticated_router.include_router(scraping_router.router)


app.include_router(unauthenticated_router)
app.include_router(authenticated_router)
app.celery_app = celery_instance


# On startup, we spawn tasks to initialise services and variables in the memory space of the celery workers
@app.on_event('startup')
async def init():
    log('Loading big objects and models into the memory space of the celery workers...')

    # Spawn tasks
    log('Spawning text_init task...')
    text_job = text_init_task.apply_async(priority=10)
    log('Spawning video_init task...')
    video_job = video_init_task.apply_async(priority=2)

    # Wait for results
    log('Waiting for text_init task...')
    text_ok = text_job.get()
    log('Done')

    log('Waiting for video_init task...')
    video_ok = video_job.get()
    log('Done')

    # Print status message
    if text_ok and video_ok:
        log('Loaded')
    else:
        log('ERROR: Loading unsuccessful, check celery logs')


# Root endpoint redirects to docs
@app.get("/")
async def redirect_docs():
    return RedirectResponse(url='/docs')
