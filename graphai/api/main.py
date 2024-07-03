from contextlib import asynccontextmanager

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
import graphai.api.routers.embedding as embedding_router
import graphai.api.routers.completion as summarization_router
import graphai.api.routers.scraping as scraping_router

from graphai.api.routers.auth import (
    unauthenticated_router,
    authenticated_router
)

from graphai.api.celery_tasks.text import text_init_task
from graphai.api.celery_tasks.common import video_init_task


# Define lifespan cycle of FastAPI app, i.e. what to do before startup and after shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function has three parts:
      * Before startup: Logic executed right before starting the API. Here we might want to load big objects into memory, etc.
      * Yield: This is the standard way of passing the execution to the FastAPI app, so it can normally boot and serve requests.
      * After shutdown: Logic executed right after shutting down the API. Here we might want to free some memory, do some cleanup, etc.
    """

    ################################################################
    # Before startup                                               #
    ################################################################

    log("Loading big objects and models into the memory space of the celery workers...")

    # Spawn tasks
    log("Spawning text_init and video_init tasks...")
    text_job = text_init_task.apply_async(priority=10)
    video_job = video_init_task.apply_async(priority=2)

    # Wait for results
    text_ok = text_job.get()
    video_ok = video_job.get()

    # Print status message
    if text_ok and video_ok:
        log("Tasks text_init and video_init both finished successfully")
    elif text_ok and not video_ok:
        log("[ERROR] Task video_init failed, check celery logs")
    elif not text_ok and video_ok:
        log("[ERROR] Task text_init failed, check celery logs")
    else:
        log("[ERROR] Both text_init and video_init tasks failed, check celery logs")

    ################################################################
    # Yield execution to API                                       #
    ################################################################
    yield

    ################################################################
    # After shutdown                                               #
    ################################################################
    pass


# Initialise FastAPI
app = FastAPI(
    title="EPFL Graph AI API",
    description="This API offers several tools related with AI in the context of the EPFL Graph project, "
                "such as automatized concept detection from a given text.",
    version="0.2.1",
    lifespan=lifespan
)


# Include all routers in the app
authenticated_router.include_router(image_router.router)
authenticated_router.include_router(ontology_router.router)
authenticated_router.include_router(text_router.router)
authenticated_router.include_router(video_router.router)
authenticated_router.include_router(voice_router.router)
authenticated_router.include_router(translation_router.router)
authenticated_router.include_router(embedding_router.router)
authenticated_router.include_router(summarization_router.router)
authenticated_router.include_router(scraping_router.router)


app.include_router(unauthenticated_router)
app.include_router(authenticated_router)
app.celery_app = celery_instance


# Root endpoint redirects to docs
@app.get("/")
async def redirect_docs():
    return RedirectResponse(url='/docs')
