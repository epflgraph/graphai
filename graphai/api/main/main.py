from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from graphai.celery.common.celery_tools import celery_instance

from graphai.celery.common.log import log

import graphai.api.image.router as image_router
import graphai.api.ontology.router as ontology_router
import graphai.api.text.router as text_router
import graphai.api.video.router as video_router
import graphai.api.voice.router as voice_router
import graphai.api.translation.router as translation_router
import graphai.api.embedding.router as embedding_router
import graphai.api.scraping.router as scraping_router
import graphai.api.retrieval.router as retrieval_router

from graphai.api.auth.router import (
    unauthenticated_router,
    authenticated_router
)
from graphai.api.auth.log import LoggerMiddleware

from graphai.celery.text.tasks import text_init_task
from graphai.celery.video.tasks import slide_detection_init_task
from graphai.celery.voice.tasks import transcript_init_task
from graphai.celery.embedding.tasks import embedding_init_task
from graphai.celery.translation.tasks import translation_init_task
from graphai.celery.ontology.tasks import ontology_init_task
from graphai.celery.scraping.tasks import scraping_init_task


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
    video_job = slide_detection_init_task.apply_async(priority=2)
    voice_job = transcript_init_task.apply_async(priority=2)
    embedding_job = embedding_init_task.apply_async(priority=6)
    translation_job = translation_init_task.apply_async(priority=6)
    ontology_job = ontology_init_task.apply_async(priority=6)
    scraping_job = scraping_init_task.apply_async(priority=6)

    # Wait for results
    text_ok = text_job.get()
    video_ok = video_job.get()
    voice_ok = voice_job.get()
    embedding_ok = embedding_job.get()
    translation_ok = translation_job.get()
    ontology_ok = ontology_job.get()
    scraping_ok = scraping_job.get()

    task_names = ['text', 'video', 'voice', 'embedding', 'translation', 'ontology', 'scraping']
    ok_list = [text_ok, video_ok, voice_ok, embedding_ok, translation_ok, ontology_ok, scraping_ok]

    if all(ok_list):
        log("All init tasks finished successfully")
    else:
        unsuccessful_indices = [i for i, x in enumerate(ok_list) if not x]
        unsuccessful_tasks = [task_names[i] for i in unsuccessful_indices]
        log(f"[ERROR] Init tasks for {', '.join(unsuccessful_tasks)} failed, check celery logs")

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
    description="This API, as part of the EPFLGraph project, offers several AI-related tools, "
                "such as automatic concept detection in text, audio and slide extraction from videos, "
                "text extraction from slides and audio transcription, text translation and embeddings, "
                "and much more.\nIf you would like to gain access to the API or are a user "
                "and require technical support, send an email to [ramtin dot yazdanian at epfl dot ch].",
    version="0.10.1",
    lifespan=lifespan
)
app.add_middleware(LoggerMiddleware)


# Include all routers in the app
authenticated_router.include_router(image_router.router)
authenticated_router.include_router(ontology_router.router)
authenticated_router.include_router(text_router.router)
authenticated_router.include_router(video_router.router)
authenticated_router.include_router(voice_router.router)
authenticated_router.include_router(translation_router.router)
authenticated_router.include_router(embedding_router.router)
authenticated_router.include_router(scraping_router.router)
authenticated_router.include_router(retrieval_router.router)


app.include_router(unauthenticated_router)
app.include_router(authenticated_router)
app.celery_app = celery_instance


# Root endpoint redirects to docs
@app.get("/")
async def redirect_docs():
    return RedirectResponse(url='/docs')
