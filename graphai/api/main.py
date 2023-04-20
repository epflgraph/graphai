from fastapi import FastAPI
from fastapi.responses import RedirectResponse

import graphai.api.routers.ontology as ontology_router
# import graphai.api.routers.text as text_router
import graphai.api.routers.video as video_router
import graphai.api.routers.voice as audio_router
import graphai.api.routers.image as image_router
from graphai.api.celery_tasks.common import lazy_loader_task

from graphai.api.common.log import log

from graphai.core.interfaces.celery_config import create_celery

# from graphai.core.text.wikisearch import ws_actor_list

# Initialise FastAPI
app = FastAPI(
    title="EPFL Graph AI API",
    description="This API offers several tools related with AI in the context of the EPFL Graph project, "
                "such as automatized concept detection from a given text.",
    version="0.2.1"
)

# Include all routers in the app
app.include_router(ontology_router.router)
# app.include_router(text_router.router)
app.include_router(video_router.router)
app.include_router(audio_router.router)
app.include_router(image_router.router)
app.celery_app = create_celery()
celery_instance = app.celery_app


# On startup, we instantiate concepts graph and ontology, so they are held into memory
@app.on_event("startup")
async def lazy_load_singletons():
    # Loading all the lazy-loaded objects in the celery process
    log(f'Loading all lazy-loaded objects')
    lazy_loading_successful = lazy_loader_task.apply_async(priority=10).get()
    if lazy_loading_successful:
        log(f'Lazy-loaded objects loaded into memory')
    else:
        log(f'ERROR: Lazy-loading unsuccessful, check celery logs')


# Root endpoint redirects to docs
@app.get("/")
async def redirect_docs():
    return RedirectResponse(url='/docs')
