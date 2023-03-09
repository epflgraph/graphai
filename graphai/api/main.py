from fastapi import FastAPI
from fastapi.responses import RedirectResponse

# import graphai.api.routers.ontology as ontology_router
import graphai.api.routers.text as text_router
# import graphai.api.routers.video as video_router

from graphai.api.common.log import log

from graphai.api.common.graph import graph
from graphai.api.common.ontology import ontology

# from graphai.core.celery_utils.celery_utils import create_celery

# from graphai.api.routers.video import video_actor_list
from graphai.core.text.wikisearch import ws_actor_list

# Initialise FastAPI
app = FastAPI(
    title="EPFL Graph AI API",
    description="This API offers several tools related with AI in the context of the EPFL Graph project, "
                "such as automatized concept detection from a given text.",
    version="0.2.1"
)

# Include all routers in the app
# app.include_router(ontology_router.router)
app.include_router(text_router.router)
# app.include_router(video_router.router)
# app.celery_app = create_celery()
# celery_instance = app.celery_app


# On startup, we instantiate concepts graph and ontology, so they are held into memory
@app.on_event("startup")
async def instantiate_graph_and_ontology():
    log(f'Fetching concepts graph from database...')
    graph.fetch_from_db()
    log(f'Fetching ontology from database...')
    ontology.fetch_from_db()

    log(f'Instantiating actor lists...')
    ws_actor_list.instantiate_actors()
    # video_actor_list.instantiate_actors()


# On shutdown, we free ray actors so theuy can be garbage collected
@app.on_event("shutdown")
async def instantiate_graph_and_ontology():
    log(f'Freeing ray actors...')
    ws_actor_list.free_actors()
    # video_actor_list.free_actors()


# Root endpoint redirects to docs
@app.get("/")
async def redirect_docs():
    return RedirectResponse(url='/docs')
