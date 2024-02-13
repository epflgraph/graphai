from fastapi import FastAPI, Depends, APIRouter
from fastapi.security import OAuth2PasswordRequestForm
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
from graphai.api.routers.auth import *

from graphai.api.celery_tasks.text import text_init_task
from graphai.api.celery_tasks.video import video_init_task


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialise FastAPI
app = FastAPI(
    title="EPFL Graph AI API",
    description="This API offers several tools related with AI in the context of the EPFL Graph project, "
                "such as automatized concept detection from a given text.",
    version="0.2.1"
)


unauthenticated_router = APIRouter()
authenticated_router = APIRouter(dependencies=[Depends(oauth2_scheme)])


# Root endpoint redirects to docs
@unauthenticated_router.get("/")
async def redirect_docs():
    return RedirectResponse(url='/docs')


@unauthenticated_router.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user_dict = fake_users_db.get(form_data.username)
    if not user_dict:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    user = UserInDB(**user_dict)
    hashed_password = fake_hash_password(form_data.password)
    if not hashed_password == user.hashed_password:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    return {"access_token": user.username, "token_type": "bearer"}


@authenticated_router.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


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
