from fastapi import APIRouter, Security
from celery import chain

from graphai.api.schemas.completion import (
    SlideSubsetRequest,
    SlideSubsetResponse
)

from graphai.api.celery_tasks.completion import (
    choose_best_subset_task
)
from graphai.api.routers.auth import get_current_active_user


router = APIRouter(
    prefix='/completion',
    tags=['completion'],
    responses={404: {'description': 'Not found'}},
    dependencies=[Security(get_current_active_user, scopes=['completion'])]
)


@router.post('/slide_subset', response_model=SlideSubsetResponse)
async def choose_best_subset(data: SlideSubsetRequest):
    slides_and_concepts = data.slides
    slides_and_concepts = {slide.number: slide.concepts for slide in slides_and_concepts}
    coverage = data.coverage
    min_freq = data.min_freq
    task_list = [choose_best_subset_task.s(slides_and_concepts, coverage, min_freq)]
    tasks = chain(task_list)
    results = tasks.apply_async(priority=6).get(timeout=300)
    return results
