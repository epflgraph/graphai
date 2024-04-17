from celery import shared_task
from graphai.core.translation.text_utils import find_best_slide_subset


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.choose_best_subset', ignore_result=False)
def choose_best_subset_task(self, slide_number_to_concepts, coverage=1.0, min_freq=2):
    slide_numbers = sorted(list(slide_number_to_concepts.keys()))
    slide_concept_list = [slide_number_to_concepts[n] for n in slide_numbers]
    cover, best_indices = find_best_slide_subset(slide_concept_list, coverage, True, min_freq)
    return {
        'subset': [slide_numbers[i] for i in best_indices]
    }
