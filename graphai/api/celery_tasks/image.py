from celery import shared_task, group

from graphai.api.celery_tasks.common import fingerprint_lookup_retrieve_from_db, fingerprint_lookup_parallel, \
    fingerprint_lookup_callback
from graphai.api.common.video import slide_db_manager, file_management_config
from graphai.core.common.video import perceptual_hash_image


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.slide_fingerprint', ignore_result=False,
             db_manager=slide_db_manager, file_manager=file_management_config)
def compute_slide_fingerprint_task(self, token, force=False):
    # Checking for existing cached results
    existing_slide = self.db_manager.get_details(token, cols=['fingerprint'])
    if existing_slide is None:
        return {
            'result': None,
            'fresh': False
        }
    if not force and existing_slide['fingerprint'] is not None:
        return {
            'result': existing_slide['fingerprint'],
            'fresh': False
        }
    slide_with_path = self.file_manager.generate_filepath(token)
    fingerprint = perceptual_hash_image(slide_with_path)
    if fingerprint is None:
        return {
            'result': None,
            'fresh': False
        }
    return {
        'result': fingerprint,
        'fresh': True
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.slide_fingerprint_callback', ignore_result=False,
             db_manager=slide_db_manager)
def compute_slide_fingerprint_callback_task(self, results, token):
    if results['fresh']:
        self.db_manager.insert_or_update_details(
            token,
            {
                'fingerprint': results['result'],
            }
        )
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.slide_fingerprint_find_closest_retrieve_from_db', ignore_result=False,
             db_manager=slide_db_manager)
def slide_fingerprint_find_closest_retrieve_from_db_task(self, results, token):
    return fingerprint_lookup_retrieve_from_db(results, token, self.db_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.slide_fingerprint_find_closest_parallel', ignore_result=False,
             db_manager=slide_db_manager)
def slide_fingerprint_find_closest_parallel_task(self, input_dict, i, n_total, min_similarity=1):
    return fingerprint_lookup_parallel(input_dict, i, n_total, min_similarity, data_type='image')


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.slide_fingerprint_find_closest_callback', ignore_result=False,
             db_manager=slide_db_manager)
def slide_fingerprint_find_closest_callback_task(self, results_list, original_token):
    return fingerprint_lookup_callback(results_list, original_token, self.db_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.retrieve_slide_fingerprint_final_callback', ignore_result=False)
def retrieve_slide_fingerprint_callback_task(self, results):
    # Returning the fingerprinting results, which is the part of this task whose results are sent back to the user.
    return results['fp_results']


def compute_slide_fingerprint_master(token, force=False, min_similarity=1, n_jobs=8):
    task = (compute_slide_fingerprint_task.s(token, force) |
            compute_slide_fingerprint_callback_task.s(token) |
            slide_fingerprint_find_closest_retrieve_from_db_task.s(token) |
            group(slide_fingerprint_find_closest_parallel_task.s(i, n_jobs, min_similarity) for i in range(n_jobs)) |
            slide_fingerprint_find_closest_callback_task.s(token) |
            retrieve_slide_fingerprint_callback_task.s()
            ).apply_async(priority=2)
    return {'id': task.id}