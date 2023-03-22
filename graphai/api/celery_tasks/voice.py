from celery import shared_task
from graphai.api.common.video import remove_silence_doublesided, perceptual_hash_audio

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.audio_fingerprint', ignore_result=False)
def compute_audio_fingerprint_task(self, filename, force=False, threshold=0.0):
    nosilence_token, fresh, duration = remove_silence_doublesided(filename, force=force, threshold=threshold)
    if nosilence_token is None:
        return {
            'result': None,
            'fresh': False,
            'duration_nosilence': 0.0
        }
    fingerprint, decoded, new_duration = perceptual_hash_audio(nosilence_token)
    return {
        'result': fingerprint.decode('utf8'),
        'fresh': fresh,
        'duration_nosilence': new_duration
    }


def compute_audio_fingerprint_master(token, force=False, threshold=0.0):
    task = compute_audio_fingerprint_task.apply_async(args=[token, force, threshold], priority=2)
    return {'task_id': task.id}