import json

import pytest
from time import sleep

from graphai.api.celery_tasks.video import (
    retrieve_file_from_url_task,
    extract_and_sample_frames_task,
    extract_audio_task,
    compute_video_fingerprint_task,
)

################################################################
################################################################
# Unit tests                                                   #
################################################################
################################################################


################################################################
# /video/retrieve_url                                          #
################################################################


@pytest.mark.usefixtures('test_video_url', 'test_video_token')
def test__video_retrieve_url__retrieve_file_from_url_task__run_task(test_video_url, test_video_token):
    results = retrieve_file_from_url_task(test_video_url, True,
                                          force_token=test_video_token[:-4])

    assert isinstance(results, dict)
    assert 'token' in results
    assert results['token'] == test_video_token
    assert results['fresh']


################################################################
# /video/calculate_fingerprint                                 #
################################################################


@pytest.mark.usefixtures('test_video_token')
def test__video_calculate_fingerprint__compute_video_fingerprint_task__run_task(test_video_token):
    results = compute_video_fingerprint_task({'token': test_video_token})

    assert isinstance(results, dict)
    assert 'result' in results
    assert results['result'] == '6daa9b6b3360585affe8a60660914189'
    assert results['fresh']
    assert results['fp_token'] == test_video_token


################################################################
# /video/detect_slides                                         #
################################################################


@pytest.mark.usefixtures('test_video_token')
def test__video_detect_slides__extract_and_sample_frames_task__run_task(test_video_token):
    results = extract_and_sample_frames_task(test_video_token)

    assert isinstance(results, dict)
    assert 'result' in results
    assert results['result'] == test_video_token + '_all_frames'
    assert results['fresh']


################################################################
# /video/extract_audio                                         #
################################################################


@pytest.mark.usefixtures('test_video_token')
def test__video_extract_audio__extract_audio_task__run_task(test_video_token):
    results = extract_audio_task(test_video_token)

    assert isinstance(results, dict)
    assert 'token' in results
    assert results['token'] == test_video_token + '_audio.ogg'
    assert results['fresh']


################################################################
################################################################
# Integration tests                                            #
################################################################
################################################################


################################################################
# Slide extraction and OCR                                     #
################################################################


# The mark is necessary because the celery_worker fixture uses JSON for serialization by default
@pytest.mark.celery(accept_content=['pickle', 'json'], result_serializer='pickle', task_serializer='pickle')
@pytest.mark.usefixtures('test_video_url')
def test__video_detect_slides__detect_slides__integration(fixture_app, celery_worker, test_video_url, timeout=30):
    # First retrieving the video (without `force` in order to use cached results if available)
    response = fixture_app.post('/video/retrieve_url',
                                data=json.dumps({"url": test_video_url, "force": True}),
                                timeout=timeout)

    assert response.status_code == 200
    # Parse resulting task id
    task_id = response.json()['task_id']

    # Waiting for task chain to succeed
    current_status = 'PENDING'
    n_tries = 0
    while current_status == 'PENDING' and n_tries < 20:
        # Wait a few seconds
        sleep(5)
        # Now get status
        response = fixture_app.get(f'/video/retrieve_url/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Checking video token response
    video_token_response = response.json()

    assert isinstance(video_token_response, dict)
    assert 'task_result' in video_token_response
    assert video_token_response['task_status'] == 'SUCCESS'
    assert video_token_response['task_result']['token'] is not None

    video_token = video_token_response['task_result']['token']

    # Then detecting slides
    response = fixture_app.post('/video/detect_slides',
                                data=json.dumps({"token": video_token, "force": True}),
                                timeout=timeout)

    assert response.status_code == 200
    # Parse resulting task id
    task_id = response.json()['task_id']

    # Waiting for task chain to succeed
    current_status = 'PENDING'
    n_tries = 0
    while current_status == 'PENDING' and n_tries < 50:
        # Wait a few seconds
        sleep(5)
        # Now get status
        response = fixture_app.get(f'/video/detect_slides/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Checking extracted slides
    slides = response.json()

    assert isinstance(slides, dict)
    assert 'task_result' in slides
    assert slides['task_status'] == 'SUCCESS'
    assert slides['task_result']['fresh'] is True
    assert isinstance(slides['task_result']['slide_tokens'], dict)
    assert video_token in slides['task_result']['slide_tokens']['1']['token']
    assert '.png' in slides['task_result']['slide_tokens']['1']['token']
    assert len(slides['task_result']['slide_tokens']) > 5

    # Re-detecting slides, which should yield a cache hit this time
    response = fixture_app.post('/video/detect_slides',
                                data=json.dumps({"token": video_token}),
                                timeout=timeout)

    assert response.status_code == 200
    # Parse resulting task id
    task_id = response.json()['task_id']

    # Waiting for task chain to succeed
    current_status = 'PENDING'
    n_tries = 0
    while current_status == 'PENDING' and n_tries < 50:
        # Wait a few seconds
        sleep(5)
        # Now get status
        response = fixture_app.get(f'/video/detect_slides/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Checking extracted slides
    slides = response.json()

    assert isinstance(slides, dict)
    assert 'task_result' in slides
    assert slides['task_status'] == 'SUCCESS'
    assert slides['task_result']['fresh'] is False
    assert isinstance(slides['task_result']['slide_tokens'], dict)
    assert video_token in slides['task_result']['slide_tokens']['1']['token']
    assert '.png' in slides['task_result']['slide_tokens']['1']['token']
    assert len(slides['task_result']['slide_tokens']) > 5

    first_slide = slides['task_result']['slide_tokens']['1']['token']

    # Finally, performing OCR on the first slide
    # setting force to False in order to require a fingerprint lookup
    response = fixture_app.post('/image/extract_text',
                                data=json.dumps({"token": first_slide, "method": "google", "force": False}),
                                timeout=timeout)

    assert response.status_code == 200
    # Parse resulting task id
    task_id = response.json()['task_id']

    # Waiting for task chain to succeed
    current_status = 'PENDING'
    n_tries = 0
    while current_status == 'PENDING' and n_tries < 10:
        # Wait a few seconds
        sleep(3)
        # Now get status
        response = fixture_app.get(f'/image/extract_text/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Checking OCR results
    slide_ocr = response.json()

    assert isinstance(slide_ocr, dict)
    assert 'task_result' in slide_ocr
    assert slide_ocr['task_status'] == 'SUCCESS'
    assert isinstance(slide_ocr['task_result']['result'], list)
    assert len(slide_ocr['task_result']['result']) == 2
    assert slide_ocr['task_result']['language'] == 'en'
    assert 'value capture' in slide_ocr['task_result']['result'][0]['text'].lower()


################################################################
# Audio extraction and language detection                      #
################################################################


@pytest.mark.celery(accept_content=['pickle', 'json'], result_serializer='pickle', task_serializer='pickle')
@pytest.mark.usefixtures('test_video_url')
def test__video_extract_audio__extract_audio__integration(fixture_app, celery_worker, test_video_url, timeout=30):
    # First retrieving the video (without `force` in order to use cached results if available)
    response = fixture_app.post('/video/retrieve_url',
                                data=json.dumps({"url": test_video_url, "playlist": True}),
                                timeout=timeout)

    assert response.status_code == 200
    # Parse resulting task id
    task_id = response.json()['task_id']

    # Waiting for task chain to succeed
    current_status = 'PENDING'
    n_tries = 0
    while current_status == 'PENDING' and n_tries < 20:
        # Wait a few seconds
        sleep(5)
        # Now get status
        response = fixture_app.get(f'/video/retrieve_url/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Checking video token response
    video_token_response = response.json()

    assert isinstance(video_token_response, dict)
    assert 'task_result' in video_token_response
    assert video_token_response['task_status'] == 'SUCCESS'
    assert video_token_response['task_result']['token'] is not None

    video_token = video_token_response['task_result']['token']

    # Then detecting audio
    response = fixture_app.post('/video/extract_audio',
                                data=json.dumps({"token": video_token, "force": True}),
                                timeout=timeout)

    assert response.status_code == 200
    # Parse resulting task id
    task_id = response.json()['task_id']

    # Waiting for task chain to succeed
    current_status = 'PENDING'
    n_tries = 0
    while current_status == 'PENDING' and n_tries < 40:
        # Wait a few seconds
        sleep(5)
        # Now get status
        response = fixture_app.get(f'/video/extract_audio/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Checking extracted audio
    audio_result = response.json()

    assert isinstance(audio_result, dict)
    assert 'task_result' in audio_result
    assert audio_result['task_status'] == 'SUCCESS'
    assert audio_result['task_result']['fresh'] is True
    assert video_token in audio_result['task_result']['token']
    assert '.ogg' in audio_result['task_result']['token']
    assert 450 < audio_result['task_result']['duration'] < 470

    audio_token = audio_result['task_result']['token']

    # Finally, performing language detection on the extracted audio
    response = fixture_app.post('/voice/detect_language',
                                data=json.dumps({"token": audio_token, "force": True}),
                                timeout=timeout)

    assert response.status_code == 200
    # Parse resulting task id
    task_id = response.json()['task_id']

    # Waiting for task chain to succeed
    current_status = 'PENDING'
    n_tries = 0
    while current_status == 'PENDING' and n_tries < 30:
        # Wait a few seconds
        sleep(3)
        # Now get status
        response = fixture_app.get(f'/voice/detect_language/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Checking language detection results
    audio_lang_result = response.json()

    assert isinstance(audio_lang_result, dict)
    assert 'task_result' in audio_lang_result
    assert audio_lang_result['task_status'] == 'SUCCESS'
    assert audio_lang_result['task_result']['fresh'] is True
    assert audio_lang_result['task_result']['language'] == 'en'
