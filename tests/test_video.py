import json

import pytest
from time import sleep

from graphai.api.celery_tasks.video import retrieve_file_from_url_task, \
    extract_and_sample_frames_task, extract_audio_task, compute_video_fingerprint_task

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
    results = retrieve_file_from_url_task(test_video_url, True, timeout=240, force=True,
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
    results = compute_video_fingerprint_task(test_video_token, True)

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
    results = extract_and_sample_frames_task(test_video_token, True)

    assert isinstance(results, dict)
    assert 'result' in results
    assert results['result'] == test_video_token + '_all_frames'
    assert results['fresh']


################################################################
# /video/extract_audio                                         #
################################################################


@pytest.mark.usefixtures('test_video_token')
def test__video_extract_audio__extract_audio_task__run_task(test_video_token):
    results = extract_audio_task(test_video_token, True)

    assert isinstance(results, dict)
    assert 'token' in results
    assert results['token'] == test_video_token + '_audio.ogg'
    assert results['fresh']


################################################################
################################################################
# Integration tests                                            #
################################################################
################################################################


# The mark is necessary because the celery_worker fixture uses JSON for serialization by default
@pytest.mark.celery(accept_content=['pickle', 'json'], result_serializer='pickle', task_serializer='pickle')
@pytest.mark.usefixtures('test_video_url')
def test__video_detect_slides__detect_slides__integration(fixture_app, celery_worker, test_video_url, timeout=30):
    # First retrieving the video
    response = fixture_app.post('/video/retrieve_url',
                                data=json.dumps({"url": test_video_url}),
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
    while current_status == 'PENDING' and n_tries < 40:
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
    assert video_token in slides['task_result']['slide_tokens']["1"]['token']
    assert '.png' in slides['task_result']['slide_tokens']["1"]['token']
    assert len(slides['task_result']['slide_tokens']) > 5
