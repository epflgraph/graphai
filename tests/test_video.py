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
