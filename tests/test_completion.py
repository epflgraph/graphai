import json

import pytest


@pytest.mark.celery(accept_content=['pickle', 'json'], result_serializer='pickle', task_serializer='pickle')
@pytest.mark.usefixtures('slides_and_raw_concepts')
def test__summarization__slide_subset__integration(fixture_app, celery_worker, slides_and_raw_concepts,
                                                   timeout=30):
    # The celery_worker object is necessary for async tasks, otherwise the status will be permanently stuck on
    # PENDING.

    # First, we call the summary endpoint with force=True to test the full task pipeline working
    response = fixture_app.post('/completion/slide_subset',
                                data=json.dumps({"slides": slides_and_raw_concepts}),
                                timeout=timeout)
    # Check status code is successful
    assert response.status_code == 200
    # Parse result
    results = response.json()
    # Check returned value
    assert isinstance(results, dict)
    assert 'subset' in results
    assert sorted(results['subset']) == [2, 9, 16, 24, 31, 34, 38]
