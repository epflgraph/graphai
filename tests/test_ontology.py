import pytest

from graphai.api.celery_tasks.ontology import (
    get_concept_category_closest_task,
    break_up_cluster_task
)


@pytest.mark.usefixtures('ballistics', 'bullet')
def test__ontology_compute_graph_nearest_category__get_concept_category_closest__run_task(ballistics, bullet):
    # Call the task
    closest_categories = get_concept_category_closest_task.run(bullet, "linear", (1, 4), 3, True, 1)

    # Assert that the results are correct
    assert isinstance(closest_categories, dict)
    assert 'scores' in closest_categories
    assert 'parent_category' in closest_categories
    assert closest_categories['parent_category'] == 'military-technology'
    assert closest_categories['scores'][0]['category_id'] == 'ballistics'
    assert len(closest_categories['scores'][0]['clusters']) == 1

    closest_categories = get_concept_category_closest_task.run(ballistics, "linear", (1, 4), 3, True, 2)

    # Assert that the results are correct
    assert isinstance(closest_categories, dict)
    assert 'scores' in closest_categories
    assert 'parent_category' in closest_categories
    assert closest_categories['parent_category'] == 'military-technology'
    assert closest_categories['scores'][0]['category_id'] == 'ballistics'
    assert len(closest_categories['scores'][0]['clusters']) >= 1


@pytest.mark.usefixtures('cluster_to_break')
def test__ontology_break_up_cluster__break_up_cluster__run_task(cluster_to_break):
    # Call the task
    broken_up_clusters = break_up_cluster_task.run(cluster_to_break, 2)

    # Assert that the results are correct
    assert isinstance(broken_up_clusters, dict)
    assert 'results' in broken_up_clusters
    assert len(broken_up_clusters['results']) == 1
    assert broken_up_clusters['results'][0]['n_clusters'] == 2
    assert (broken_up_clusters['results'][0]['clusters'] is None
            or len(broken_up_clusters['results'][0]['clusters']) == 2)
