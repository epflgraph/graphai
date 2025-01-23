from celery import chain, group

import graphai.celery.text.tasks as tasks


def keywords(raw_text, use_nltk):
    job = chain(tasks.extract_keywords_task.s(raw_text, use_nltk=use_nltk))
    return job.apply_async(priority=10).get(timeout=10)


def wikify_text(
    text,
    method,
    restrict_to_ontology,
    score_smoothing,
    aggregation_coef,
    filtering_threshold,
    refresh_scores,
):
    n = 16
    job = chain(
        tasks.extract_keywords_task.s(text),
        group(tasks.wikisearch_task.s(fraction=(i / n, (i + 1) / n), method=method) for i in range(n)),
        tasks.compute_scores_task.s(
            restrict_to_ontology=restrict_to_ontology,
            score_smoothing=score_smoothing,
            aggregation_coef=aggregation_coef,
            filtering_threshold=filtering_threshold,
            refresh_scores=refresh_scores,
        )
    )
    results = job.apply_async(priority=10).get(timeout=300)

    return results.to_dict(orient='records')


def wikify_keywords(
    keyword_list,
    method,
    restrict_to_ontology,
    score_smoothing,
    aggregation_coef,
    filtering_threshold,
    refresh_scores,
):
    n = 16
    job = chain(
        group(tasks.wikisearch_task.s(keyword_list, fraction=(i / n, (i + 1) / n), method=method) for i in range(n)),
        tasks.compute_scores_task.s(
            restrict_to_ontology=restrict_to_ontology,
            score_smoothing=score_smoothing,
            aggregation_coef=aggregation_coef,
            filtering_threshold=filtering_threshold,
            refresh_scores=refresh_scores,
        )
    )
    results = job.apply_async(priority=10).get(timeout=300)

    return results.to_dict(orient='records')


def wikify_ontology_svg(results, level):
    job = tasks.draw_ontology_task.s(results, level=level)
    job.apply_async(priority=10).get(timeout=10)


def wikify_graph_svg(results, concept_score_threshold, edge_threshold, min_component_size):
    job = tasks.draw_graph_task.s(results, concept_score_threshold=concept_score_threshold, edge_threshold=edge_threshold, min_component_size=min_component_size)
    job.apply_async(priority=10).get(timeout=10)


def generate_exercise(data):
    job = chain(tasks.generate_exercise_task.s(data))
    return job.apply_async(priority=10).get(timeout=60)
