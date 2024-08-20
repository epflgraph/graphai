from celery import chain, group

from graphai.celery.text.tasks import (
    extract_keywords_task,
    wikisearch_task,
    compute_scores_task,
    draw_ontology_task,
    draw_graph_task,
)


def keywords(raw_text, use_nltk):
    job = chain(extract_keywords_task.s(raw_text, use_nltk=use_nltk))
    return job.apply_async(priority=10).get(timeout=10)


def wikify_text(
    text,
    method,
    restrict_to_ontology,
    graph_score_smoothing,
    ontology_score_smoothing,
    keywords_score_smoothing,
    aggregation_coef,
    filtering_threshold,
    filtering_min_agreement_fraction,
    refresh_scores,
):
    n = 16
    job = chain(
        extract_keywords_task.s(text),
        group(wikisearch_task.s(fraction=(i / n, (i + 1) / n), method=method) for i in range(n)),
        compute_scores_task.s(
            restrict_to_ontology=restrict_to_ontology,
            graph_score_smoothing=graph_score_smoothing,
            ontology_score_smoothing=ontology_score_smoothing,
            keywords_score_smoothing=keywords_score_smoothing,
            aggregation_coef=aggregation_coef,
            filtering_threshold=filtering_threshold,
            filtering_min_agreement_fraction=filtering_min_agreement_fraction,
            refresh_scores=refresh_scores,
        )
    )
    results = job.apply_async(priority=10).get(timeout=300)

    return results.to_dict(orient='records')


def wikify_keywords(
    keyword_list,
    method,
    restrict_to_ontology,
    graph_score_smoothing,
    ontology_score_smoothing,
    keywords_score_smoothing,
    aggregation_coef,
    filtering_threshold,
    filtering_min_agreement_fraction,
    refresh_scores,
):
    n = 16
    job = chain(
        group(wikisearch_task.s(keyword_list, fraction=(i / n, (i + 1) / n), method=method) for i in range(n)),
        compute_scores_task.s(
            restrict_to_ontology=restrict_to_ontology,
            graph_score_smoothing=graph_score_smoothing,
            ontology_score_smoothing=ontology_score_smoothing,
            keywords_score_smoothing=keywords_score_smoothing,
            aggregation_coef=aggregation_coef,
            filtering_threshold=filtering_threshold,
            filtering_min_agreement_fraction=filtering_min_agreement_fraction,
            refresh_scores=refresh_scores,
        )
    )
    results = job.apply_async(priority=10).get(timeout=300)

    return results.to_dict(orient='records')


def wikify_ontology_svg(results, level):
    job = draw_ontology_task.s(results, level=level)
    job.apply_async(priority=10).get(timeout=10)


def wikify_graph_svg(results, concept_score_threshold, edge_threshold, min_component_size):
    job = draw_graph_task.s(results, concept_score_threshold=concept_score_threshold, edge_threshold=edge_threshold, min_component_size=min_component_size)
    job.apply_async(priority=10).get(timeout=10)
