from celery import chain, group

from graphai.core.common.logging import get_logger

import graphai.celery.text.tasks as tasks

logger = get_logger('graphai.celery.text.jobs')


def keywords(raw_text, use_nltk):
    job = chain(tasks.extract_keywords_task.s(raw_text, use_nltk=use_nltk))
    async_result = job.apply_async(priority=10)
    logger.info('🔑 Submitted keywords job', task_id=async_result.id, use_nltk=use_nltk)
    return async_result.get(timeout=300000)


def wiki_search(search_term, limit):
    job = chain(tasks.wiki_search_task.s(search_term, limit=limit))
    async_result = job.apply_async(priority=10)
    logger.info('🔍 Submitted wiki_search job', task_id=async_result.id, search_term=search_term, limit=limit)
    return async_result.get(timeout=30)


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
    async_result = job.apply_async(priority=10)
    logger.info(
        '🚀 Submitted wikify_text job',
        task_id=async_result.id,
        method=method,
        shards=n,
        restrict_to_ontology=restrict_to_ontology,
        score_smoothing=score_smoothing,
    )
    results = async_result.get(timeout=300000)
    logger.info('✅ wikify_text job completed', task_id=async_result.id, num_results=len(results) if hasattr(results, '__len__') else None)

    return results.records if hasattr(results, 'records') else results.to_dict(orient='records')


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
    async_result = job.apply_async(priority=10)
    logger.info(
        '🚀 Submitted wikify_keywords job',
        task_id=async_result.id,
        method=method,
        shards=n,
        num_keywords=len(keyword_list),
        restrict_to_ontology=restrict_to_ontology,
    )
    results = async_result.get(timeout=300000)
    logger.info('✅ wikify_keywords job completed', task_id=async_result.id, num_results=len(results) if hasattr(results, '__len__') else None)

    return results.records if hasattr(results, 'records') else results.to_dict(orient='records')


def wikify_ontology_svg(results, level):
    job = tasks.draw_ontology_task.s(results, level=level)
    async_result = job.apply_async(priority=10)
    logger.info('🎨 Submitted wikify_ontology_svg job', task_id=async_result.id, num_results=len(results), level=level)
    async_result.get(timeout=300000)
    logger.info('✅ wikify_ontology_svg job completed', task_id=async_result.id)


def wikify_graph_svg(results, concept_score_threshold, edge_threshold, min_component_size):
    job = tasks.draw_graph_task.s(results, concept_score_threshold=concept_score_threshold, edge_threshold=edge_threshold, min_component_size=min_component_size)
    async_result = job.apply_async(priority=10)
    logger.info(
        '🕸️ Submitted wikify_graph_svg job',
        task_id=async_result.id,
        num_results=len(results),
        concept_score_threshold=concept_score_threshold,
        edge_threshold=edge_threshold,
        min_component_size=min_component_size,
    )
    async_result.get(timeout=300000)
    logger.info('✅ wikify_graph_svg job completed', task_id=async_result.id)


def generate_exercise(data):
    job = chain(tasks.generate_exercise_task.s(data))
    async_result = job.apply_async(priority=10)
    logger.info('🎓 Submitted generate_exercise job', task_id=async_result.id)
    result = async_result.get(timeout=300000)
    logger.info('✅ generate_exercise job completed', task_id=async_result.id)
    return result
