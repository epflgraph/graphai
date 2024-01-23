from graphai.core.common.ontology_utils.data import OntologyData
import pandas as pd
import numpy as np


def predict_all(model: OntologyData, x: list, avg='linear', coeffs=(1, 4), top_down=True, n=5):
    y = list()
    for p in x:
        best_cats, best_scores, selected_d3_category, best_clusters = \
            model.get_concept_closest_category(p, avg=avg, coeffs=coeffs, top_n=n, use_depth_3=top_down,
                                               return_clusters=None)
        if best_cats is None:
            best_cats = list()
        y.append(best_cats)
    return y


def get_top_k_list_of_lists(res, k):
    max_k = max([len(x) for x in res])
    if k > max_k:
        print('Maximum n is %d, cannot return evaluation metrics @ %d' % (max_k, k))
        return None
    return [set(x[:k]) for x in res]


def eval_at_k(res, labels, k):
    results_to_evaluate = get_top_k_list_of_lists(res, k)
    n_correct = sum([1 if labels[i] in results_to_evaluate[i] else 0 for i in range(len(results_to_evaluate))])
    n_total = len(labels)
    acc = n_correct / n_total
    return acc


def get_errors_at_k(res, concepts, labels, k):
    results_to_evaluate = get_top_k_list_of_lists(res, k)
    error_indices = [i for i in range(len(results_to_evaluate)) if labels[i] not in results_to_evaluate[i]]
    return [concepts[i] for i in error_indices]


def get_max_accuracy(res):
    return len([x for x in res if len(x) > 0]) / len(res)


def find_problem_areas_using_error_ratio(results, test_category_concept, train_category_concept, k=5, ratio=1.0):
    concepts = test_category_concept.to_id.values.tolist()
    labels = test_category_concept.from_id.values.tolist()
    errors_at_k = get_errors_at_k(results, concepts, labels, k)
    errors_at_k_categories = test_category_concept.loc[
        test_category_concept.to_id.apply(lambda x: x in errors_at_k)
    ].copy()
    errors_at_k_categories = errors_at_k_categories.assign(count=1)[['from_id', 'count']].groupby(
        'from_id').sum().reset_index()
    category_full_sizes = (
        pd.concat([test_category_concept, train_category_concept]).assign(count=1)[['from_id', 'count']].
        groupby('from_id').sum().reset_index()
    )
    category_test_sizes = test_category_concept.copy().assign(count=1)[['from_id', 'count']].groupby(
        'from_id').sum().reset_index()
    errors_at_k_categories = pd.merge(errors_at_k_categories, category_full_sizes, how='inner',
                                      on='from_id', suffixes=('_errors', '_total'))
    errors_at_k_categories = pd.merge(errors_at_k_categories, category_test_sizes, how='inner',
                                      on='from_id').rename(columns={'count': 'count_test'})
    errors_at_k_categories['error_ratio'] = (
            errors_at_k_categories['count_errors'] / errors_at_k_categories['count_test']
    )
    errors_at_k_categories = errors_at_k_categories.rename(columns={'from_id': 'category_id'})
    problem_areas = errors_at_k_categories.loc[errors_at_k_categories.error_ratio >= ratio].from_id.values.tolist()
    return problem_areas, errors_at_k, errors_at_k_categories


def find_all_problem_areas(n_rounds=20, avg='log', coeffs=(1, 10), top_down=False, cutoff=0.5):
    all_problem_areas = list()
    for i in range(n_rounds):
        ontology_data = OntologyData(test_mode=True, test_ratio=0.3, random_state=i)
        ontology_data.load_data()
        test_category_concept = ontology_data.get_test_category_concept()
        test_concepts = test_category_concept['to_id'].values.tolist()
        results = predict_all(ontology_data, test_concepts, avg=avg, coeffs=coeffs, top_down=top_down, n=5)
        problem_areas, errors_at_k, errors_at_k_categories = (
            find_problem_areas_using_error_ratio(results, test_category_concept,
                                                 ontology_data.get_category_concept_table(), k=5, ratio=cutoff)
        )
        errors_at_k_categories = errors_at_k_categories.assign(random_state=i)
        all_problem_areas.append(errors_at_k_categories.loc[
                                     errors_at_k_categories.category_id.apply(lambda x: x in problem_areas)
                                 ])
    all_problem_areas = pd.concat(all_problem_areas, axis=0)
    all_problem_areas_count = all_problem_areas[['category_id', 'error_rate']].copy()
    all_problem_areas_count = pd.merge(
        all_problem_areas_count.groupby('category_id').count().reset_index().rename(columns={'error_rate': 'count'}),
        all_problem_areas_count.groupby('category_id').mean().reset_index().rename(columns={'error_rate': 'mean_rate'}),
        on='category_id'
    )
    return all_problem_areas, all_problem_areas_count
