from graphai.core.common.ontology_utils.data import OntologyData
import pandas as pd
import numpy as np


def predict_all(model: OntologyData, x: list, avg='linear', coeffs=(1, 4), top_down=True, n=5, **kwargs):
    y = list()
    for p in x:
        best_cats, best_scores, selected_d3_category, best_clusters = \
            model.get_concept_closest_category(p, avg=avg, coeffs=coeffs, top_n=n, use_depth_3=top_down,
                                               return_clusters=None,
                                               adaptive_threshold=kwargs.get('adaptive_threshold', None))
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


def get_direct_child_to_parent_dict(model: OntologyData):
    child_to_parent = model.get_category_to_category()
    child_to_parent = {d['child_id']: d['parent_id'] for d in child_to_parent}
    return child_to_parent


def convert_to_level_3(child_to_parent, labels):
    return [child_to_parent[label] for label in labels]


def get_errors_at_k(res, concepts, labels, k):
    results_to_evaluate = get_top_k_list_of_lists(res, k)
    error_indices = [i for i in range(len(results_to_evaluate)) if labels[i] not in results_to_evaluate[i]]
    return [concepts[i] for i in error_indices]


def get_max_accuracy(res):
    return len([x for x in res if len(x) > 0]) / len(res)


def find_problem_areas_using_error_ratio(results, test_category_concept, train_category_concept,
                                         category_cluster, test_cluster_concept, k=5, ratio=1.0):
    concepts = test_category_concept.to_id.values.tolist()
    labels = test_category_concept.from_id.values.tolist()
    errors_at_k = get_errors_at_k(results, concepts, labels, k)
    errors_at_k_categories = test_category_concept.loc[
        test_category_concept.to_id.apply(lambda x: x in errors_at_k)
    ].copy()
    errors_at_k_cat_cluster_concept = (
        pd.merge(
            category_cluster, test_cluster_concept.loc[test_cluster_concept.to_id.apply(lambda x: x in errors_at_k)],
            left_on='to_id', right_on='from_id', suffixes=('_x', '_y'), how='inner'
        )[['from_id_x', 'to_id_x', 'to_id_y']].
        rename(columns={'from_id_x': 'category_id', 'to_id_x': 'cluster_id', 'to_id_y': 'concept_id'})
    )
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
    problem_areas = errors_at_k_categories.loc[errors_at_k_categories.error_ratio >= ratio].category_id.values.tolist()
    errors_at_k_cat_cluster_concept = errors_at_k_cat_cluster_concept.loc[
        errors_at_k_cat_cluster_concept.category_id.apply(lambda x: x in problem_areas)
    ]
    errors_at_k_cat_cluster_concept = errors_at_k_cat_cluster_concept.groupby(
        ['category_id', 'cluster_id']
    ).count().reset_index()
    return problem_areas, errors_at_k, errors_at_k_categories, errors_at_k_cat_cluster_concept


def find_all_problem_areas(n_rounds=20, sampling_method='weighted',
                           avg='log', coeffs=(1, 10), top_down=False, cutoff=0.5):
    all_problem_areas = list()
    accuracy_at_1_values = list()
    accuracy_at_5_values = list()
    for i in range(n_rounds):
        print(f'Starting round {i + 1} out of {n_rounds}')
        ontology_data = OntologyData(test_mode=True, test_ratio=0.3, random_state=i, sampling_method=sampling_method)
        ontology_data.load_data()
        test_category_concept = ontology_data.get_test_category_concept()
        test_concepts = test_category_concept['to_id'].values.tolist()
        test_labels = test_category_concept['from_id'].values.tolist()
        results = predict_all(ontology_data, test_concepts, avg=avg, coeffs=coeffs, top_down=top_down, n=5)
        accuracy_at_1_values.append(eval_at_k(results, test_labels, 1))
        accuracy_at_5_values.append(eval_at_k(results, test_labels, 5))
        problem_areas, errors_at_k, errors_at_k_categories, errors_at_k_cat_cluster_concept = (
            find_problem_areas_using_error_ratio(results, test_category_concept,
                                                 ontology_data.get_category_concept_table(),
                                                 ontology_data.get_category_cluster_table(),
                                                 ontology_data.get_test_cluster_concept(),
                                                 k=5, ratio=cutoff)
        )
        errors_at_k_categories = errors_at_k_categories.assign(random_state=i)
        all_problem_areas.append(
            errors_at_k_categories.loc[
                errors_at_k_categories.category_id.apply(lambda x: x in problem_areas)
            ]
        )
    all_problem_areas = pd.concat(all_problem_areas, axis=0)
    all_problem_areas_count = all_problem_areas[['category_id', 'error_ratio']].copy()
    all_problem_areas_count = pd.merge(
        all_problem_areas_count.groupby('category_id').count().reset_index().rename(columns={'error_ratio': 'count'}),
        all_problem_areas_count.groupby('category_id').mean().reset_index().rename(columns={'error_ratio': 'mean_rate'}),
        on='category_id'
    )
    return all_problem_areas, all_problem_areas_count, accuracy_at_1_values, accuracy_at_5_values


def main():
    for avg in ['linear', 'log', 'adaptive']:
        for sampling_method in ['simple', 'weighted']:
            all_problem_areas, all_problem_areas_count, accuracy_at_1_values, accuracy_at_5_values = (
                find_all_problem_areas(avg=avg, sampling_method=sampling_method)
            )
            all_problem_areas.to_csv(f'all_{avg}_{sampling_method}.csv', index=False)
            all_problem_areas_count.to_csv(f'all_counts_{avg}_{sampling_method}.csv', index=False)

            print(f'{avg}:')
            print(f'{sampling_method}:')
            print('Accuracy @ 1:')
            print(f'{np.mean(accuracy_at_1_values)} ± {np.std(accuracy_at_1_values)}')
            print('Accuracy @ 5:')
            print(f'{np.mean(accuracy_at_5_values)} ± {np.std(accuracy_at_5_values)}')


if __name__ == '__main__':
    main()
