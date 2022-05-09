import numpy as np

from concept_detection.test.courses.compare import *
from concept_detection.text.io import pprint

methods = ['wikipedia-api', 'es-base', 'es-score']

comparative = compare_wikify_course_descriptions_api_db(methods, limit=2)
pprint(comparative, only_first=True)

for method in methods:
    method_pair_stats = [stat[method] for stat in comparative['pair_stats']]
    method_page_stats = [stat[method] for stat in comparative['page_stats']]

    plot_confs(method_pair_stats, 'course_id', f'Pairs ({method})')
    plot_confs(method_page_stats, 'course_id', f'Pages ({method})')

    print(f'Method {method}')
    for metric in ['p', 'r', 'f-score']:
        pairs_metric = np.mean([conf[metric] for conf in method_pair_stats])
        pages_metric = np.mean([conf[metric] for conf in method_page_stats])
        print(f'Average {metric} for pairs: {pairs_metric:.4f}')
        print(f'Average {metric} for pages: {pages_metric:.4f}')
