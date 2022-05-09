import numpy as np

from concept_detection.test.courses.compare import *
from concept_detection.text.io import pprint

comparative = compare_course_descriptions_api_db(limit=10)

pprint(comparative, only_first=True)

plot_confs(comparative['pair_stats'], 'course_id', f'Pairs')
plot_confs(comparative['page_stats'], 'course_id', f'Pages')

for metric in ['p', 'r', 'f-score']:
    pairs_metric = np.mean([conf[metric] for conf in comparative['pair_stats']])
    pages_metric = np.mean([conf[metric] for conf in comparative['page_stats']])
    print(f'Average {metric} for pairs: {pairs_metric}')
    print(f'Average {metric} for pages: {pages_metric}')
