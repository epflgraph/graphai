import time
import numpy as np

from concept_detection.test.courses.compare import *
from concept_detection.text.io import pprint

st = time.time()

methods = ['wikipedia-api', 'es-base', 'es-score']
limit = 10

comparative = compare_wikify_course_descriptions_api_db(methods, limit=limit)
pprint(comparative, only_first=True)

for how in ['pair', 'page']:
    for method in methods:
        plot_confs(comparative['stats'][how][method], 'course_id', f'{how} - {method}')

for how in ['pair', 'page']:
    for metric in ['p', 'r', 'f-score']:
        for method in methods:
            value = np.mean([stat[metric] for stat in comparative['stats'][how][method]])
            print(f'Average {metric} for {how} - {method}: {value:.4f}')

ft = time.time()
print(f'Total time for {limit} courses: {ft - st:.2f}s (each ~{(ft - st) / limit:.2f}s, full execution ~{2000 * (ft - st) / (limit * 3600):.2f}h).')
