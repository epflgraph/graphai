import numpy as np

from db_api import DBApi
from new_api import NewApi

from compare import *

db_api = DBApi()
new_api = NewApi()


def test_text_to_results(limit=None):
    course_descriptions = db_api.query_course_descriptions()
    course_ids = list(course_descriptions.keys())
    n_courses = len(course_ids)
    print(f'Got {n_courses} courses')

    db_wikified_course_descriptions = db_api.query_wikified_course_descriptions()

    t = {
        'results': [],
        'pair_confs': [],
        'page_confs': []
    }
    i = 0
    for course_id in course_ids:
        i += 1
        print(i)

        raw_text = course_descriptions[course_id]
        anchor_page_ids = db_api.course_anchor_page_ids(course_id)

        new_wikified_course_description = new_api.wikify(raw_text, anchor_page_ids)
        comp = compare(new_wikified_course_description, db_wikified_course_descriptions[course_id])

        t['results'].append({
            'course_id': course_id,
            'raw_text': raw_text,
            'new': comp['results_1'],
            'db': comp['results_2']
        })

        new_pairs = [(result['keywords'], result['page_id']) for result in comp['results_1']]
        db_pairs = [(result['keywords'], result['page_id']) for result in comp['results_2']]

        conf = confusion_matrix(new_pairs, db_pairs)
        conf['course_id'] = course_id
        t['pair_confs'].append(conf)

        new_page_ids = [result['page_id'] for result in comp['results_1']]
        db_page_ids = [result['page_id'] for result in comp['results_2']]

        conf = confusion_matrix(new_page_ids, db_page_ids)
        conf['course_id'] = course_id
        t['page_confs'].append(conf)

        if i == limit:
            break

    return t


t = test_text_to_results()

plot_confs(t['pair_confs'], 'course_id', f'results/course-descriptions-text-to-results-pairs-plot.png')
plot_confs(t['page_confs'], 'course_id', f'results/course-descriptions-text-to-results-pages-plot.png')

save(t['results'], f'results/course-descriptions-text-to-results-results.json')

for metric in ['p', 'r', 'f-score']:
    pairs_metric = np.mean([conf[metric] for conf in t['pair_confs']])
    pages_metric = np.mean([conf[metric] for conf in t['page_confs']])
    print(f'Average {metric} for pairs: {pairs_metric}')
    print(f'Average {metric} for pages: {pages_metric}')
