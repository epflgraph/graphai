import numpy as np

from db_api import DBApi
from new_api import NewApi

from compare import *

db_api = DBApi()
new_api = NewApi()


def test_keywords_to_results(limit=None):
    course_descriptions = db_api.query_course_descriptions()
    course_ids = list(course_descriptions.keys())
    n_courses = len(course_ids)
    print(f'Got {n_courses} courses')

    db_wikified_course_descriptions = db_api.query_wikified_course_descriptions()

    t = {
        'results': [],
        'confs': []
    }
    i = 0
    for course_id in course_ids:
        i += 1
        print(i)

        keyword_list = [result['keywords'] for result in db_wikified_course_descriptions[course_id]]
        anchor_page_ids = db_api.course_anchor_page_ids(course_id)

        new_wikified_course_description = new_api.wikify_keywords(keyword_list, anchor_page_ids)
        comp = compare(new_wikified_course_description, db_wikified_course_descriptions[course_id])

        keywords_results = {}
        for keywords in keyword_list:
            keywords_results.setdefault(keywords, {'new': [], 'db': []})

            for result in comp['results_1']:
                if result['keywords'] == keywords:
                    keywords_results[keywords]['new'].append(result)

            for result in comp['results_2']:
                if result['keywords'] == keywords:
                    keywords_results[keywords]['db'].append(result)

        t['results'].append({
            'course_id': course_id,
            'keyword_list': keyword_list,
            'keywords_results': keywords_results
        })

        new_pairs = [(result['keywords'], result['page_id']) for result in comp['results_1']]
        db_pairs = [(result['keywords'], result['page_id']) for result in comp['results_2']]

        conf = confusion_matrix(new_pairs, db_pairs)
        conf['course_id'] = course_id
        t['confs'].append(conf)

        if i == limit:
            break

    return t


t = test_keywords_to_results()

plot_confs(t['confs'], 'course_id', f'results/course-descriptions-keywords-to-results-plot.png')

save(t['results'], f'results/course-descriptions-keywords-to-results-results.json')

for metric in ['p', 'r', 'f-score']:
    avg_metric = np.mean([conf[metric] for conf in t['confs']])
    print(f'Average {metric}: {avg_metric}')
