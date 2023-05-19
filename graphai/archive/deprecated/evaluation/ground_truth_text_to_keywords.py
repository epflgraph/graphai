import numpy as np

from db_api import DBApi
from new_api import NewApi

from compare import *

db_api = DBApi()
new_api = NewApi()


def test_text_to_keywords(limit=None):
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

        raw_text = course_descriptions[course_id]
        pyth_course_keywords = new_api.keywords(raw_text)
        nltk_course_keywords = new_api.keywords_nltk(raw_text)
        db_course_keywords = [result['keywords'] for result in db_wikified_course_descriptions[course_id]]

        t['results'].append({
            'course_id': course_id,
            'raw_text': raw_text,
            'pyth_keywords': pyth_course_keywords,
            'nltk_keywords': nltk_course_keywords,
            'db_keywords': db_course_keywords
        })

        pyth_conf = confusion_matrix(pyth_course_keywords, db_course_keywords)
        pyth_conf['course_id'] = course_id

        nltk_conf = confusion_matrix(nltk_course_keywords, db_course_keywords)
        nltk_conf['course_id'] = course_id

        t['confs'].append({
            'pyth': pyth_conf,
            'nltk': nltk_conf
        })

        if i == limit:
            break

    return t


t = test_text_to_keywords()

pyth_confs = [conf['pyth'] for conf in t['confs']]
nltk_confs = [conf['nltk'] for conf in t['confs']]

plot_confs(pyth_confs, 'course_id', f'results/course-descriptions-text-to-keywords-pyth-db-plot.png')
plot_confs(nltk_confs, 'course_id', f'results/course-descriptions-text-to-keywords-nltk-db-plot.png')

save(t['results'], f'results/course-descriptions-text-to-keywords-results.json')

for metric in ['p', 'r', 'f-score']:
    pyth_metric = np.mean([conf[metric] for conf in pyth_confs])
    nltk_metric = np.mean([conf[metric] for conf in nltk_confs])
    print(f'Average {metric} for pyth: {pyth_metric}')
    print(f'Average {metric} for nltk: {nltk_metric}')
