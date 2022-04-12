import random
from html_cleaner import HTMLCleaner

from db_api import DBApi

db_api = DBApi()

course_descriptions = db_api.query_course_descriptions()
course_ids = list(course_descriptions.keys())
n_courses = len(course_ids)
print(f'Got {n_courses} courses')

random.seed(8)
sample_course_ids = random.sample(course_ids, 3)

for sample_course_id in sample_course_ids:
    print('###########################')
    print(sample_course_id)

    print('###########################')
    print(course_descriptions[sample_course_id])

    print('###########################')
    c = HTMLCleaner()
    c.feed(course_descriptions[sample_course_id])
    text = c.get_data()
    print(text)
