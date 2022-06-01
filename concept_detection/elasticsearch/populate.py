import time
import random

from concept_detection.text.io import ProgressBar
from concept_detection.interfaces.db import DB
from concept_detection.interfaces.es import ES
from concept_detection.text.io import pprint
from concept_detection.text.stripper import strip

db = DB()
es = ES()

random.seed(0)
ids = [str(n) for n in random.sample(range(1000, 10000000), 100)]
pages = db.query_wikipedia_pages(ids=ids, limit=3)

st = time.time()
b = ProgressBar(len(pages))
for page in pages:
    b.update()

    stripped_page = strip(page['page_content'])
    doc = {
        'id': page['page_id'],
        'title': page['page_title'],
        'text': stripped_page['text'],
        'heading': stripped_page['heading'],
        'opening_text': stripped_page['opening_text'],
        'auxiliary_text': stripped_page['auxiliary_text']
    }
    print()
    print(f'id: {doc["id"]}')
    print(f'title: {doc["title"]}')
    print(f'auxiliary_text: {doc["auxiliary_text"]}')
    print(f'opening_text: {doc["opening_text"]}')
    print(f'text: {doc["text"]}')
    print(f'heading: {doc["heading"]}')
    break
    # es.index_doc(doc)

# Refresh index
es.refresh()

ft = time.time()
print(f'\nFinished! Took {ft - st:.2f}s.')
