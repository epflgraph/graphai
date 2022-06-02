import time

from concept_detection.interfaces.db import DB
from concept_detection.interfaces.es import ES
from concept_detection.text.stripper import strip
from concept_detection.text.io import ProgressBar

db = DB()
es = ES()

window_size = 100
pages = db.get_wikipages(id_min_max=(0, window_size))
print(f'Got {len(pages)} pages (id in [0, {window_size}])!')
print(pages.keys())
categories = db.get_wikipage_categories(id_min_max=(0, window_size))
print(f'Got categories for {len(categories)} pages (id in [0, {window_size}])!')
print(categories.keys())

print(set(categories.keys()) - set(pages.keys()))

st = time.time()
b = ProgressBar(len(pages), bar_length=200)
for page_id in pages:
    b.update()

    page = pages[page_id]
    page_categories = categories[page_id]
    stripped_page = strip(page['content'])

    doc = {
        'id': page_id,
        'title': page['title'],
        'text': stripped_page['text'],
        'heading': stripped_page['heading'],
        'opening_text': stripped_page['opening_text'],
        'auxiliary_text': stripped_page['auxiliary_text'],
        'file_text': '',
        'redirect': [],
        'category': [],
        'incoming_links': 1,
        'popularity_score': 1
    }

    es.index_doc(doc)

# Refresh index
es.refresh()

ft = time.time()
print(f'\nFinished! Took {ft - st:.2f}s.')
