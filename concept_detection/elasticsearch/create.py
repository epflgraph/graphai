import time

from concept_detection.interfaces.db import DB
from concept_detection.interfaces.es import ES
from concept_detection.text.stripper import strip

db = DB()
es = ES()

st = time.time()

pages = db.query_wikipedia_pages(limit=1)
for page in pages:
    stripped_page = strip(page['page_content'])
    doc = {
        'id': page['page_id'],
        'title': page['page_title'],
        'content': stripped_page['text']
    }
    es.index_doc(doc)

# Refresh index
es.refresh()

ft = time.time()
print(f'Finished! Took {ft - st:.2f}s.')
