import time

from concept_detection.interfaces.db import DB
from concept_detection.interfaces.es import ES
from concept_detection.text.stripper import strip

db = DB()
es = ES()

# Get ids of document batch checking which still have
batch_size = 1000
query = {
    'bool': {
        'must': {
            'exists': {
                'field': 'content'
            }
        }
    }
}
r = es._search(query=query, source=['id'], limit=batch_size)
batch_page_ids = [str(hit['_source']['id']) for hit in r['hits']['hits']]

print(f"Indexing {batch_size} new documents out of {r['hits']['total']['relation']} {r['hits']['total']['value']} pending ones.")

pages = db.query_wikipedia_pages(ids=batch_page_ids)

st = time.time()
i = 0
for page in pages:
    i += 1
    bar_length = 50
    done = int(bar_length * i / len(pages))
    to_do = bar_length - done
    print(f'\r[{"#" * done}{"." * to_do}]', end='', flush=True)

    stripped_page = strip(page['page_content'])
    doc = {
        'id': page['page_id'],
        'title': page['page_title'],
        'text': stripped_page['text'],
        'headings': stripped_page['headings'],
        'opening_text': stripped_page['opening_text'],
        'categories': stripped_page['categories']
    }
    es.index_doc(doc)

# Refresh index
es.refresh()

ft = time.time()
print(f'\nFinished! Took {ft - st:.2f}s.')
