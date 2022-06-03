from concept_detection.interfaces.db import DB
from concept_detection.interfaces.es import ES
from concept_detection.text.stripper import strip
from concept_detection.text.io import ProgressBar
from concept_detection.time.stopwatch import Stopwatch

db = DB()
es = ES()

# Define window size and offset to filter page ids
window_size = 100
window_offset = 0
min_id = window_offset * window_size
max_id = (window_offset + 1) * window_size

# Init stopwatch to track time
sw = Stopwatch()

# Fetch pages from database
pages = db.get_wikipages(id_min_max=(min_id, max_id))
n_pages = len(pages)
print(f'Got {n_pages} pages (id in [{min_id}, {max_id}])!')
sw.lap()

# Fetch categories from database
categories = db.get_wikipage_categories(id_min_max=(min_id, max_id))
print(f'Got categories for {len(categories)} pages (id in [{min_id}, {max_id}])!')

# Print time summary
sw.report('Finished fetching from database')

# Init progressbar to visually keep track of progress
pb = ProgressBar(n_pages, bar_length=200)

# Reset stopwatch and iterate over all pages
sw.reset()
for page_id in pages:
    # Update progressbar
    pb.update()

    # Extract page data and strip markdown
    page = pages[page_id]
    page_categories = categories.get(page_id, [])
    stripped_page = strip(page['content'])

    # Prepare document to index
    doc = {
        'id': page_id,
        'title': page['title'],
        'text': stripped_page['text'],
        'heading': stripped_page['heading'],
        'opening_text': stripped_page['opening_text'],
        'auxiliary_text': stripped_page['auxiliary_text'],
        'file_text': '',
        'redirect': [],
        'category': page_categories,
        'incoming_links': 1,
        'popularity_score': 1
    }

    # Index on elasticsearch
    es.index_doc(doc)

# Refresh index
es.refresh()

# Print time summary
sw.report(f'Finished indexing {n_pages} pages on elasticsearch')
