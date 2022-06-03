import ray

from concept_detection.interfaces.db import DB
from concept_detection.interfaces.es import ES
from concept_detection.text.stripper import strip
from concept_detection.time.stopwatch import Stopwatch

# Init ray
ray.init(namespace="populate_elasticsearch", include_dashboard=False, log_to_driver=True)

@ray.remote
class Actor:
    def __init__(self):
        self.es = ES()

    def strip_and_index(self, page_id, page_title, page_content, page_categories):
        # Strip page content
        stripped_page = strip(page_content)

        # Prepare document to index
        doc = {
            'id': page_id,
            'title': page_title,
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
        self.es.index_doc(doc)


# Instantiate ray actor list
n_actors = 16
actors = [Actor.remote() for i in range(n_actors)]

# Define window size and offset to filter page ids
window_size = 1000
window_offset = 0
min_id = window_offset * window_size
max_id = (window_offset + 1) * window_size

# Init stopwatch to track time
sw = Stopwatch()

# Fetch pages from database
db = DB()
pages = db.get_wikipages(id_min_max=(min_id, max_id))
n_pages = len(pages)
print(f'Got {n_pages} pages (id in [{min_id}, {max_id}])!')
sw.lap()

# Fetch categories from database
categories = db.get_wikipage_categories(id_min_max=(min_id, max_id))
print(f'Got categories for {len(categories)} pages (id in [{min_id}, {max_id}])!')

# Print time summary
sw.report('Finished fetching from database')

# Reset stopwatch and iterate over all pages
sw.reset()
i = 0
results = []
for page_id in pages:
    # Extract page data and categories
    page = pages[page_id]
    page_categories = categories.get(page_id, [])

    # Execute strip_and_index in parallel
    results.append(actors[i % n_actors].strip_and_index.remote(page_id, page['title'], page['content'], page_categories))
    i += 1

# Wait for the results
results = ray.get(results)

# Print time summary
sw.report(f'Finished indexing {n_pages} pages on elasticsearch')
