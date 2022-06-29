import ray

from interfaces.db import DB
from interfaces.es import ES
from utils.text.markdown import strip
from utils.time.stopwatch import Stopwatch

# Init ray
ray.init(namespace="populate_elasticsearch", include_dashboard=False, log_to_driver=True)

indices = ['wikipages_1_shards', 'wikipages_3_shards', 'wikipages_6_shards', 'wikipages_12_shards']


@ray.remote
class PopulateActor:
    def __init__(self):
        self.ess = [ES(index) for index in indices]

    def strip_and_index(self, page_id, page, page_categories):
        # Strip page content
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
            'redirect': page['redirect'],
            'category': page_categories,
            'incoming_links': 1,
            'popularity_score': page['popularity']
        }

        # Index on elasticsearch
        for es in self.ess:
            es.index_doc(doc)


# Instantiate ray actor list
n_actors = 16
actors = [PopulateActor.remote() for i in range(n_actors)]

# Init DB interface
db = DB()

# Get all page ids filtering orphans
all_page_ids = db.get_wikipage_ids(filter_orphan=True)

# Init stopwatch to track time
sw = Stopwatch()

# Define window size to filter page ids
window_size = 200000

window = 0
while True:
    # Get min/max page id for current window
    min_id = window * window_size
    max_id = (window + 1) * window_size

    # Fetch pages from database
    sw.tick()
    pages = db.get_wikipages(ids=all_page_ids, id_min_max=(min_id, max_id))
    n_pages = len(pages)
    print(f'Got {n_pages} pages (id in [{min_id}, {max_id}))!')

    # Fetch categories from database
    sw.tick()
    categories = db.get_wikipage_categories(ids=all_page_ids, id_min_max=(min_id, max_id))
    print(f'Got categories for {len(categories)} pages (id in [{min_id}, {max_id}))!')

    # Print time summary
    sw.report('Finished fetching from database')

    # Exit if no pages in window
    if not pages:
        break

    # Reset stopwatch and iterate over all pages
    sw.reset()
    actor = 0
    results = []
    for page_id in pages:
        # Extract page data and categories
        page = pages[page_id]
        page_categories = categories.get(page_id, [])

        # Execute strip_and_index in parallel
        results.append(actors[actor].strip_and_index.remote(page_id, page, page_categories))

        # Update actor index
        actor = (actor + 1) % n_actors

    # Wait for the results
    results = ray.get(results)

    # Print time summary
    sw.report(f'Finished indexing {n_pages} pages on elasticsearch')

    # Update window
    window += 1
