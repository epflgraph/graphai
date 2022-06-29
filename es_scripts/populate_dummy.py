from interfaces.es import ES
from utils.progress_bar import ProgressBar
from utils.time.stopwatch import Stopwatch

from es_scripts.dummy import gen_random_docs

es = ES('test')

sw = Stopwatch()

docs = gen_random_docs(10)

pb = ProgressBar(len(docs))
for doc in docs:
    pb.update()
    es.index_doc(doc)

# Refresh index
es.refresh()

print(f'\nFinished! Took {sw.delta():.2f}s.')
