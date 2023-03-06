from graphai.core.interfaces.es import ES
from graphai.core.utils.progress_bar import ProgressBar
from graphai.core.utils.time.stopwatch import Stopwatch

from graphai.scripts.elasticsearch.dummy import gen_random_docs

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
