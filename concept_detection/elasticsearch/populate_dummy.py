import time

from utils.progress_bar import ProgressBar
from concept_detection.elasticsearch.dummy import gen_random_docs
from concept_detection.interfaces.es import ES

es = ES('test')

docs = gen_random_docs(10)


st = time.time()
b = ProgressBar(len(docs))
for doc in docs:
    b.update()
    es.index_doc(doc)

# Refresh index
es.refresh()

ft = time.time()
print(f'\nFinished! Took {ft - st:.2f}s.')
