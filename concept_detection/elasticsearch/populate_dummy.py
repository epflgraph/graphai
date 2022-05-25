import time

from concept_detection.text.io import ProgressBar
from concept_detection.interfaces.es import ES

es = ES()

docs = [
    {
        'id': i,
        'title': f'Title {i}',
        'category': ['cat 1', 'cat2', 'cat3 3'],
        'heading': ['head 11', 'head - 2', 'head---3'],
        'redirect': [f'Alternative Title {i}', f'Other title {i}', f'Yet one more title {i}'],
        'text': f'redirected from Alternative Title {i} this is the sample {i} text, it is very nice and well organized etc etc {i}',
        'opening_text': f'this is the sample {i} text',
        'auxiliary_text': f'redirected from Alternative Title {i}',
        'file_text': f'this is a file for {i}',
        'popularity_score': 1 / (i + 1),
        'incoming_links': 33 * i
    }
    for i in range(10)
]

st = time.time()
b = ProgressBar(len(docs))
for doc in docs:
    b.update()

    es.index_doc(doc)

    # Refresh index
    es.refresh()

ft = time.time()
print(f'\nFinished! Took {ft - st:.2f}s.')
