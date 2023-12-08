"""
Compare results from the old and new concept detection algorithm.
"""

import pandas as pd
import matplotlib.pyplot as plt

from db_cache_manager.db import DB

from graphai.core.common.config import config

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Read old and new results
pages_old = pd.read_json('pages_old.json')
pages_new = pd.read_json('pages_new.json')

# Filter out concepts not in the ontology for the old algorithm
db = DB(config['database'])
table_name = 'graph.Nodes_N_Concept'
fields = ['PageID']
pages_ontology = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

pages_old = pd.merge(pages_old, pages_ontology, how='inner', on='PageID')

# Merge pages obtained with old and new algorithm
pages = pd.merge(
    pages_old.rename(columns={'Score': 'OldScore'}),
    pages_new.rename(columns={'Score': 'NewScore'}),
    how='outer',
    on=['SlideID', 'SlideText', 'PageID', 'PageTitle']
)

# Define flags
pages['Old'] = ~pages['OldScore'].isna()
pages['New'] = ~pages['NewScore'].isna()

slides = pages.groupby(by=['SlideID', 'SlideText']).aggregate(Old=('Old', 'sum'), New=('New', 'sum'), Total=('SlideID', 'count')).reset_index()
slides['Both'] = slides['Old'] + slides['New'] - slides['Total']

# Sort slides by Both, Old, New, Total
slides = slides.sort_values(by=['Total', 'Both', 'Old', 'New'])

# Study random sample of 10 slides
sample_slide_ids = list(slides['SlideID'].sample(10, random_state=1))

for sample_slide_id in sample_slide_ids:
    sample_pages = pages[pages['SlideID'] == sample_slide_id].sort_values(by=['OldScore', 'NewScore'], ascending=False)

    sample_text = sample_pages['SlideText'].iat[0].replace('<linebreak/>', '\n')

    print(sample_slide_id)

    print(sample_text)

    print(sample_pages[['PageID', 'PageTitle', 'OldScore', 'NewScore']])

    print('#' * 180)


# Plot number of detected pages for each slide
fig, ax = plt.subplots()

ax.bar(x=slides['SlideID'], height=slides['Both'], label='Intersection')
ax.bar(x=slides['SlideID'], height=(slides['Old'] - slides['Both']), bottom=slides['Both'], label='Only old')
ax.bar(x=slides['SlideID'], height=(slides['New'] - slides['Both']), bottom=slides['Old'], label='Only new')

labels = list(slides['SlideID'])
labels = list(map(lambda s: s if s in sample_slide_ids else '', labels))

for i in range(len(sample_slide_ids)):
    labels = list(map(lambda s: (f'Slide {i+1}' if sample_slide_ids[i] == s else s), labels))

for i in range(len(labels)):
    if labels[i]:
        ax.annotate(labels[i], xy=(i, slides['Total'].iat[i]), xytext=(i, min(90, 10 + slides['Total'].iat[i])), va='bottom', ha='center', arrowprops=dict(arrowstyle='->', color='gray'))

ax.set_xticks([])
ax.set_xlabel('Slides')
ax.set_ylabel('Number of pages')

ax.legend(loc='upper left')

ax.set_title('Number of detected concepts for a random sample of 352 slides')

plt.show()
