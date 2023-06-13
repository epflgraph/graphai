import requests
import json

import math

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from graphai.core.utils.text.io import read_json

from graphai.core.common.ontology import Ontology

pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

wikify_url = 'http://localhost:28800/text/wikify'

names = ['wave-fields', 'schreier', 'collider', 'skills']

name = names[0]

################################################################

data = read_json(f'../../api/requests/0-simple/{name}.json')
results = pd.DataFrame(requests.post(wikify_url, data=json.dumps(data)).json())

ontology = Ontology()
ontology.fetch_from_db()

results = pd.merge(results, ontology.concepts_categories, how='inner', on='PageID')
results = pd.merge(
    results,
    ontology.categories_categories.rename(columns={'ChildCategoryID': 'CategoryID', 'ParentCategoryID': 'Category2ID'}),
    how='inner',
    on='CategoryID'
)

################################################################

concept_nodes = results[['PageID', 'PageTitle', 'SearchScore', 'LevenshteinScore', 'GraphScore', 'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore', 'MixedScore']].to_dict(orient='records')
concept_nodes = [(d['PageTitle'], {**d, 'Type': 'concept'}) for d in concept_nodes]

category_nodes = [(c, {'Type': 'category'}) for c in results['CategoryID'].drop_duplicates()]
category2_nodes = [(c, {'Type': 'category2'}) for c in results['Category2ID'].drop_duplicates()]

concept_category_edges = results[['PageTitle', 'CategoryID']].to_dict(orient='records')
concept_category_edges = [(d['PageTitle'], d['CategoryID']) for d in concept_category_edges]

category_category_edges = results[['CategoryID', 'Category2ID']].drop_duplicates().to_dict(orient='records')
category_category_edges = [(d['CategoryID'], d['Category2ID']) for d in category_category_edges]

################################################################

G = nx.Graph()

G.add_nodes_from(concept_nodes)
G.add_nodes_from(category_nodes)
G.add_nodes_from(category2_nodes)

G.add_edges_from(concept_category_edges)
G.add_edges_from(category_category_edges)

connected_components = list(nx.connected_components(G))
n_connected_components = len(connected_components)

################################################################

ncols = math.floor(math.sqrt(n_connected_components))
nrows = math.ceil(n_connected_components / ncols)

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex='all', sharey='all')
fig.tight_layout()
axs = [ax for ax_row in axs for ax in ax_row]

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)

i = 0
for component_nodes in connected_components:
    ax = axs[i]
    i += 1

    H = G.subgraph(component_nodes)

    pos = nx.spring_layout(H, seed=0)

    nx.draw_networkx_nodes(H, pos, ax=ax, nodelist=[n for n, _ in concept_nodes if n in H.nodes()], node_size=[400 * d['MixedScore'] for n, d in concept_nodes if n in H.nodes()], node_color=[d['MixedScore'] for n, d in concept_nodes if n in H.nodes()], cmap='Blues', vmin=0, vmax=1, alpha=0.8)
    nx.draw_networkx_nodes(H, pos, ax=ax, nodelist=[n for n, _ in category_nodes if n in H.nodes()], node_color='tab:green', alpha=0.8)
    nx.draw_networkx_nodes(H, pos, ax=ax, nodelist=[n for n, _ in category2_nodes if n in H.nodes()], node_color='tab:red', alpha=0.8)

    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.5)
    nx.draw_networkx_labels(H, pos, ax=ax, font_size=5)

fig.suptitle(f'Results for "{data["raw_text"][:92] + "..." if len(data["raw_text"]) > 92 else data["raw_text"]}"')
plt.subplots_adjust(hspace=0, wspace=0)
plt.show()
