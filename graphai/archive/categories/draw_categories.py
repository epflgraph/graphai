import requests
import json

import math

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from graphai.core.utils.text.io import read_json

from graphai.core.common.ontology import Ontology

wikify_url = 'http://localhost:28800/text/wikify'

################################################################


def wikify_and_plot(level, name=None, folder='0-simple', raw_text=None):
    assert level in [1, 2, 3, 4, 5], 'level not valid, must be integer between 1 and 5'

    # Prepare input for wikify
    if raw_text is None:
        # Read input from file
        data = read_json(f'../../api/requests/{folder}/{name}.json')
    else:
        # Prepare data with given string
        data = {'raw_text': raw_text}

    # Perform wikify request and parse the results
    results = pd.DataFrame(requests.post(wikify_url, data=json.dumps(data)).json())

    ################################################################

    # Instantiate and initialise ontology
    ontology = Ontology()
    ontology.fetch_from_db()

    ################################################################

    # Add categories to results
    results = pd.merge(
        results,
        ontology.concepts_categories.rename(columns={'CategoryID': f'Category1ID'}),
        how='inner',
        on='PageID'
    )

    for i in range(1, level):
        results = pd.merge(
            results,
            ontology.categories_categories.rename(columns={'ChildCategoryID': f'Category{i}ID', 'ParentCategoryID': f'Category{i + 1}ID'}),
            how='inner',
            on=f'Category{i}ID'
        )

    ################################################################

    # Create node lists with node attributes in networkx format
    concept_nodes = results[['PageID', 'PageTitle', 'SearchScore', 'LevenshteinScore', 'GraphScore', 'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore', 'MixedScore']].to_dict(orient='records')
    concept_nodes = [(d['PageTitle'], {**d, 'Type': 'concept'}) for d in concept_nodes]

    category_nodes = []
    for i in range(level):
        category_nodes.append([(c, {'Type': f'category{i+1}'}) for c in results[f'Category{i + 1}ID'].drop_duplicates()])

    # Create edge lists in networkx format
    concept_category_edges = results[['PageTitle', 'Category1ID']].to_dict(orient='records')
    concept_category_edges = [(d['PageTitle'], d['Category1ID']) for d in concept_category_edges]

    category_category_edges = []
    for i in range(1, level):
        edges = results[[f'Category{i}ID', f'Category{i + 1}ID']].drop_duplicates().to_dict(orient='records')
        edges = [(d[f'Category{i}ID'], d[f'Category{i + 1}ID']) for d in edges]
        category_category_edges.append(edges)

    ################################################################

    # Create the graph and populate it
    G = nx.Graph()

    G.add_nodes_from(concept_nodes)

    for category_nodes_subset in category_nodes:
        G.add_nodes_from(category_nodes_subset)

    G.add_edges_from(concept_category_edges)

    for category_category_edges_subset in category_category_edges:
        G.add_edges_from(category_category_edges_subset)

    # Compute the connected components as we will plot each separately to avoid cluttering
    connected_components = list(nx.connected_components(G))
    n_connected_components = len(connected_components)

    ################################################################

    # Compute the grid dimensions so that all connected components are displayed
    n_rows = math.floor(math.sqrt(n_connected_components))
    n_cols = math.ceil(n_connected_components / n_rows)

    # Create figure and flatten axs
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex='all', sharey='all', figsize=(3 * n_cols, 2.5 * n_rows), dpi=150, layout='constrained')
    if n_connected_components > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    # Hide ticks and spines from all axs
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)

    # Plot each connected component
    i = 0
    for component_nodes in connected_components:
        ax = axs[i]
        i += 1

        H = G.subgraph(component_nodes)

        # Compute position of all nodes before plotting them
        pos = nx.spring_layout(H, seed=0)

        # Plot each subset of nodes separately
        nx.draw_networkx_nodes(H, pos, ax=ax, nodelist=[n for n, _ in concept_nodes if n in H.nodes()], node_size=[400 * d['MixedScore'] for n, d in concept_nodes if n in H.nodes()], node_color=[d['MixedScore'] for n, d in concept_nodes if n in H.nodes()], cmap='Blues', vmin=0, vmax=1, alpha=0.8)

        colors = ['tab:green', 'tab:red', 'tab:orange', 'tab:purple', 'tab:gray']
        j = 0
        for category_nodes_subset in category_nodes:
            nx.draw_networkx_nodes(H, pos, ax=ax, nodelist=[n for n, _ in category_nodes_subset if n in H.nodes()], node_color=colors[j], alpha=0.8)
            j += 1

        # Plot edges and labels
        nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.5)
        nx.draw_networkx_labels(H, pos, ax=ax, font_size=5, clip_on=False)

        ax.autoscale()

    # Set title, remove space between axs and display the plot
    fig.suptitle(f'Results for "{data["raw_text"][:92] + "..." if len(data["raw_text"]) > 92 else data["raw_text"]}"')
    plt.show()


################################################################

names = ['wave-fields', 'schreier', 'collider', 'skills']
levels = [5, 4, 3, 2, 1]

name = names[0]
level = 2

wikify_and_plot(level, name=name)

# raw_text = """"""
# wikify_and_plot(level, raw_text=raw_text)
