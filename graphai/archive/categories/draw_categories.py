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
    # Prepare input for wikify
    if raw_text is None:
        # Read input from file
        data = read_json(f'../../api/requests/{folder}/{name}.json')
    else:
        # Prepare data with given string
        data = {'raw_text': raw_text}

    # Perform wikify request and parse the results
    results = pd.DataFrame(requests.post(wikify_url, data=json.dumps(data)).json())

    # Instantiate and initialise ontology
    ontology = Ontology()
    ontology.fetch_from_db()

    # Add categories to results
    results = pd.merge(results, ontology.concepts_categories, how='inner', on='PageID')

    if level >= 2:
        results = pd.merge(
            results,
            ontology.categories_categories.rename(columns={'ChildCategoryID': 'CategoryID', 'ParentCategoryID': 'Category2ID'}),
            how='inner',
            on='CategoryID'
        )

    if level >= 3:
        results = pd.merge(
            results,
            ontology.categories_categories.rename(columns={'ChildCategoryID': 'Category2ID', 'ParentCategoryID': 'Category3ID'}),
            how='inner',
            on='Category2ID'
        )

    if level >= 4:
        results = pd.merge(
            results,
            ontology.categories_categories.rename(columns={'ChildCategoryID': 'Category3ID', 'ParentCategoryID': 'Category4ID'}),
            how='inner',
            on='Category3ID'
        )

    if level >= 5:
        results = pd.merge(
            results,
            ontology.categories_categories.rename(columns={'ChildCategoryID': 'Category4ID', 'ParentCategoryID': 'Category5ID'}),
            how='inner',
            on='Category4ID'
        )

    ################################################################

    # Create node lists with node attributes in networkx format
    concept_nodes = results[['PageID', 'PageTitle', 'SearchScore', 'LevenshteinScore', 'GraphScore', 'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore', 'MixedScore']].to_dict(orient='records')
    concept_nodes = [(d['PageTitle'], {**d, 'Type': 'concept'}) for d in concept_nodes]

    category_nodes = [(c, {'Type': 'category'}) for c in results['CategoryID'].drop_duplicates()]

    if level >= 2:
        category2_nodes = [(c, {'Type': 'category2'}) for c in results['Category2ID'].drop_duplicates()]

    if level >= 3:
        category3_nodes = [(c, {'Type': 'category3'}) for c in results['Category3ID'].drop_duplicates()]

    if level >= 4:
        category4_nodes = [(c, {'Type': 'category4'}) for c in results['Category4ID'].drop_duplicates()]

    if level >= 5:
        category5_nodes = [(c, {'Type': 'category5'}) for c in results['Category5ID'].drop_duplicates()]

    # Create edge lists in networkx format
    concept_category_edges = results[['PageTitle', 'CategoryID']].to_dict(orient='records')
    concept_category_edges = [(d['PageTitle'], d['CategoryID']) for d in concept_category_edges]

    if level >= 2:
        category_category_edges = results[['CategoryID', 'Category2ID']].drop_duplicates().to_dict(orient='records')
        category_category_edges = [(d['CategoryID'], d['Category2ID']) for d in category_category_edges]

    if level >= 3:
        category_category2_edges = results[['Category2ID', 'Category3ID']].drop_duplicates().to_dict(orient='records')
        category_category2_edges = [(d['Category2ID'], d['Category3ID']) for d in category_category2_edges]

    if level >= 4:
        category_category3_edges = results[['Category3ID', 'Category4ID']].drop_duplicates().to_dict(orient='records')
        category_category3_edges = [(d['Category3ID'], d['Category4ID']) for d in category_category3_edges]

    if level >= 5:
        category_category4_edges = results[['Category4ID', 'Category5ID']].drop_duplicates().to_dict(orient='records')
        category_category4_edges = [(d['Category4ID'], d['Category5ID']) for d in category_category4_edges]

    ################################################################

    # Create the graph and populate it
    G = nx.Graph()

    G.add_nodes_from(concept_nodes)
    G.add_nodes_from(category_nodes)

    if level >= 2:
        G.add_nodes_from(category2_nodes)

    if level >= 3:
        G.add_nodes_from(category3_nodes)

    if level >= 4:
        G.add_nodes_from(category4_nodes)

    if level >= 5:
        G.add_nodes_from(category5_nodes)

    G.add_edges_from(concept_category_edges)

    if level >= 2:
        G.add_edges_from(category_category_edges)

    if level >= 3:
        G.add_edges_from(category_category2_edges)

    if level >= 4:
        G.add_edges_from(category_category3_edges)

    if level >= 5:
        G.add_edges_from(category_category4_edges)

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
        nx.draw_networkx_nodes(H, pos, ax=ax, nodelist=[n for n, _ in category_nodes if n in H.nodes()], node_color='tab:green', alpha=0.8)

        if level >= 2:
            nx.draw_networkx_nodes(H, pos, ax=ax, nodelist=[n for n, _ in category2_nodes if n in H.nodes()], node_color='tab:red', alpha=0.8)

        if level >= 3:
            nx.draw_networkx_nodes(H, pos, ax=ax, nodelist=[n for n, _ in category3_nodes if n in H.nodes()], node_color='tab:orange', alpha=0.8)

        if level >= 4:
            nx.draw_networkx_nodes(H, pos, ax=ax, nodelist=[n for n, _ in category4_nodes if n in H.nodes()], node_color='tab:purple', alpha=0.8)

        if level >= 5:
            nx.draw_networkx_nodes(H, pos, ax=ax, nodelist=[n for n, _ in category5_nodes if n in H.nodes()], node_color='tab:gray', alpha=0.8)

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

# wikify_and_plot(level, name=name)

raw_text = """  """
wikify_and_plot(level, raw_text=raw_text)
