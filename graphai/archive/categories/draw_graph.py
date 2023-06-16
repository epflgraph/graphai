import requests
import json

import math

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from graphai.core.utils.text.io import read_json

from graphai.core.common.graph import ConceptsGraph

wikify_url = 'http://localhost:28800/text/wikify'

pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

################################################################


def wikify_and_plot(name=None, folder='0-simple', raw_text=None, concept_score_threshold=0.3, edge_threshold=0.3, component_size_threshold=3):
    # Prepare input for wikify
    if raw_text is None:
        # Read input from file
        data = read_json(f'../../api/requests/{folder}/{name}.json')
    else:
        # Prepare data with given string
        data = {'raw_text': raw_text}

    # Perform wikify request and parse the results
    results = pd.DataFrame(requests.post(wikify_url, data=json.dumps(data)).json())

    # Filter results depending on score
    results = results[results['MixedScore'] >= concept_score_threshold]

    ################################################################

    # Instantiate and initialise graph
    graph = ConceptsGraph()
    graph.fetch_from_db()

    ################################################################

    # Keep only edges lexicographically sorted and above threshold
    concepts_concepts = graph.concepts_concepts[
        (graph.concepts_concepts['SourcePageID'] < graph.concepts_concepts['TargetPageID'])
        & (graph.concepts_concepts['NormalisedScore'] >= edge_threshold)
    ]

    # Add concept titles
    concepts_concepts = pd.merge(
        concepts_concepts,
        results[['PageID', 'PageTitle']].rename(columns={'PageID': 'SourcePageID', 'PageTitle': 'SourcePageTitle'}),
        how='inner',
        on='SourcePageID'
    )

    concepts_concepts = pd.merge(
        concepts_concepts,
        results[['PageID', 'PageTitle']].rename(columns={'PageID': 'TargetPageID', 'PageTitle': 'TargetPageTitle'}),
        how='inner',
        on='TargetPageID'
    )

    #################################################################

    # Create node lists with node attributes in networkx format
    nodes = results[['PageID', 'PageTitle', 'SearchScore', 'LevenshteinScore', 'GraphScore', 'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore', 'MixedScore']].to_dict(orient='records')
    nodes = [(d['PageTitle'], d) for d in nodes]

    # Create edge lists in networkx format
    edges = concepts_concepts[['SourcePageTitle', 'TargetPageTitle', 'NormalisedScore']].to_dict(orient='records')
    edges = [(d['SourcePageTitle'], d['TargetPageTitle'], {'Score': d['NormalisedScore']}) for d in edges]

    ################################################################

    # Create the graph and populate it
    G = nx.Graph()

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Compute the connected components as we will plot each separately to avoid cluttering
    connected_components = list(nx.connected_components(G))
    connected_components = [component_nodes for component_nodes in connected_components if len(component_nodes) >= component_size_threshold]

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
        pos = nx.spring_layout(H, iterations=1000, weight='Score', seed=0)

        # Plot nodes
        nx.draw_networkx_nodes(H, pos, ax=ax, nodelist=[n for n, _ in nodes if n in H.nodes()], node_size=[400 * d['MixedScore'] for n, d in nodes if n in H.nodes()], node_color=[d['MixedScore'] for n, d in nodes if n in H.nodes()], cmap='Blues', vmin=0, vmax=1, alpha=0.8)

        # Plot edges and labels
        nx.draw_networkx_edges(H, pos, ax=ax, edgelist=[(s, t) for s, t, _ in edges if (s, t) in H.edges()], width=[d['Score'] for s, t, d in edges if (s, t) in H.edges()], alpha=0.5)
        nx.draw_networkx_labels(H, pos, ax=ax, font_size=5, clip_on=False)

        ax.autoscale()

    # Set title, remove space between axs and display the plot
    fig.suptitle(f'Results for "{data["raw_text"][:92] + "..." if len(data["raw_text"]) > 92 else data["raw_text"]}"')
    plt.show()


################################################################

names = ['wave-fields', 'schreier', 'collider', 'skills']

name = names[0]

wikify_and_plot(name=name)

# raw_text = """
# """
# wikify_and_plot(raw_text=raw_text)
