import math

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def draw_ontology(results, graph, level):
    """
    Draws the ontology neighbourhood induced by the given set of wikify results. The resulting svg is not returned but stored in /tmp/file.svg.

    Args:
        results (list(dict)): A serialised (orient='records') pandas DataFrame with columns ['concept_id', 'concept_name', 'search_score',
        'levenshtein_score', 'graph_score', 'ontology_local_score', 'ontology_global_score', 'keywords_score', 'mixed_score'].
        graph (ConceptsGraph): The concepts graph and ontology object.
        level (int): Level up to which the visualisation considers categories.
    """

    # Turn off pyplot's interactive mode and use non-rendering backend
    plt.switch_backend('Agg')

    ################################################################

    # Save empty figure in case of empty input
    if not results:
        plt.savefig('/tmp/file.svg', format='svg')
        return True

    ################################################################

    # Init ontology in case it is not initialised
    graph.load_from_db()

    ################################################################

    # Add categories to results
    results = pd.merge(
        pd.DataFrame(results),
        graph.concepts_categories.rename(columns={'category_id': 'level_4_category_id'}),
        how='inner',
        on='concept_id'
    )

    for i in reversed(range(level, 4)):
        results = pd.merge(
            results,
            graph.categories_categories.rename(columns={'child_category_id': f'level_{i + 1}_category_id', 'parent_category_id': f'level_{i}_category_id'}),
            how='inner',
            on=f'level_{i + 1}_category_id'
        )

    ################################################################

    # Create node lists with node attributes in networkx format
    concept_nodes = results[['concept_id', 'concept_name', 'search_score', 'levenshtein_score', 'graph_score', 'ontology_local_score', 'ontology_global_score', 'keywords_score', 'mixed_score']].to_dict(orient='records')
    concept_nodes = [(d['concept_name'], {**d, 'type': 'concept'}) for d in concept_nodes]

    category_nodes = []
    for i in reversed(range(level, 4 + 1)):
        category_nodes.append([(c, {'type': f'level_{i}_category_id'}) for c in results[f'level_{i}_category_id'].drop_duplicates()])

    # Create edge lists in networkx format
    concept_category_edges = results[['concept_name', 'level_4_category_id']].to_dict(orient='records')
    concept_category_edges = [(d['concept_name'], d['level_4_category_id']) for d in concept_category_edges]

    category_category_edges = []
    for i in reversed(range(level, 4)):
        edges = results[[f'level_{i + 1}_category_id', f'level_{i}_category_id']].drop_duplicates().to_dict(orient='records')
        edges = [(d[f'level_{i + 1}_category_id'], d[f'level_{i}_category_id']) for d in edges]
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

    # Save empty figure in case there is not any connected component
    if n_connected_components == 0:
        plt.savefig('/tmp/file.svg', format='svg')
        return True

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
        nx.draw_networkx_nodes(H, pos, ax=ax, nodelist=[n for n, _ in concept_nodes if n in H.nodes()], node_size=[400 * d['mixed_score'] for n, d in concept_nodes if n in H.nodes()], node_color=[d['mixed_score'] for n, d in concept_nodes if n in H.nodes()], cmap='Blues', vmin=0, vmax=1, alpha=0.8)

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
    # fig.suptitle(f'Results for "{data["raw_text"][:92] + "..." if len(data["raw_text"]) > 92 else data["raw_text"]}"')
    plt.savefig('/tmp/file.svg', format='svg')


def draw_graph(results, graph, concept_score_threshold=0.3, edge_threshold=0.3, min_component_size=3):
    """
    Draws the concept graph neighbourhood induced by the given set of wikify results. The resulting svg is not returned but stored in /tmp/file.svg.

    Args:
        results (list(dict)): A serialised (orient='records') pandas DataFrame with columns ['concept_id', 'concept_name', 'search_score',
        'levenshtein_score', 'graph_score', 'ontology_local_score', 'ontology_global_score', 'keywords_score', 'mixed_score'].
        graph (ConceptsGraph): The concepts graph and ontology object.
        concept_score_threshold (float): Score threshold below which concepts are filtered out. Default: 0.3.
        edge_threshold (float): Score threshold below which edges are filtered out. Default: 0.3.
        min_component_size (int): Size threshold below which connected components are filtered out. Default: 3.
    """

    # Turn off pyplot's interactive mode and use non-rendering backend
    plt.switch_backend('Agg')

    ################################################################

    # Save empty figure in case of empty input
    if not results:
        plt.savefig('/tmp/file.svg', format='svg')
        return True

    ################################################################

    # Init graph in case it is not initialised
    graph.load_from_db()

    ################################################################

    # Filter results depending on score
    results = pd.DataFrame(results)
    results = results[results['mixed_score'] >= concept_score_threshold]

    ################################################################

    # Keep only edges lexicographically sorted and above threshold
    concepts_concepts = graph.concepts_concepts[
        (graph.concepts_concepts['source_concept_id'] < graph.concepts_concepts['target_concept_id'])
        & (graph.concepts_concepts['score'] >= edge_threshold)
    ]

    # Add concept titles
    concepts_concepts = pd.merge(
        concepts_concepts,
        results[['concept_id', 'concept_name']].rename(columns={'concept_id': 'source_concept_id', 'concept_name': 'source_concept_name'}),
        how='inner',
        on='source_concept_id'
    )

    concepts_concepts = pd.merge(
        concepts_concepts,
        results[['concept_id', 'concept_name']].rename(columns={'concept_id': 'target_concept_id', 'concept_name': 'target_concept_name'}),
        how='inner',
        on='target_concept_id'
    )

    #################################################################

    # Create node lists with node attributes in networkx format
    nodes = results[['concept_id', 'concept_name', 'search_score', 'levenshtein_score', 'graph_score', 'ontology_local_score', 'ontology_global_score', 'keywords_score', 'mixed_score']].to_dict(orient='records')
    nodes = [(d['concept_name'], d) for d in nodes]

    # Create edge lists in networkx format
    edges = concepts_concepts[['source_concept_name', 'target_concept_name', 'score']].to_dict(orient='records')
    edges = [(d['source_concept_name'], d['target_concept_name'], {'score': d['score']}) for d in edges]

    ################################################################

    # Create the graph and populate it
    G = nx.Graph()

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Compute the connected components as we will plot each separately to avoid cluttering
    connected_components = list(nx.connected_components(G))
    connected_components = [component_nodes for component_nodes in connected_components if len(component_nodes) >= min_component_size]

    n_connected_components = len(connected_components)

    ################################################################

    # Save empty figure in case there is not any connected component
    if n_connected_components == 0:
        plt.savefig('/tmp/file.svg', format='svg')
        return True

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
        pos = nx.spring_layout(H, iterations=1000, weight='score', seed=0)

        # Plot nodes
        nx.draw_networkx_nodes(H, pos, ax=ax, nodelist=[n for n, _ in nodes if n in H.nodes()], node_size=[400 * d['mixed_score'] for n, d in nodes if n in H.nodes()], node_color=[d['mixed_score'] for n, d in nodes if n in H.nodes()], cmap='Blues', vmin=0, vmax=1, alpha=0.8)

        # Plot edges and labels
        nx.draw_networkx_edges(H, pos, ax=ax, edgelist=[(s, t) for s, t, _ in edges if (s, t) in H.edges()], width=[d['score'] for s, t, d in edges if (s, t) in H.edges()], alpha=0.5)
        nx.draw_networkx_labels(H, pos, ax=ax, font_size=5, clip_on=False)

        ax.autoscale()

    # Set title, remove space between axs and display the plot
    # fig.suptitle(f'Results for "{data["raw_text"][:92] + "..." if len(data["raw_text"]) > 92 else data["raw_text"]}"')
    plt.savefig('/tmp/file.svg', format='svg')
