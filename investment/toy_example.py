import pandas as pd

from investment.compute_investors_units import *

# x = pd.DataFrame({
#     'InvestorID': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C'],
#     'Year': [2020, 2021, 2021, 2020, 2020, 2021, 2020, 2020],
#     'PageID': [5363, 5363, 9611, 5363, 9611, 1164, 9611, 1164],
#     'Score': [0.7, 0.1, 0.2, 0.4, 0.5, 0.5, 0.6, 0.9]
# })
#
# y = pd.DataFrame({
#     'UnitID': ['X', 'X', 'X', 'Y', 'Y'],
#     'PageID': [5363, 9611, 1164, 1164, 5309],
#     'Score': [0.2, 0.1, 0.3, 0.3, 0.5]
# })
#
# pairs = pd.DataFrame({
#     'InvestorID': ['A', 'A', 'B', 'B', 'B', 'C'],
#     'Year': [2020, 2021, 2020, 2021, 2021, 2020],
#     'UnitID': ['X', 'X', 'X', 'X', 'Y', 'Y']
# })
#
# edges = pd.DataFrame({
#     'SourcePageID': [5363, 5363, 1164, 5309],
#     'TargetPageID': [9611, 1164, 5309, 1164],
#     'Score': [0.7, 0.6, 0.2, 0.9]
# })

###############################################################


def base():
    x = pd.DataFrame({
        'KeyX': ['A', 'A', 'B', 'B', 'B', 'C', 'C'],
        'PageID': [1, 2, 1, 2, 3, 2, 3],
        'Score': [0.7, 0.2, 0.4, 0.5, 0.5, 0.6, 0.9]
    })

    y = pd.DataFrame({
        'KeyY': ['P', 'P', 'P', 'Q', 'Q'],
        'PageID': [1, 2, 3, 3, 4],
        'Score': [0.2, 0.1, 0.3, 0.3, 0.5]
    })

    edges = pd.DataFrame({
        'SourcePageID': [1, 1, 3],
        'TargetPageID': [2, 3, 4],
        'Score': [1, 1, 1]
    })

    return x, y, edges


def path():
    edges = pd.DataFrame({
        'SourcePageID': [1, 2, 3, 4],
        'TargetPageID': [2, 3, 4, 5],
        'Score': [1, 1, 1, 1]
    })

    x = pd.DataFrame({
        'KeyX': ['A', 'B', 'C', 'D', 'E'],
        'PageID': [1, 2, 3, 4, 5],
        'Score': [1, 1, 1, 1, 1]
    })

    y = x.rename(columns={'KeyX': 'KeyY'})

    return x, y, edges


def complete():
    edges = pd.DataFrame({
        'SourcePageID': [1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
        'TargetPageID': [2, 3, 4, 5, 3, 4, 5, 4, 5, 5],
        'Score': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    })

    x = pd.DataFrame({
        'KeyX': ['A', 'B', 'C', 'C'],
        'PageID': [1, 2, 1, 2],
        'Score': [1, 1, 0.5, 0.5]
    })

    y = x.rename(columns={'KeyX': 'KeyY'})

    return x, y, edges


def random(n):
    np.random.seed(0)

    edges = pd.merge(
        pd.DataFrame({'SourcePageID': range(n)}),
        pd.DataFrame({'TargetPageID': range(n)}),
        how='cross'
    )
    edges['Score'] = np.power(np.random.rand(len(edges)), 4)

    m = 3
    x = pd.DataFrame({
        'KeyX': ['A'] * m,
        'PageID': range(m),
        'Score': np.random.rand(m)
    })

    y = x.rename(columns={'KeyX': 'KeyY'})

    return x, y, edges


x, y, edges = random(100)

pairs = pd.merge(
    x['KeyX'].drop_duplicates(),
    y['KeyY'].drop_duplicates(),
    how='cross'
)


mix(x, normalise_graph(edges))

print(compute_affinities(x, y, pairs, edges=edges, mix_x=True, mix_y=True))
