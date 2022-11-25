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

x = pd.DataFrame({
    'KeyX': ['A', 'A', 'B', 'B', 'B', 'C', 'C'],
    'PageID': [1, 2, 1, 2, 3, 2, 3],
    'Score': [0.7, 0.2, 0.4, 0.5, 0.5, 0.6, 0.9]
})

y = pd.DataFrame({
    'KeyY': ['U', 'U', 'U', 'V', 'V'],
    'PageID': [1, 2, 3, 3, 4],
    'Score': [0.2, 0.1, 0.3, 0.3, 0.5]
})

# edges = pd.DataFrame({
#     'SourcePageID': [1, 1, 3, 4],
#     'TargetPageID': [2, 3, 4, 3],
#     'Score': [0.7, 0.6, 0.2, 0.9]
# })

# edges = pd.DataFrame({
#     'SourcePageID': [1, 1, 2, 3, 3, 4],
#     'TargetPageID': [2, 3, 1, 1, 4, 3],
#     'Score': [1, 1, 1, 1, 1, 1]
# })

edges = pd.DataFrame({
    'SourcePageID': [1, 1, 3],
    'TargetPageID': [2, 3, 4],
    'Score': [1, 1, 1]
})
edges = normalise_graph(edges)


pairs = pd.merge(
    x['KeyX'].drop_duplicates(),
    y['KeyY'].drop_duplicates(),
    how='cross'
)


print(compute_affinities(x, y, pairs, edges))
