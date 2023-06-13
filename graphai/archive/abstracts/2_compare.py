import pandas as pd

import matplotlib.pyplot as plt

from graphai.core.utils.text.io import read_json

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

names = ['wave-fields', 'schreier', 'collider', 'skills']

name = names[0]

prefixes = ['no-rescore', 'rescore']

left = pd.DataFrame(read_json(f'{prefixes[0]}-{name}.json'))
right = pd.DataFrame(read_json(f'{prefixes[1]}-{name}.json'))

both = pd.merge(
    left,
    right,
    how='outer',
    on=['PageID', 'PageTitle'],
    suffixes=('L', 'R')
)

both = both.sort_values(by=['MixedScoreR', 'MixedScoreL'], ascending=False).reset_index(drop=True)

print(both)

################################################################

both['MixedScoreDiff'] = both['MixedScoreR'] - both['MixedScoreL']
both = both[both['MixedScoreDiff'].abs() >= 0.01]
both = both.sort_values(by='MixedScoreDiff').reset_index(drop=True)

print(both)

fig, ax = plt.subplots()
ax.barh(both['PageTitle'], both['MixedScoreDiff'])
ax.set_xlabel('Difference in MixedScore after rescoring')
ax.set_ylabel('Concepts')
ax.set_title('Difference in MixedScore after rescoring')
plt.show()
