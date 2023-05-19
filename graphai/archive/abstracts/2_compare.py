import pandas as pd

from graphai.core.utils.text.io import read_json

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

names = ['wave-fields', 'schreier', 'collider', 'skills']

name = names[0]

original = pd.DataFrame(read_json(f'original-{name}.json'))
patched = pd.DataFrame(read_json(f'patched-{name}.json'))

both = pd.merge(
    original[['PageID', 'PageTitle', 'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore', 'MixedScore']],
    patched[['PageID', 'PageTitle', 'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore', 'MixedScore']],
    how='outer',
    on=['PageID', 'PageTitle'],
    suffixes=('Orig', 'Patch')
)

both = both[['PageID', 'PageTitle', 'OntologyLocalScoreOrig', 'OntologyLocalScorePatch', 'OntologyGlobalScoreOrig', 'OntologyGlobalScorePatch', 'KeywordsScoreOrig', 'KeywordsScorePatch', 'MixedScoreOrig', 'MixedScorePatch']]

both = both.sort_values(by=['MixedScorePatch'], ascending=False)

print(both)
