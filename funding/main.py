import pandas as pd
import matplotlib.pyplot as plt

from interfaces.db import DB

pd.set_option('display.width', 320)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 10)

db = DB()

top_concept_ids = [5309, 36674345, 282635, 39388, 261925, 433425, 9611, 2861, 14539, 59252]
bottom_concept_ids = [4015968, 25405, 32558, 8965447, 21197, 30589, 1030104, 26201, 42125403, 31775]

# concept_ids = bottom_concept_ids[-5:]
concept_ids = top_concept_ids[:5]

# Get concepts with id in starting list
concepts = pd.DataFrame(db.get_concepts(concept_ids), columns=['concept_id', 'concept_name'])
print(concepts)

# Extract list of concept ids
concept_ids = list(set(concepts['concept_id']))
print(f'Got {len(concept_ids)} concepts!')

# Add association concepts <-> organisations
concepts_orgs = pd.DataFrame(db.get_concept_organisations(concept_ids), columns=['concept_id', 'org_id'])
concepts_orgs = pd.merge(concepts, concepts_orgs, how='inner', on='concept_id')
print(concepts_orgs)

# Extract list of organisation ids
org_ids = list(set(concepts_orgs['org_id']))
print(f'Got {len(org_ids)} organisations!')

# Add association funded organisations <-> funding rounds
funded_orgs_frs = pd.DataFrame(db.get_beneficiary_funding_rounds(org_ids), columns=['org_id', 'fr_id'])
concepts_orgs_frs = pd.merge(concepts_orgs, funded_orgs_frs, how='inner', on='org_id')
print(concepts_orgs_frs)

# Extract list of funding round ids
fr_ids = list(set(concepts_orgs_frs['fr_id']))
print(f'Got {len(fr_ids)} funding rounds!')

# Add funding round dates and amounts
frs = pd.DataFrame(db.get_funding_rounds(fr_ids), columns=['fr_id', 'date', 'amount'])
concepts_orgs_frs = pd.merge(concepts_orgs_frs, frs, how='inner', on='fr_id')
print(concepts_orgs_frs)

# Derive year from date
concepts_orgs_frs['year'] = concepts_orgs_frs['date'].astype(str).str.split('-').str[0].astype(int)
concepts_orgs_frs = concepts_orgs_frs.drop('date', axis=1)
print(concepts_orgs_frs)

# Group by concept and year
investments = concepts_orgs_frs[['concept_id', 'year', 'amount']].groupby(by=['concept_id', 'year'], as_index=False).sum()
print(investments)

# Create complete grid of concepts and years
min_year = max(2000, min(investments['year']))
max_year = max(investments['year'])

skeleton = pd.DataFrame({'concept_id': list(set(investments['concept_id']))})
skeleton = skeleton.merge(pd.DataFrame({'year': range(min_year, max_year + 1)}), how='cross')
investments = pd.merge(skeleton, investments, how='left', on=['concept_id', 'year'])

# Fill NA values and change amount scale to milions of USD
investments = investments.fillna(0)
investments['amount'] = investments['amount'] / 1000000
print(investments)

# Produce plot with results
fig, ax = plt.subplots(dpi=150, figsize=(9, 6))

for concept_id in list(set(investments['concept_id'])):
    concept_name = concepts.loc[concepts['concept_id'] == concept_id, 'concept_name'].iloc[0]
    concept_years = investments.loc[investments['concept_id'] == concept_id, 'year']
    concept_amounts = investments.loc[investments['concept_id'] == concept_id, 'amount']
    ax.plot(concept_years, concept_amounts, label=concept_name, alpha=0.8, linewidth=0.8)

ax.set_xlabel('Time (years)')
ax.set_ylabel('Aggregated investment (M$)')
ax.set_title('Aggregated investment per concept over time')

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})

plt.show()
