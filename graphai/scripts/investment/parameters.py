"""
Parameters that define the creation of the investment tables
"""
from core.utils.time.date import *


###########################
# Recent past time window #
###########################
# Time window defining "recent past".
# Only funding rounds in this time window will be considered in the aggregation step,
# and hence only those funding rounds will be used for the computation of the Jaccard
# index and investor-unit and fundraiser-unit affinities.
max_date = str(now().date())
max_year = int(max_date.split('-')[0])
min_year = max_year - 4
min_date = f'{min_year}-01-01'

recent_past = {
    'min_date': min_date,
    'min_year': min_year,
    'max_date': max_date,
    'max_year': max_year
}

#############
# Countries #
#############
# Only funding rounds whose country is in the list will be considered, the rest will be ignored.
# None means no filtering is done and thus all countries are considered.
switzerland = ['CHE']
switzerland_nbh = ['CHE', 'DEU', 'FRA', 'ITA', 'AUT', 'BEL', 'NLD', 'LUX']

# countries = switzerland
countries = switzerland_nbh
# countries = None

##################
# Investor types #
##################
# Only investors whose type is in the list will be considered, the rest will be ignored.
# investor_types = ['Person']
investor_types = ['Organization', 'Person']

#######################
# Funding round types #
#######################
# Only funding rounds whose type is in the list will be considered, the rest will be ignored.
# None means no filtering is done and thus all funding rounds are considered.
fr_types = ['pre seed']
# fr_types = None

