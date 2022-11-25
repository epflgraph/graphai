"""
Parameters that define the creation of the investment tables
"""

#############
# Countries #
#############
# Only funding rounds whose country is in the list will be considered, the rest will be ignored.
# None means no filtering is done and thus all countries are considered.
switzerland = ['CHE']
switzerland_nbh = ['CHE', 'DEU', 'FRA', 'ITA', 'AUT', 'BEL', 'NLD', 'LUX']

countries = switzerland
# countries = switzerland_nbh
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

