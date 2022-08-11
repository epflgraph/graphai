import pandas as pd

from interfaces.db import DB

db = DB()

fields = ['FundingRoundID', 'FundingRoundType', 'FundingRoundName', 'FundingRoundDate', 'FundingAmount_USD', 'CB_InvestorCount', 'City', 'Region', 'CountryISO3']
columns = ['fr_id', 'fr_type', 'fr_name', 'fr_date', 'fr_amount', 'fr_n_investors', 'fr_city', 'fr_region', 'fr_country']
frs = pd.DataFrame(db.get_funding_rounds(fields=fields), columns=columns)

frs