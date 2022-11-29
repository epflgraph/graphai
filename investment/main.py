from investment.create_investments_graph import create_investments_graph
from investment.aggregate_recent_past import aggregate_recent_past
from investment.compute_jaccard_index import compute_jaccard_index
from investment.compute_investors_units import compute_investors_units
from investment.compute_fundraisers_units import compute_fundraisers_units

import investment.parameters as params

from utils.breadcrumb import Breadcrumb


def main():

    bc = Breadcrumb()

    bc.log('Creating investments graph...')
    create_investments_graph(params)

    bc.log('Aggregating recent past...')
    aggregate_recent_past(params)

    bc.log('Computing Jaccard index...')
    compute_jaccard_index()

    bc.log('Computing investor-unit edges...')
    compute_investors_units()

    bc.log('Computing fundraiser-unit edges...')
    compute_fundraisers_units()

    bc.report()


if __name__ == '__main__':
    main()
