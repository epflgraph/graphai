from graphai.scripts.investment.create_investments_graph import create_investments_graph
from graphai.scripts.investment.aggregate_recent_past import aggregate_recent_past
from graphai.scripts.investment.compute_jaccard_index import compute_jaccard_index
from graphai.scripts.investment.detect_fundraisers_concepts import detect_fundraisers_concepts
from graphai.scripts.investment.compute_investors_units import compute_investors_units
from graphai.scripts.investment.compute_fundraisers_units import compute_fundraisers_units
from graphai.scripts.investment.compute_investors_units_2 import compute_investors_units_2

import graphai.scripts.investment.parameters as params

from graphai.core.utils import Breadcrumb


def main():

    bc = Breadcrumb()

    bc.log('Creating investments graph...')
    create_investments_graph(params)

    bc.log('Aggregating recent past...')
    aggregate_recent_past(params)

    bc.log('Computing Jaccard index...')
    compute_jaccard_index()

    bc.log('Detecting fundraisers concepts...')
    detect_fundraisers_concepts()

    bc.log('Computing fundraiser-unit edges...')
    compute_fundraisers_units()

    bc.log('Computing investor-unit edges...')
    compute_investors_units()

    bc.log('Computing investor-unit edges 2...')
    compute_investors_units_2()

    bc.report()


if __name__ == '__main__':
    main()
