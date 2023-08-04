from graphai.pipelines.investment.create_investments_graph import create_investments_graph
from graphai.pipelines.investment.aggregate_recent_past import aggregate_recent_past
from graphai.pipelines.investment.compute_jaccard_index import compute_jaccard_index
from graphai.pipelines.investment.detect_fundraisers_concepts import detect_fundraisers_concepts
from graphai.pipelines.investment.compute_investors_units import compute_investors_units
from graphai.pipelines.investment.compute_fundraisers_units import compute_fundraisers_units
from graphai.pipelines.investment.compute_investors_units_2 import compute_investors_units_2

import graphai.pipelines.investment.parameters as params

from graphai.core.utils.breadcrumb import Breadcrumb


def main():

    bc = Breadcrumb()

    bc.log('Creating investments graph...')
    create_investments_graph(params)

    bc.log('Aggregating recent past...')
    aggregate_recent_past(params)

    bc.log('Computing Jaccard index...')
    compute_jaccard_index(params)

    bc.log('Detecting fundraisers concepts...')
    detect_fundraisers_concepts(params)

    bc.log('Computing fundraiser-unit edges...')
    compute_fundraisers_units(params)

    bc.log('Computing investor-unit edges...')
    compute_investors_units(params)

    bc.log('Computing investor-unit edges 2...')
    compute_investors_units_2(params)

    bc.report()


if __name__ == '__main__':
    main()
