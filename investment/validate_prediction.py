import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from interfaces.db import DB

from utils.breadcrumb import Breadcrumb

from investment.data import *
from investment.create_investments_graph import derive_historical_data


def main():
    # Initialize breadcrumb to log and keep track of time
    bc = Breadcrumb()

    # Instantiate db interface to communicate with database
    db = DB()

    ############################################################
    # INITIALIZATION                                           #
    ############################################################

    # Define time window to compare predictions against
    min_date = '2022-01-01'
    max_date = '2023-01-01'

    bc.log(f'Checking predictions against actual investments in time window [{min_date}, {max_date})')

    ############################################################
    # BUILD GROUND TRUTH DATAFRAME                             #
    ############################################################

    bc.log('Retrieving funding rounds in time window...')

    frs = get_frs(db, min_date, max_date)

    fr_ids = list(frs['FundingRoundID'])

    ############################################################

    bc.log('Retrieving investors...')

    investors_frs = get_investors_frs(db, fr_ids)

    ############################################################

    bc.log('Retrieving investees...')

    frs_investees = get_frs_investees(db, fr_ids)
    investee_ids = list(frs_investees['InvesteeID'])

    ############################################################

    bc.log('Retrieving concepts...')

    investees_concepts = get_investees_concepts(db, investee_ids)

    ############################################################

    bc.log('Computing true investor-concept edges...')

    df = pd.merge(investors_frs, frs_investees, how='inner', on='FundingRoundID')
    df = pd.merge(df, investees_concepts, how='inner', on='InvesteeID')
    df = pd.merge(df, frs, how='inner', on='FundingRoundID')

    true_investors_concepts = derive_historical_data(df, groupby_columns=['InvestorID', 'PageID'], date_column='FundingRoundDate', amount_column='FundingAmountPerInvestor_USD', min_date=min_date, max_date=max_date)

    ############################################################
    # BUILD PREDICTIONS DATAFRAME                              #
    ############################################################

    bc.log('Retrieving predicted investor-concept edges...')

    table_name = 'ca_temp.Edges_N_Investor_N_Concept_T_Jaccard'
    fields = ['InvestorID', 'PageID', 'Jaccard_000', 'Jaccard_100', 'Jaccard_010', 'Jaccard_110']
    pred_investors_concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    ############################################################
    # EXTRACT PAIRS AND COMPUTE METRICS                        #
    ############################################################

    bc.log('Extracting investor-concept pairs...')

    true_pairs = set(true_investors_concepts[['InvestorID', 'PageID']].itertuples(index=False, name=None))
    pred_pairs = set(pred_investors_concepts[['InvestorID', 'PageID']].itertuples(index=False, name=None))
    all_pairs = true_pairs | pred_pairs

    ############################################################

    bc.log('Computing metrics for each threshold...')

    thresholds = np.linspace(0, 1, 101)

    metrics = []
    for threshold in thresholds:
        pred_pairs = set(pred_investors_concepts.loc[pred_investors_concepts['Jaccard_000'] >= threshold, ['InvestorID', 'PageID']].itertuples(index=False, name=None))

        tp = len(pred_pairs & true_pairs)
        fp = len(pred_pairs - true_pairs)
        fn = len(true_pairs - pred_pairs)
        tn = len(all_pairs - pred_pairs - true_pairs)

        accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn > 0 else 0
        precision = (tp / (tp + fp)) if tp + fp > 0 else 0
        recall = (tp / (tp + fn)) if tp + fn > 0 else 0

        metrics.append([threshold, accuracy, precision, recall])

    metrics = pd.DataFrame(metrics, columns=['Threshold', 'Accuracy', 'Precision', 'Recall'])

    ############################################################

    fig, ax = plt.subplots(dpi=150)

    ax.plot(metrics['Threshold'], metrics['Accuracy'], label='Accuracy')
    ax.plot(metrics['Threshold'], metrics['Precision'], label='Precision')
    ax.plot(metrics['Threshold'], metrics['Recall'], label='Recall')

    ax.legend()

    plt.show()

    ############################################################

    bc.report()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    main()