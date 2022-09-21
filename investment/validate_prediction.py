import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from interfaces.db import DB

from utils.breadcrumb import Breadcrumb

from investment.data import *
from investment.create_investments_graph import derive_historical_data


def get_metrics(field, pred_investors_concepts, true_investors_concepts):
    union_investors_concepts = pd.merge(true_investors_concepts[['InvestorID', 'PageID']], pred_investors_concepts[['InvestorID', 'PageID']], how='outer', on=['InvestorID', 'PageID'])

    true_n_investments = true_investors_concepts['CountAmount'].sum()
    true_total_amount = true_investors_concepts['SumAmount'].sum()

    thresholds = np.linspace(0, 1, 100 + 1)

    metrics = []
    for threshold in thresholds:
        threshold_pred_investors_concepts = pred_investors_concepts[pred_investors_concepts[field] >= threshold]
        threshold_common_investors_concepts = pd.merge(threshold_pred_investors_concepts, true_investors_concepts, how='inner', on=['InvestorID', 'PageID'])

        tp = len(threshold_common_investors_concepts)
        fp = len(threshold_pred_investors_concepts) - len(threshold_common_investors_concepts)
        fn = len(true_investors_concepts) - len(threshold_common_investors_concepts)
        tn = len(union_investors_concepts) - len(threshold_pred_investors_concepts) - len(true_investors_concepts) + len(threshold_common_investors_concepts)

        accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn > 0 else 0
        precision = (tp / (tp + fp)) if tp + fp > 0 else 0
        recall = (tp / (tp + fn)) if tp + fn > 0 else 0

        threshold_common_count_amount = threshold_common_investors_concepts['CountAmount'].sum()
        threshold_common_sum_amount = threshold_common_investors_concepts['SumAmount'].sum()
        ponderated_recall_count = (threshold_common_count_amount / true_n_investments) if true_n_investments > 0 else 0
        ponderated_recall_amount = (threshold_common_sum_amount / true_total_amount) if true_total_amount > 0 else 0

        metrics.append([threshold, accuracy, precision, recall, ponderated_recall_count, ponderated_recall_amount])

    metrics = pd.DataFrame(metrics, columns=['Threshold', 'Accuracy', 'Precision', 'Recall', 'PonderatedRecallCount', 'PonderatedRecallAmount'])

    return metrics


def plot_metrics(metrics, title=''):
    fig, ax = plt.subplots(dpi=150)

    ax.plot(metrics['Threshold'], metrics['Accuracy'], label='Accuracy')
    ax.plot(metrics['Threshold'], metrics['Precision'], label='Precision')
    ax.plot(metrics['Threshold'], metrics['Recall'], label='Recall')
    ax.plot(metrics['Threshold'], metrics['PonderatedRecallCount'], label='Ponderated recall (Count)')
    ax.plot(metrics['Threshold'], metrics['PonderatedRecallAmount'], label='Ponderated recall (Amount)')

    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=1)

    ax.legend()

    ax.set_xlabel('Jaccard index threshold')

    ax.set_title(title)

    plt.show()


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

    bc.log('Filtering new investors from true edges...')

    table_name = 'ca_temp.Nodes_N_Investor'
    fields = ['InvestorID']
    known_investors = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    true_investors_concepts = true_investors_concepts[true_investors_concepts['InvestorID'].isin(known_investors['InvestorID'])]

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

    bc.log('Computing metrics for each threshold...')

    metrics = {
        field: get_metrics(field, pred_investors_concepts, true_investors_concepts)
        for field in ['Jaccard_000', 'Jaccard_100', 'Jaccard_010', 'Jaccard_110']
    }

    ############################################################

    bc.log('Plotting metrics...')

    for field in metrics:
        plot_metrics(metrics[field], title=f'Metrics for {field}')

    ############################################################

    bc.report()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    main()
