import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from db_cache_manager.db import DB
from graphai.core.interfaces.config_loader import load_db_config

from graphai.core.utils.text.io import cprint


def show_trends(unit_id, unit_concepts, concepts):
    cprint(('#' * 10) + ' INVESTMENT TRENDS ' + ('#' * 10), color='blue')

    # Extract concepts to display
    top_concept_ids = list(unit_concepts.sort_values(by='Score', ascending=False).head(5)['PageID'])
    concepts = concepts[concepts['PageID'].isin(top_concept_ids)].reset_index(drop=True)

    # Print summary for last year
    last_year = concepts['Year'].max() - 1
    for name, group in concepts.groupby(by=['PageID', 'PageTitle']):
        last_year_amount = group.loc[group['Year'] == last_year, 'SumAmount'].min()
        second_last_year_amount = group.loc[group['Year'] == (last_year - 1), 'SumAmount'].min()

        perc_increment = 100 * (second_last_year_amount / last_year_amount - 1) if last_year_amount else 0

        cprint(f'\U0001F30D {name[1]}', color='blue')
        cprint(f'    · {last_year_amount / 1e6:.2f} M$ raised in {last_year}, {perc_increment:+.2f}% with respect to {last_year - 1}.', color='blue')

    # Derive extra columns
    concepts['SumAmount'] /= 1e6
    concepts['AvgAmountPerInvestment'] = (concepts['SumAmount'] / concepts['CountAmount']).replace([-np.inf, np.inf], 0)
    concepts['AvgAmountPerInvestor'] = (concepts['SumAmount'] / concepts['CountInvestment']).replace([-np.inf, np.inf, np.nan], 0)
    concepts['AvgInvestmentsPerInvestor'] = (concepts['CountInvestment'] / concepts['CountInvestor']).replace([-np.inf, np.inf], 0)

    titles = {
        'SumAmount': 'Total amount (M$)',
        'CountAmount': 'Number of funding rounds',
        'CountInvestor': 'Number of investors',
        'AvgAmountPerInvestment': 'Average amount (M$) per funding round',
        'AvgAmountPerInvestor': 'Average amount (M$) per investor',
        'AvgInvestmentsPerInvestor': 'Average number of funding rounds per investor'
    }

    # Line plots Year vs total amount, number of investments and number of investors
    fig, axs = plt.subplots(2, 3, dpi=100, figsize=(20, 12))
    for i, column in enumerate(['SumAmount', 'CountAmount', 'CountInvestor', 'AvgAmountPerInvestment', 'AvgAmountPerInvestor', 'AvgInvestmentsPerInvestor']):
        ax = axs.flat[i]

        for name, group in concepts.groupby(by=['PageID', 'PageTitle']):
            x = group['Year']
            y = group[column]
            ax.plot(x, y, marker='o', markersize=4, label=name[1])

        ax.set_xlabel('Year')
        ax.set_title(titles[column])

        ax.set_ylim(bottom=0, top=ax.get_ylim()[1])

    fig.suptitle('INVESTMENT TRENDS (chart view)')
    axs.flat[0].legend()

    plt.savefig(f'img/{unit_id}')

    cprint('')


def show_matchmaking_list_view(investors_concepts_jaccard, investors_concepts, unit_concept_ids):
    cprint(('#' * 10) + ' POTENTIAL INVESTORS (list view) ' + ('#' * 10), color='blue')

    last_year = investors_concepts['Year'].max() - 1

    # Select only investor-concept edges from last year
    investors_concepts = investors_concepts[(investors_concepts['Year'] == last_year) & (investors_concepts['PageID'].isin(unit_concept_ids))]

    # # Select top n rows such that they include the top 5 investors
    # top_investors_concepts = investors_concepts.sort_values(by='SumAmount', ascending=False).reset_index(drop=True)
    # idx = top_investors_concepts['InvestorID'].drop_duplicates().head(5).index[-1]
    # top_investors_concepts = top_investors_concepts[:idx + 1]

    def str_unpack(s):
        if len(s) == 1:
            return s.iloc[0]

        if len(s) == 2:
            return f'{s.iloc[0]} and {s.iloc[1]}'

        else:
            return f'{s.iloc[0]}, {str_unpack(s.iloc[1:])}'

    for name, group in investors_concepts.groupby(by=['InvestorID', 'InvestorName']):
        group = group.sort_values(by=['CountAmount'], ascending=False).reset_index(drop=True)

        active_concepts_str = str_unpack(group['PageTitle'].head(3))
        likely_concepts_str = str_unpack(investors_concepts_jaccard.loc[investors_concepts_jaccard['InvestorID'] == name[0], 'PageTitle'].head(3))

        cprint(f'\U0001F4BC {name[1]}', color='blue')
        cprint(f'    · Actively invested last year in {active_concepts_str}.', color='blue')
        cprint(f'    · Likely to invest in {likely_concepts_str}.', color='blue')

    cprint('')


def show_matchmaking_chart_view(investors_concepts, unit_concepts, unit_investors, time_window):
    cprint(('#' * 10) + ' POTENTIAL INVESTORS (chart view) ' + ('#' * 10), color='blue')
    cprint('(see plot...)', color='blue')

    # Sample main unit concepts for display
    unit_sample_concepts = list(unit_concepts.sort_values(by='Score', ascending=False)['PageTitle'].head(3))

    # Renormalise unit concept scores to add to 1
    sum = unit_concepts['Score'].sum()
    unit_concepts['Score'] = unit_concepts['Score'] / sum if sum else 0

    # Filter investors-concepts by time window and extract relevant columns
    investors_concepts = investors_concepts[investors_concepts['Year'].isin(time_window)]
    investors_concepts = investors_concepts[['InvestorID', 'InvestorName', 'Year', 'PageID', 'PageTitle', 'CountAmount']]

    # Add unit concept scores to investors-concepts DataFrame
    investors_concepts = pd.merge(
        investors_concepts,
        unit_concepts.rename(columns={'Score': 'UnitPageScore'}).drop(columns=['PageTitle']),
        how='inner',
        on='PageID'
    )

    # Rescale CountAmount according to relevance within unit
    investors_concepts['NormCountAmount'] = investors_concepts['CountAmount'] * investors_concepts['UnitPageScore']

    # Group by investor and year
    investors = investors_concepts.groupby(by=['InvestorID', 'InvestorName', 'Year']).aggregate({'NormCountAmount': 'sum'}).reset_index()

    # Add missing years
    skeleton = pd.merge(
        pd.DataFrame({'InvestorID': investors['InvestorID'].drop_duplicates()}),
        pd.DataFrame({'Year': time_window}),
        how='cross'
    )
    investors = pd.merge(skeleton, investors, how='left', on=['InvestorID', 'Year']).fillna(0)

    # Create variation column
    investors['Variation'] = investors['NormCountAmount'].pct_change().fillna(0)

    # Add investor-unit yearly affinity
    investors = pd.merge(
        investors,
        unit_investors.rename(columns={'Score': 'UnitInvestorScore'}).drop(columns=['InvestorName']),
        how='left',
        on=['InvestorID', 'Year']
    )

    for (investor_id, investor_name), investor_group in investors.groupby(by=['InvestorID', 'InvestorName']):

        fig, ax = plt.subplots(dpi=150, figsize=(8, 4))
        fig.set_tight_layout(True)

        x = investor_group['Year']
        y = investor_group['UnitInvestorScore']
        sizes = 200 * investor_group['NormCountAmount']
        colors = np.tanh(investor_group['Variation'])
        fontsizes = 8 + investor_group['NormCountAmount']

        ax.scatter(x, y, s=sizes, c=colors, cmap='RdYlGn')

        for i, (year, y) in enumerate(investor_group[['Year', 'UnitInvestorScore']].itertuples(index=False, name=None)):
            ax.annotate(year, (year, y), ha='center', va='center', fontsize=fontsizes.iloc[i], c='#444444')

        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlabel('Time (years)')
        ax.set_ylabel(f'Investor-Unit affinity\n({", ".join(unit_sample_concepts)})')
        ax.set_title(f'POTENTIAL INVESTORS (chart view, affinity)\n{investor_name}')

        ###########################################################

        plt.savefig(f'img/{unit_id}-{investor_id}')

    cprint('')


def main(unit_id):
    # Instantiate db interface to communicate with database
    db = DB(load_db_config())

    # Unit for which to produce results
    cprint(('#' * 20) + f' {unit_id} ' + ('#' * 20), color='green')

    # Define time window to consider
    min_year = 2015
    max_year = 2022
    time_window = list(range(min_year, max_year))

    ############################################################

    # Fetch crunchbase concepts
    table_name = 'graph_piper.Edges_N_Organisation_N_Concept'
    fields = ['DISTINCT PageID']
    cb_concept_ids = list(pd.DataFrame(db.find(table_name, fields=fields), columns=['PageID'])['PageID'])

    # Fetch concept title mapping
    table_name = 'graph_piper.Nodes_N_Concept_T_Title'
    fields = ['PageID', 'PageTitle']
    conditions = {'PageID': cb_concept_ids}
    concept_titles = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    # Fetch unit concepts and normalize score
    table_name = 'graph_piper.Edges_N_Unit_N_Concept_T_Research'
    fields = ['PageID', 'Score']
    conditions = {'UnitID': unit_id, 'PageID': cb_concept_ids}
    unit_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    max_score = unit_concepts['Score'].max()
    unit_concepts['Score'] = unit_concepts['Score'] / max_score if max_score >= 0 else 0
    unit_concept_ids = list(unit_concepts['PageID'])

    ############################################################

    # Add title information
    unit_concepts = pd.merge(unit_concepts, concept_titles, how='left', on='PageID')

    # Fetch concepts
    table_name = 'aitor.Nodes_N_Concept_T_Years'
    fields = ['PageID', 'Year', 'CountAmount', 'MinAmount', 'MaxAmount', 'MedianAmount', 'SumAmount', 'ScoreQuadCount']
    conditions = {'PageID': unit_concept_ids, 'Year': time_window}
    concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    # Complete missing combinations for (PageID, Year)
    concepts_skeleton = pd.merge(pd.DataFrame({'PageID': unit_concept_ids}), pd.DataFrame({'Year': time_window}), how='cross')
    concepts = pd.merge(concepts_skeleton, concepts, how='left', on=['PageID', 'Year']).fillna(0)

    # Add title information
    concepts = pd.merge(concepts, concept_titles, how='left', on='PageID')

    ############################################################

    table_name = 'aitor.Edges_N_Investor_N_Concept_T_Years'
    fields = ['InvestorID', 'PageID', 'Year', 'CountAmount']
    conditions = {'PageID': unit_concept_ids, 'Year': time_window}
    investors_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    concepts = pd.merge(
        concepts,
        investors_concepts.groupby(by=['PageID', 'Year']).agg(CountInvestor=('InvestorID', 'count'), CountInvestment=('CountAmount', 'sum')).reset_index(),
        how='left',
        on=['PageID', 'Year']
    ).fillna(0)

    ############################################################

    show_trends(unit_id, unit_concepts, concepts)

    ############################################################

    # Fetch investor-unit table for given unit
    table_name = 'aitor.Edges_N_Investor_N_Unit_T_Years'
    fields = ['InvestorID', 'Year', 'Score']
    conditions = {'UnitID': unit_id}
    unit_investors = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)
    unit_investors = unit_investors.sort_values(by=['InvestorID', 'Year']).reset_index(drop=True)

    # Extract unit investor ids
    unit_investor_ids = list(unit_investors['InvestorID'].drop_duplicates())

    if not unit_investor_ids:
        unit_investor_ids = ['']

    ############################################################

    # Fetch investor names
    table_name = 'graph_piper.Nodes_N_Organisation'
    fields = ['OrganisationID', 'OrganisationName']
    conditions = {'OrganisationID': unit_investor_ids}
    org_investor_names = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=['InvestorID', 'InvestorName'])

    table_name = 'graph_piper.Nodes_N_Person'
    fields = ['PersonID', 'FullName']
    conditions = {'PersonID': unit_investor_ids}
    person_investor_names = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=['InvestorID', 'InvestorName'])

    investor_names = pd.concat([org_investor_names, person_investor_names])

    ############################################################

    # Fetch investor-concepts table only for unit investors
    table_name = 'aitor.Edges_N_Investor_N_Concept_T_Years'
    fields = ['InvestorID', 'PageID', 'Year', 'CountAmount', 'MinAmount', 'MaxAmount', 'MedianAmount', 'SumAmount', 'ScoreQuadCount']
    conditions = {'InvestorID': unit_investor_ids, 'Year': time_window}
    investors_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    ############################################################

    # Fetch investor-concepts Jaccard table only for unit investors and concepts
    table_name = 'aitor.Edges_N_Investor_N_Concept_T_Jaccard'
    fields = ['InvestorID', 'PageID', 'Jaccard_000']
    conditions = {'InvestorID': unit_investor_ids, 'PageID': unit_concept_ids}
    investors_concepts_jaccard = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)
    investors_concepts_jaccard = investors_concepts_jaccard.sort_values(by=['InvestorID', 'Jaccard_000'], ascending=[True, False]).reset_index(drop=True)

    ############################################################

    # Add investor and concept names to the tables we just fetched
    unit_investors = pd.merge(unit_investors, investor_names, how='left', on='InvestorID')

    investors_concepts = pd.merge(investors_concepts, concept_titles, how='left', on='PageID')
    investors_concepts = pd.merge(investors_concepts, investor_names, how='left', on='InvestorID')

    investors_concepts_jaccard = pd.merge(investors_concepts_jaccard, concept_titles, how='left', on='PageID')
    investors_concepts_jaccard = pd.merge(investors_concepts_jaccard, investor_names, how='left', on='InvestorID')

    ############################################################

    show_matchmaking_list_view(investors_concepts_jaccard, investors_concepts, unit_concept_ids)

    show_matchmaking_chart_view(investors_concepts, unit_concepts, unit_investors, time_window)


if __name__ == '__main__':
    pd.set_option('display.max_rows', 400)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    for unit_id in ['LCAV', 'CHILI', 'NANOLAB', 'ECOS']:
        main(unit_id)
