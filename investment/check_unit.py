import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from investment.projections import get_affinities, project

from interfaces.db import DB

from utils.text.io import cprint


def show_trends(unit_concepts, concepts):
    cprint(('#' * 10) + ' INVESTMENT TRENDS ' + ('#' * 10), color='blue')

    # Extract concepts to display
    top_concept_ids = list(unit_concepts.sort_values(by='Score', ascending=False).head(5)['PageID'])
    concepts = concepts[concepts['PageID'].isin(top_concept_ids)]

    # Print summary for last year
    last_year = concepts['Year'].max() - 1
    for name, group in concepts.groupby(by=['PageID', 'PageTitle']):
        last_year_amount = group.loc[group['Year'] == last_year, 'SumAmount'].min()
        second_last_year_amount = group.loc[group['Year'] == (last_year - 1), 'SumAmount'].min()

        perc_increment = 100 * (second_last_year_amount / last_year_amount - 1) if last_year_amount else 0

        cprint(f'\U0001F30D {name[1]}', color='blue')
        cprint(f'    · {last_year_amount / 1e6:.2f} M$ raised in {last_year}, {perc_increment:+.2f}% with respect to {last_year - 1}.', color='blue')

    # Plot Year vs Amounts
    fig, axs = plt.subplots(3, 1, sharex=True, dpi=300)

    for name, group in concepts.groupby(by=['PageID', 'PageTitle']):
        axs[0].plot(group['Year'], group['SumAmount'] / 1e6, label=name[1])
        axs[1].plot(group['Year'], group['CountAmount'], label=name[1])
        axs[2].plot(group['Year'], group['CountInvestor'], label=name[1])

    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('M$')
    axs[0].set_title('Investment amounts')

    axs[1].set_xlabel('Year')
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[1].set_title('Number of investments')

    axs[2].set_xlabel('Year')
    axs[2].set_title('Number of investors')

    fig.suptitle('INVESTMENT TRENDS (chart view)')
    fig.set_tight_layout(True)

    plt.show()

    cprint('')


def show_matchmaking_list_view(investors_concepts_jaccard, investors_concepts, unit_concept_ids):
    cprint(('#' * 10) + ' POTENTIAL INVESTORS (list view) ' + ('#' * 10), color='blue')

    last_year = investors_concepts['Year'].max() - 1

    # Select only investor-concept edges from last year
    investors_concepts = investors_concepts[(investors_concepts['Year'] == last_year) & (investors_concepts['PageID'].isin(unit_concept_ids))]

    # Select top n rows such that they include the top 5 investors
    top_investors_concepts = investors_concepts.sort_values(by='SumAmount', ascending=False).reset_index(drop=True)
    idx = top_investors_concepts['InvestorID'].drop_duplicates().head(5).index[-1]
    top_investors_concepts = top_investors_concepts[:idx + 1]

    def str_unpack(s):
        if len(s) == 1:
            return s.iloc[0]

        if len(s) == 2:
            return f'{s.iloc[0]} and {s.iloc[1]}'

        else:
            return f'{s.iloc[0]}, {str_unpack(s.iloc[1:])}'

    for name, group in top_investors_concepts.groupby(by=['InvestorID', 'InvestorName']):
        active_concepts_str = str_unpack(group['PageTitle'])
        likely_concepts_str = str_unpack(investors_concepts_jaccard.loc[investors_concepts_jaccard['InvestorID'] == name[0], 'PageTitle'])

        cprint(f'\U0001F4BC {name[1]}', color='blue')
        cprint(f'    · Actively invested last year in {active_concepts_str}.', color='blue')
        cprint(f'    · Likely to invest in {likely_concepts_str}.', color='blue')

    cprint('')


def show_matchmaking_chart_view(investors_concepts, unit_concepts, historical_time_window):
    cprint(('#' * 10) + ' POTENTIAL INVESTORS (chart view) ' + ('#' * 10), color='blue')
    cprint('(see plot...)', color='blue')

    # Normalize unit concepts' scores
    min_score = unit_concepts['Score'].min()
    max_score = unit_concepts['Score'].max()
    unit_concepts['Score'] = ((unit_concepts['Score'] - min_score) / (max_score - min_score)) if min_score < max_score else 0
    unit_concepts = unit_concepts[unit_concepts['Score'] > 0]

    # Sample main unit concepts for display
    unit_sample_concepts = list(unit_concepts.sort_values(by='Score', ascending=False)['PageTitle'].head(3))

    for (investor_id, investor_name), investor_group in investors_concepts.groupby(by=['InvestorID', 'InvestorName']):
        # Filter concepts in historical time window
        investor_historical_concepts = investor_group[investor_group['Year'].isin(historical_time_window)].groupby(by=['PageID', 'PageTitle'])['SumAmount'].sum().reset_index()
        investor_historical_concepts = investor_historical_concepts.rename(columns={'SumAmount': 'Score'})

        # Normalize unit concepts' scores
        min_score = investor_historical_concepts['Score'].min()
        max_score = investor_historical_concepts['Score'].max()
        investor_historical_concepts['Score'] = ((investor_historical_concepts['Score'] - min_score) / (max_score - min_score)) if min_score < max_score else 0
        investor_historical_concepts = investor_historical_concepts[investor_historical_concepts['Score'] > 0]

        investor_sample_concepts = investor_historical_concepts.sort_values(by='Score', ascending=False)['PageTitle'].head(3)

        yearly_concepts = {}
        for year, group in investor_group.groupby(by='Year'):
            year_concepts = group[['PageID', 'SumAmount']]
            year_concepts = year_concepts.rename(columns={'SumAmount': 'Score'})

            min_score = year_concepts['Score'].min()
            max_score = year_concepts['Score'].max()
            year_concepts['Score'] = ((year_concepts['Score'] - min_score) / (max_score - min_score)) if min_score < max_score else 0
            year_concepts = year_concepts[year_concepts['Score'] > 0]

            yearly_concepts[year] = year_concepts

        hist_year_affinities = get_affinities(investor_historical_concepts[['PageID', 'Score']], yearly_concepts)
        unit_year_affinities = get_affinities(unit_concepts[['PageID', 'Score']], yearly_concepts)
        hist_year_affinities = pd.DataFrame({'Year': hist_year_affinities.keys(), 'HistYear': hist_year_affinities.values()})
        unit_year_affinities = pd.DataFrame({'Year': unit_year_affinities.keys(), 'UnitYear': unit_year_affinities.values()})
        affinities = pd.merge(hist_year_affinities, unit_year_affinities, how='inner', on='Year')

        projections = project(investor_historical_concepts, unit_concepts[['PageID', 'Score']], yearly_concepts)
        projections = pd.DataFrame({'Year': projections.keys(), 'Projection': projections.values()})

        investor_yearly_totals = investor_group.groupby(by='Year')['SumAmount'].sum().reset_index()
        investor_yearly_totals['Variation'] = investor_yearly_totals['SumAmount'].pct_change().fillna(0)

        ############################################################

        sizes = 200 + 500 * (investor_yearly_totals['SumAmount'] / 1e9)
        fontsizes = 8 + (investor_yearly_totals['SumAmount'] / 1e9)
        colors = np.tanh(investor_yearly_totals['Variation'])

        ############################################################

        fig, ax = plt.subplots(dpi=150, figsize=(8, 4))
        fig.set_tight_layout(True)
        ax.scatter(affinities['HistYear'], affinities['UnitYear'], s=sizes, c=colors, cmap='RdYlGn')

        for i, (year, x, y) in enumerate(affinities.itertuples(index=False, name=None)):
            ax.annotate(year, (x, y), ha='center', va='center', fontsize=fontsizes[i], c='darkgray')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlabel(f"Investor's historical concepts\n({', '.join(investor_sample_concepts)})")
        ax.set_ylabel(f"Unit's research domain\n({', '.join(unit_sample_concepts)})")
        ax.set_title(f'POTENTIAL INVESTORS (chart view, amounts)\n{investor_name}')

        ############################################################

        fig, ax = plt.subplots(dpi=150, figsize=(8, 4))
        fig.set_tight_layout(True)
        ax.hlines(y=0.5, xmin=(projections['Year'].min() - 1), xmax=(projections['Year'].max() + 1), linewidth=0.5, color='lightgray', zorder=-1)
        ax.scatter(projections['Year'], projections['Projection'], s=sizes, c=colors, cmap='RdYlGn')

        for i, (year, y) in enumerate(projections.itertuples(index=False, name=None)):
            ax.annotate(year, (year, y), ha='center', va='center', fontsize=fontsizes[i], c='darkgray')

        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        unit_sample_concepts_str = '\n'.join(unit_sample_concepts)
        investor_sample_concepts_str = '\n'.join(investor_sample_concepts)
        ylabel = f"""
            Unit's research domain\n({unit_sample_concepts_str})\n\n\n\n\n\n\n\n\n\n
            Investor's historical concepts\n({investor_sample_concepts_str})
        """

        ax.set_xlabel(f"Time (years)")
        ax.set_ylabel(ylabel, rotation=0, ha='right', va='center')
        ax.set_title(f'POTENTIAL INVESTORS (chart view, projections)\n{investor_name}')

        ###########################################################

        fig, ax = plt.subplots(dpi=150, figsize=(8, 4))
        fig.set_tight_layout(True)
        ax.scatter(affinities['Year'], affinities['UnitYear'], s=sizes, c=colors, cmap='RdYlGn')

        for i, (year, _, y) in enumerate(affinities.itertuples(index=False, name=None)):
            ax.annotate(year, (year, y), ha='center', va='center', fontsize=fontsizes[i], c='darkgray')

        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlabel(f"Time (years)")
        ax.set_ylabel(f"Investor-Unit affinity\n({', '.join(unit_sample_concepts)})")
        ax.set_title(f'POTENTIAL INVESTORS (chart view, affinity)\n{investor_name}')

        ###########################################################

        # print(investor_name)
        # for y, gp in investor_group.groupby('Year'):
        #     print(y)
        #     print(gp)

    plt.show()

    cprint('')


def main():
    # Instantiate db interface to communicate with database
    db = DB()

    # Define unit for which to produce results
    unit_id = 'LCAV'
    cprint(('#' * 20) + f' {unit_id} ' + ('#' * 20), color='green')

    # Define time window to consider
    min_year = 1998
    max_year = 2022
    time_window = list(range(min_year, max_year))
    historical_time_window = list(range(min_year, min_year + 3))

    ############################################################

    # Fetch crunchbase concepts
    table_name = 'graph.Edges_N_Organisation_N_Concept'
    fields = ['DISTINCT PageID']
    cb_concept_ids = list(pd.DataFrame(db.find(table_name, fields=fields), columns=['PageID'])['PageID'])

    # Fetch concept title mapping
    table_name = 'graph.Nodes_N_Concept_T_Title'
    fields = ['PageID', 'PageTitle']
    conditions = {'PageID': cb_concept_ids}
    concept_titles = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    # Fetch unit concepts and normalize score
    table_name = 'graph.Edges_N_Unit_N_Concept_T_Research'
    fields = ['PageID', 'Score']
    conditions = {'UnitID': unit_id, 'PageID': cb_concept_ids}
    unit_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    min_score = unit_concepts['Score'].min()
    max_score = unit_concepts['Score'].max()
    unit_concepts['Score'] = (unit_concepts['Score'] - min_score) / (max_score - min_score) if max_score != min_score else 0
    unit_concept_ids = list(unit_concepts['PageID'])

    # Add title information
    unit_concepts = pd.merge(unit_concepts, concept_titles, how='left', on='PageID')

    # Fetch concepts
    table_name = 'ca_temp.Nodes_N_Concept'
    fields = ['PageID', 'Year', 'CountAmount', 'MinAmount', 'MaxAmount', 'MedianAmount', 'SumAmount', 'ScoreQuadCount']
    conditions = {'PageID': unit_concept_ids, 'Year': time_window}
    concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    # Complete missing combinations for (PageID, Year)
    concepts_skeleton = pd.merge(pd.DataFrame({'PageID': unit_concept_ids}), pd.DataFrame({'Year': time_window}), how='cross')
    concepts = pd.merge(concepts_skeleton, concepts, how='left', on=['PageID', 'Year']).fillna(0)

    # Add title information
    concepts = pd.merge(concepts, concept_titles, how='left', on='PageID')

    ############################################################

    # Extract diluted investors (investors with low ratio of investments in unit concepts)
    table_name = 'ca_temp.Edges_N_Investor_N_Concept'
    fields = ['InvestorID', 'PageID', 'Year']
    conditions = {'PageID': unit_concept_ids, 'Year': time_window}
    investors_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    concepts = pd.merge(
        concepts,
        investors_concepts.groupby(by=['PageID', 'Year']).agg(CountInvestor=('InvestorID', 'count')).reset_index(),
        how='left',
        on=['PageID', 'Year']
    ).fillna(0)

    ############################################################

    show_trends(unit_concepts, concepts)

    exit(0)

    ############################################################

    # Extract diluted investors (investors with low ratio of investments in unit concepts)
    table_name = 'ca_temp.Edges_N_Investor_N_Concept'
    fields = ['InvestorID', 'PageID', 'Year', 'CountAmountRatio']
    conditions = {'PageID': unit_concept_ids, 'Year': time_window}
    diluted_investors_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    diluted_investors_concepts = diluted_investors_concepts.groupby(by=['InvestorID'])['CountAmountRatio'].mean().reset_index()
    diluted_investors_concepts = diluted_investors_concepts[diluted_investors_concepts['CountAmountRatio'] < 0.2]
    diluted_investor_ids = list(diluted_investors_concepts['InvestorID'])

    diluted_investor_ids = ['aaa']

    # Compute top potential investors for unit (investors with highest Jaccard index excluding diluted investors)
    table_name = 'ca_temp.Edges_N_Investor_N_Concept_T_Jaccard'
    fields = ['InvestorID', 'PageID', 'Jaccard_110']
    conditions = {'NOT': {'InvestorID': diluted_investor_ids}, 'PageID': unit_concept_ids}
    investors_concepts_jaccard = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)
    investors_concepts_jaccard = investors_concepts_jaccard.sort_values(by='Jaccard_110', ascending=False).reset_index(drop=True)

    idx = investors_concepts_jaccard['InvestorID'].drop_duplicates().head(5).index[-1]
    investors_concepts_jaccard = investors_concepts_jaccard[:idx + 1]
    investor_ids = list(investors_concepts_jaccard['InvestorID'].drop_duplicates())

    # Fetch investor names
    table_name = 'graph.Nodes_N_Organisation'
    fields = ['OrganisationID', 'OrganisationName']
    conditions = {'OrganisationID': investor_ids}
    org_investor_names = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=['InvestorID', 'InvestorName'])

    table_name = 'graph.Nodes_N_Person'
    fields = ['PersonID', 'FullName']
    conditions = {'PersonID': investor_ids}
    person_investor_names = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=['InvestorID', 'InvestorName'])

    investor_names = pd.concat([org_investor_names, person_investor_names])

    # Add concept title information
    investors_concepts_jaccard = pd.merge(investors_concepts_jaccard, concept_titles, how='left', on='PageID')

    # Add investor name information
    investors_concepts_jaccard = pd.merge(investors_concepts_jaccard, investor_names, how='left', on='InvestorID')

    # Fetch investor-concepts table only for top investors
    table_name = 'ca_temp.Edges_N_Investor_N_Concept'
    fields = ['InvestorID', 'PageID', 'Year', 'CountAmount', 'MinAmount', 'MaxAmount', 'MedianAmount', 'SumAmount', 'ScoreQuadCount', 'CountAmountRatio']
    conditions = {'InvestorID': investor_ids, 'Year': time_window}
    investors_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    # Add concept title information
    investors_concepts = pd.merge(investors_concepts, concept_titles, how='left', on='PageID')

    # Add investor name information
    investors_concepts = pd.merge(investors_concepts, investor_names, how='left', on='InvestorID')

    ############################################################

    show_matchmaking_list_view(investors_concepts_jaccard, investors_concepts, unit_concept_ids)

    show_matchmaking_chart_view(investors_concepts, unit_concepts, historical_time_window)

    ############################################################


if __name__ == '__main__':
    pd.set_option('display.max_rows', 400)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    main()
