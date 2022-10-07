import pandas as pd

from interfaces.db import DB


def get_ball(x, concepts_concepts):
    forward_sphere = pd.merge(x[['PageID']].rename(columns={'PageID': 'SourcePageID'}), concepts_concepts, how='inner', on='SourcePageID')[['TargetPageID']].rename(columns={'TargetPageID': 'PageID'})
    backward_sphere = pd.merge(x[['PageID']].rename(columns={'PageID': 'TargetPageID'}), concepts_concepts, how='inner', on='TargetPageID')[['SourcePageID']].rename(columns={'SourcePageID': 'PageID'})
    sphere = pd.concat([forward_sphere, backward_sphere]).drop_duplicates()

    ball = pd.concat([x[['PageID']], sphere]).drop_duplicates()
    ball = pd.merge(ball, x, how='left', on='PageID').fillna(0)

    return ball


def compute_pseudodistance_factor(x, y):
    # Intersect both subsets and compute the sum of minima of scores
    intersection = pd.merge(x, y, how='inner', on='PageID')
    min_intersection = intersection[['Score_x', 'Score_y']].min(axis=1).sum()

    # If the sum of minima of scores is zero, then factor = 1 - 0/* = 1
    if min_intersection <= 0:
        return 1

    # Compute the union of both subsets and the sum of maxima of scores
    union = pd.merge(x, y, how='outer', on='PageID')
    max_union = union[['Score_x', 'Score_y']].max(axis=1).sum()

    # The sum of maxima of scores should be nonzero, since it is greater than the sum of minima, but check for sanity.
    if max_union <= 0:
        return 1

    # Compute factor and return it
    factor = 1 - min_intersection / max_union
    return factor


def project(bottom, top, mid):
    """
    Returns a list of values in [0, 1] representing the position of each of the DataFrames in x in the
    segment (a continuum of concept subsets) defined by bottom-top. A low (high) value for a given concept subset
    indicates that these concepts are more related to those in bottom (top) than those in top (bottom).

    Args:
        bottom: A pandas DataFrame like
            PageID         Score
              8525      0.639206
            235757      0.431793
            233488      0.389501
        top: A pandas DataFrame like
            PageID         Score
              8525      0.639206
            235757      0.431793
            233488      0.389501
        mid: A dict of pandas Dataframes like
            PageID         Score
              8525      0.639206
            235757      0.431793
            233488      0.389501

    Returns (dict[float]): A dict of values in [0, 1], indexed by the same keys as mid.
    """

    # Instantiate db interface to communicate with database
    db = DB()

    # Obtain all concept ids to filter the table to fetch
    concept_ids = list(bottom['PageID']) + list(top['PageID'])
    for x in mid.values():
        concept_ids.extend(list(x['PageID']))

    # Fetch table with concept-concept edges
    table_name = 'graph.Edges_N_Concept_N_Concept_T_GraphScore'
    fields = ['SourcePageID', 'TargetPageID']
    conditions = {'SourcePageID': concept_ids, 'TargetPageID': concept_ids}
    concepts_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    # Precompute bottom and top 1-balls
    bottom_ball = get_ball(bottom, concepts_concepts)
    top_ball = get_ball(top, concepts_concepts)

    # Iterate over mid and compute projections
    projections = {}
    for key, x in mid.items():
        # Compute x 1-ball
        x_ball = get_ball(x, concepts_concepts)

        # Compute pseudodistance x-bottom
        x_bottom_factor = compute_pseudodistance_factor(x, bottom)
        x_bottom_ball_factor = compute_pseudodistance_factor(x_ball, bottom_ball)
        x_bottom_pseudodistance = x_bottom_factor * x_bottom_ball_factor

        # If pseudodistance x-bottom is zero, the projection is zero
        if x_bottom_pseudodistance <= 0:
            projections[key] = 0
            continue

        # Compute pseudodistance x-top
        x_top_factor = compute_pseudodistance_factor(x, top)
        x_top_ball_factor = compute_pseudodistance_factor(x_ball, top_ball)
        x_top_pseudodistance = x_top_factor * x_top_ball_factor

        # The projection is the ratio of x-bottom and (x-bottom + x-top) pseudodistances
        projection = x_bottom_pseudodistance / (x_bottom_pseudodistance + x_top_pseudodistance)
        projections[key] = projection

    return projections


def main():

    # Instantiate db interface to communicate with database
    db = DB()

    # Define all history time window, investments outside it will be ignored.
    min_year = 2015
    max_year = 2023

    # Digital signal processing, Sensor, Machine learning
    lab_concepts = pd.DataFrame({'PageID': [8525, 235757, 233488], 'Score': [0.639206, 0.431793, 0.389501]})
    lab_concept_ids = list(lab_concepts['PageID'])

    # Fetch Jaccard table
    table_name = 'ca_temp.Edges_N_Investor_N_Concept_T_Jaccard'
    fields = ['InvestorID', 'PageID', 'Jaccard_110']
    conditions = {'PageID': lab_concept_ids}
    investors_concepts_jaccard = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    # Extract likely investors
    investors_concepts_jaccard = investors_concepts_jaccard.sort_values('Jaccard_110', ascending=False)
    investor_ids = list(investors_concepts_jaccard['InvestorID'][:5].drop_duplicates())

    ############################################################

    table_name = 'ca_temp.Edges_N_Investor_N_Concept'
    fields = ['InvestorID', 'PageID', 'Year', 'ScoreQuadCount']
    columns = ['InvestorID', 'PageID', 'Year', 'Score']
    conditions = {'InvestorID': investor_ids, 'Year': list(range(min_year, max_year))}
    investors_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=columns)

    historical_years_idx = investors_concepts['Year'].isin(list(range(min_year, min_year + 3)))
    historical_concept_subsets = investors_concepts.loc[historical_years_idx, ['InvestorID', 'PageID', 'Score']].groupby(by=['InvestorID', 'PageID']).mean().reset_index()

    for investor_id in investor_ids:
        historical_concept_subset = historical_concept_subsets.loc[historical_concept_subsets['InvestorID'] == investor_id, ['PageID', 'Score']].reset_index(drop=True)

        yearly_concept_subsets = {}
        for year in range(min_year, max_year):
            x = investors_concepts.loc[(investors_concepts['InvestorID'] == investor_id) & (investors_concepts['Year'] == year), ['PageID', 'Score']].reset_index(drop=True)
            yearly_concept_subsets[year] = x

        projections = project(historical_concept_subset, lab_concepts, yearly_concept_subsets)

        print(f'Investor {investor_id}')
        print(f'Projections {projections}')


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    main()
