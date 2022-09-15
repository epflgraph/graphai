import pandas as pd

from interfaces.db import DB

from utils.breadcrumb import Breadcrumb


def get_investor_type(db, investor_id):
    table_name = 'graph.Nodes_N_Organisation'
    fields = ['OrganisationID', 'OrganisationName']
    conditions = {'OrganisationID': investor_id}
    orgs = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    if len(orgs) > 0:
        return 'Organisation'
    else:
        return 'Person'


def get_investor_name(db, investor_id):
    investor_type = get_investor_type(db, investor_id)

    if investor_type == 'Organisation':
        table_name = 'graph.Nodes_N_Organisation'
        fields = ['OrganisationID', 'OrganisationName']
        conditions = {'OrganisationID': investor_id}
        orgs = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)
        try:
            return orgs['OrganisationName'][0]
        except IndexError:
            return '__UNKNOWN_ORG__'
    else:
        table_name = 'graph.Nodes_N_Person'
        fields = ['PersonID', 'FullName']
        conditions = {'PersonID': investor_id}
        people = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)
        try:
            return people['FullName'][0]
        except IndexError:
            return '__UNKNOWN_PERSON__'


def get_fr_date(db, fr_id):
    table_name = 'graph.Nodes_N_FundingRound'
    fields = ['FundingRoundID', 'FundingRoundDate']
    conditions = {'FundingRoundID': fr_id}
    frs = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    return frs['FundingRoundDate'][0]


def get_concept_name(db, concept_id):
    table_name = 'graph.Nodes_N_Concept_T_Title'
    fields = ['PageID', 'PageTitle']
    conditions = {'PageID': concept_id}
    concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    try:
        return concepts['PageTitle'][0]
    except IndexError:
        return '__UNKNOWN_CONCEPT__'


############################################################


def get_investor_fr_ids(db, fr_ids, investor_id):
    table_name = 'graph.Edges_N_Organisation_N_FundingRound'
    fields = ['OrganisationID', 'FundingRoundID']
    conditions = {'OrganisationID': investor_id, 'FundingRoundID': fr_ids, 'Action': 'Invested in'}
    org_investor_fr_ids = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)['FundingRoundID'].tolist()

    table_name = 'graph.Edges_N_Person_N_FundingRound'
    fields = ['PersonID', 'FundingRoundID']
    conditions = {'PersonID': investor_id, 'FundingRoundID': fr_ids, 'Action': 'Invested in'}
    person_investor_fr_ids = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)['FundingRoundID'].tolist()

    return list(dict.fromkeys(org_investor_fr_ids + person_investor_fr_ids))


def get_fr_investor_ids(db, fr_id):
    table_name = 'graph.Edges_N_Organisation_N_FundingRound'
    fields = ['OrganisationID', 'FundingRoundID']
    conditions = {'FundingRoundID': fr_id, 'Action': 'Invested in'}
    org_investor_ids = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)['OrganisationID'].tolist()

    table_name = 'graph.Edges_N_Person_N_FundingRound'
    fields = ['PersonID', 'FundingRoundID']
    conditions = {'FundingRoundID': fr_id, 'Action': 'Invested in'}
    person_investor_ids = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)['PersonID'].tolist()

    return list(dict.fromkeys(org_investor_ids + person_investor_ids))


def get_fr_investee_ids(db, fr_id):
    table_name = 'graph.Edges_N_Organisation_N_FundingRound'
    fields = ['OrganisationID', 'FundingRoundID']
    conditions = {'FundingRoundID': fr_id, 'Action': 'Raised from'}
    fr_investee_ids = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)['OrganisationID'].tolist()

    return list(dict.fromkeys(fr_investee_ids))


def get_investee_fr_ids(db, fr_ids, investee_id):
    table_name = 'graph.Edges_N_Organisation_N_FundingRound'
    fields = ['OrganisationID', 'FundingRoundID']
    conditions = {'OrganisationID': investee_id, 'FundingRoundID': fr_ids,  'Action': 'Raised from'}
    investee_fr_ids = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)['FundingRoundID'].tolist()

    return list(dict.fromkeys(investee_fr_ids))


def get_investee_concept_ids(db, investee_id):
    table_name = 'graph.Edges_N_Organisation_N_Concept'
    fields = ['OrganisationID', 'PageID']
    conditions = {'OrganisationID': investee_id}
    investee_concept_ids = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)['PageID'].tolist()

    return list(dict.fromkeys(investee_concept_ids))


def get_concept_investee_ids(db, concept_id):
    table_name = 'graph.Edges_N_Organisation_N_Concept'
    fields = ['OrganisationID', 'PageID']
    conditions = {'PageID': concept_id}
    concept_investee_ids = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)['OrganisationID'].tolist()

    return list(dict.fromkeys(concept_investee_ids))


############################################################


def get_investor_investor_ids(db, fr_ids, investor_id):
    investor_fr_ids = get_investor_fr_ids(db, fr_ids, investor_id)
    investor_investor_ids = get_fr_investor_ids(db, investor_fr_ids)
    investor_investor_ids.remove(investor_id)

    return investor_investor_ids


def get_investor_concept_ids(db, fr_ids, investor_id):
    investor_fr_ids = get_investor_fr_ids(db, fr_ids, investor_id)
    investor_investee_ids = get_fr_investee_ids(db, investor_fr_ids)
    investor_concept_ids = get_investee_concept_ids(db, investor_investee_ids)

    return investor_concept_ids


def get_concept_investor_ids(db, fr_ids, concept_id):
    concept_investee_ids = get_concept_investee_ids(db, concept_id)
    concept_fr_ids = get_investee_fr_ids(db, fr_ids, concept_investee_ids)
    concept_investor_ids = get_fr_investor_ids(db, concept_fr_ids)

    return concept_investor_ids


def get_concept_concept_ids(db, concept_id):
    table_name = 'graph.Edges_N_Organisation_N_Concept'
    fields = ['DISTINCT PageID']
    all_concept_ids = pd.DataFrame(db.find(table_name, fields=fields), columns=['PageID'])['PageID'].tolist()
    all_concept_ids = list(dict.fromkeys(all_concept_ids))

    table_name = 'graph.Edges_N_Concept_N_Concept_T_GraphScore'
    fields = ['SourcePageID', 'TargetPageID']
    conditions = {'SourcePageID': all_concept_ids, 'TargetPageID': concept_id}
    concept_ids = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)['SourcePageID'].tolist()
    concept_ids = list(dict.fromkeys(concept_ids))

    return concept_ids


############################################################


def main():
    # Initialize breadcrumb to log and keep track of time
    bc = Breadcrumb(color='blue', time_color='gray')

    # Instantiate db interface to communicate with database
    db = DB()

    # Investor-concept pair to be checked
    investor_id = '1d474074-5b12-4ac8-9461-d7a9647a04d7'
    concept_id = 44518453

    investor_name = get_investor_name(db, investor_id)
    investor_type = get_investor_type(db, investor_id)

    concept_name = get_concept_name(db, concept_id)

    bc.log(f'Source investor is {investor_id} ({investor_name}, {investor_type})')
    bc.log(f'Target concept is {concept_id} ({concept_name})')

    # Time window to be checked
    min_date = '2021-01-01'
    max_date = '2022-01-01'

    bc.log(f'Time window is [{min_date}, {max_date})')

    ############################################################
    # PRELIMINARIES                                            #
    ############################################################

    # Get a list of funding rounds in time window
    table_name = 'graph.Nodes_N_FundingRound'
    fields = ['FundingRoundID']
    conditions = {'FundingRoundDate': {'>': min_date, '<': max_date}}
    fr_ids = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)['FundingRoundID'].tolist()

    ############################################################
    # INVESTOR INVESTORS                                       #
    ############################################################

    bc.log(f'#### INVESTOR INVESTORS ####')
    bc.indent()

    investor_fr_ids = get_investor_fr_ids(db, fr_ids, investor_id)
    bc.log(f'Source investor has {len(investor_fr_ids)} funding rounds in time window')

    investor_investor_ids = get_investor_investor_ids(db, fr_ids, investor_id)
    bc.log(f'Source investor has {len(investor_investor_ids)} investors in time period')

    ############################################################

    if len(investor_investor_ids) < 50:
        bc.indent()

        for investor_investor_id in investor_investor_ids:
            investor_investor_name = get_investor_name(db, investor_investor_id)
            bc.log(f'Investor investor {investor_investor_id} ({investor_investor_name})')

            bc.indent()
            investor_investor_fr_ids = get_investor_fr_ids(db, fr_ids, investor_investor_id)
            joint_fr_ids = list(set(investor_fr_ids) & set(investor_investor_fr_ids))
            bc.log(f'Invested in {len(investor_investor_fr_ids)} funding rounds in time window. Joint funding rounds {joint_fr_ids}')
            bc.outdent()

        bc.outdent()

    ############################################################

    bc.outdent()

    ############################################################
    # INVESTOR CONCEPTS                                        #
    ############################################################

    bc.log(f'#### INVESTOR CONCEPTS ####')
    bc.indent()

    investor_concept_ids = get_investor_concept_ids(db, fr_ids, investor_id)
    bc.log(f'Source investor has {len(investor_concept_ids)} concepts in time period')

    ############################################################

    bc.indent()

    for investor_concept_id in investor_concept_ids:
        investor_concept_name = get_concept_name(db, investor_concept_id)
        bc.log(f'Concept {investor_concept_id} ({investor_concept_name})')

    bc.outdent()

    ############################################################

    bc.outdent()

    ############################################################
    # CONCEPT INVESTORS                                        #
    ############################################################

    bc.log(f'#### CONCEPT INVESTORS ####')
    bc.indent()

    concept_investor_ids = get_concept_investor_ids(db, fr_ids, concept_id)
    bc.log(f'Target concept has {len(concept_investor_ids)} investors in time period')

    ############################################################

    if len(concept_investor_ids) < 50:
        bc.indent()

        for concept_investor_id in concept_investor_ids:
            concept_investor_name = get_investor_name(db, concept_investor_id)
            bc.log(f'Concept investor {concept_investor_id} ({concept_investor_name})')

            bc.indent()
            concept_investor_fr_ids = get_investor_fr_ids(db, fr_ids, concept_investor_id)
            bc.log(f'Invested in {len(concept_investor_fr_ids)} funding rounds in time window.')
            bc.outdent()

        bc.outdent()

    ############################################################

    bc.outdent()

    ############################################################
    # CONCEPT CONCEPTS                                         #
    ############################################################

    bc.log(f'#### CONCEPT CONCEPTS ####')
    bc.indent()

    ############################################################

    concept_concept_ids = get_concept_concept_ids(db, concept_id)
    bc.log(f'Target concept has {len(concept_concept_ids)} concepts')

    ############################################################

    bc.indent()

    for concept_concept_id in concept_concept_ids:
        concept_concept_name = get_concept_name(db, concept_concept_id)
        bc.log(f'Target concept concept {concept_concept_id} ({concept_concept_name})')

    bc.outdent()

    ############################################################

    bc.outdent()

    ############################################################
    # INTERSECTIONS                                            #
    ############################################################

    bc.log(f'#### INTERSECTION INVESTORS ####')
    bc.indent()

    ############################################################

    intersection_investor_ids = list(set(investor_investor_ids) & set(concept_investor_ids))
    bc.log(f'Source investor and target concept have {len(intersection_investor_ids)} common investors in time period')

    ############################################################

    if len(intersection_investor_ids) < 50:
        bc.indent()

        for intersection_investor_id in intersection_investor_ids:
            intersection_investor_name = get_investor_name(db, intersection_investor_id)
            bc.log(f'Intersection investor {intersection_investor_id} ({intersection_investor_name})')

            bc.indent()
            intersection_investor_fr_ids = get_investor_fr_ids(db, fr_ids, intersection_investor_id)
            bc.log(f'Invested in {len(intersection_investor_fr_ids)} funding rounds in time window.')
            bc.outdent()

        bc.outdent()

    ############################################################

    bc.outdent()

    ############################################################

    bc.log(f'#### INTERSECTION CONCEPTS ####')
    bc.indent()

    ############################################################

    intersection_concept_ids = [investor_concept_id for investor_concept_id in investor_concept_ids if investor_concept_id in concept_concept_ids]
    bc.log(f'Source investor and target concept have {len(intersection_concept_ids)} common concepts in time period')

    ############################################################

    if len(intersection_concept_ids) < 50:
        bc.indent()

        for intersection_concept_id in intersection_concept_ids:
            intersection_concept_name = get_concept_name(db, intersection_concept_id)
            bc.log(f'Intersection concept {intersection_concept_id} ({intersection_concept_name})')

        bc.outdent()

    ############################################################

    bc.outdent()

    ############################################################

    bc.report()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 400)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    main()
