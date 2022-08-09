from funding.data_processing import create_feature_set
from funding.tuning import create_tuned_xgb_params
from funding.training import create_model
from funding.predicting import predict_concepts_time_period

from funding.concepts import concept_ids

from funding.visualization import plot_df

if __name__ == '__main__':
    min_date = '2018-01-01'
    max_date = '2022-01-01'

    features_name = create_feature_set(min_date, max_date, debug=True)

    print('djhjfhsjfhjdhfdj')

    # features_name = '2018-01-01-2022-01-01-all'
    xgb_params_name = create_tuned_xgb_params(features_name, debug=True)

    # create_model(features_name, xgb_params_name, debug=True)
    #
    # y = predict_concepts_year(2021, concept_ids, features_name, xgb_params_name, debug=True)

    # year = 2022
    # df = predict_concepts_year(year, concept_ids, name=f'simple_{min_year}_{max_year}', debug=True)
    # plot_df(df)


