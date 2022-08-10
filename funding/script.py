from funding.data_processing import create_feature_set, build_time_series, next_time_period, time_period_first_date
from funding.tuning import create_tuned_xgb_params
from funding.training import create_model
from funding.predicting import predict_concepts_time_period
from funding.visualization import plot_df

from funding.concepts import concept_ids

if __name__ == '__main__':
    min_date = '2018-01-01'
    max_date = '2022-01-01'

    features_name = f'{min_date}-{max_date}-all'
    xgb_params_name = 'tuned'

    # features_name = create_feature_set(min_date, max_date, debug=True)

    # xgb_params_name = create_tuned_xgb_params(features_name, debug=True)

    # create_model(features_name, xgb_params_name, debug=True)

    time_period = '2022-Q1'
    df, y_pred = predict_concepts_time_period(time_period, concept_ids, features_name, xgb_params_name, debug=True)

    # Build true data for given time period
    min_date = time_period_first_date(time_period)
    max_date = time_period_first_date(next_time_period(time_period))
    period_df = build_time_series(min_date=min_date, max_date=max_date, concept_ids=concept_ids, debug=False)
    period_df.index = period_df['concept_id']
    y_true = period_df['amount']

    print(df)
    print(y_pred)
    print(y_true)

    plot_df(df, time_period, y_pred, y_true)


