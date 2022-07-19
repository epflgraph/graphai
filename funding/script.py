from funding.training import create_model
from funding.validation import evaluate_model
from funding.predicting import predict_concepts_year

from funding.concepts import concept_ids

from funding.visualization import plot_df

if __name__ == '__main__':
    min_year = 2018
    max_year = 2021

    # create_model(min_year, max_year, name=f'simple_{min_year}_{max_year}', debug=True)

    # evaluate_model(min_year, max_year, name=f'simple_{min_year}_{max_year}', debug=True)

    year = 2022
    df = predict_concepts_year(year, concept_ids, name=f'simple_{min_year}_{max_year}', debug=True)
    plot_df(df)

