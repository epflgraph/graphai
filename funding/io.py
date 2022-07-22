from definitions import FUNDING_DIR
from utils.text.io import save_json


def save_scores(min_year, max_year, scores, name):
    model_dirname = f'{FUNDING_DIR}/models/{name}'

    # Save data
    data = {
        'min_year': min_year,
        'max_year': max_year,
        'scores': scores
    }
    save_json(data, f'{model_dirname}/scores.json')
