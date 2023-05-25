"""
Create json file with results of new concept detection algorithm.
"""

import requests

import json
import json.decoder

import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Set of slides for experiment. Result of the following query:
#     SELECT SlideID, SlideText
#     FROM gen_switchtube.Slide_Text
#     WHERE SlideText IS NOT NULL
#     AND MD5(SlideID) LIKE "%000"
slides = pd.read_json('slides.json')


def new_wikify(row):
    data = {'raw_text': row['SlideText']}

    try:
        result = requests.post('http://graphai.epfl.ch/text/wikify', json=data)

        # Avoid error in the case of Internal Server Error
        try:
            result = result.json()
        except json.decoder.JSONDecodeError:
            result = []

    except TypeError:
        print(data)
        result = []

    if not result:
        return pd.DataFrame()

    result = pd.DataFrame(result).rename(columns={'MixedScore': 'Score'})

    result = result[['PageID', 'PageTitle', 'Score']].sort_values(by='Score', ascending=False)

    return result


pages = pd.DataFrame()
for index, row in slides.iterrows():
    print('.', end='')
    wikified_row = new_wikify(row)

    if len(wikified_row) == 0:
        continue

    wikified_row['SlideID'] = row['SlideID']
    wikified_row['SlideText'] = row['SlideText']

    wikified_row = wikified_row[['SlideID', 'SlideText', 'PageID', 'PageTitle', 'Score']]

    pages = pd.concat([pages, wikified_row])

pages = pages.reset_index(drop=True)

with open('pages_new.json', 'w', encoding='utf-8') as file:
    json.dump(pages.to_dict(orient='records'), file, ensure_ascii=True)
