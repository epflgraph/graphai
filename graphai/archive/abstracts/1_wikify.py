import requests
import json

from graphai.core.utils.text.io import read_json, save_json

wikify_url = 'http://localhost:28800/text/wikify?refresh_scores=false'

prefix = 'no-rescore'

names = ['wave-fields', 'schreier', 'collider', 'skills']

for name in names:
    data = read_json(f'../../api/requests/0-simple/{name}.json')
    results = requests.post(wikify_url, data=json.dumps(data)).json()
    save_json(results, f'{prefix}-{name}.json')
