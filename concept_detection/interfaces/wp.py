import requests

from concept_detection.types.page_result import PageResult


class WP:
    def __init__(self):
        self.params = {
            'format': 'json',
            'action': 'query',
            'list': 'search',
            'srsearch': '',
            'srlimit': 10,
            'srprop': ''
        }
        self.headers = {'User-Agent': 'graphai (https://github.com/epflgraph/graphai)'}
        self.url = 'http://en.wikipedia.org/w/api.php'

    def minsearch(self, text):
        self.params['srsearch'] = text
        r = requests.get(self.url, params=self.params, headers=self.headers).json()

        top_pages = r['query']['search']

        return [
            PageResult(
                page_id=top_pages[i]['pageid'],
                page_title=top_pages[i]['title'],
                searchrank=(i + 1),
                score=(1 / (i + 1))
            )
            for i in range(len(top_pages))
        ]
