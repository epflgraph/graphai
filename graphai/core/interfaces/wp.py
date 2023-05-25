import requests
import pandas as pd


class WP:
    """
    Base class to communicate with the Wikipedia API.
    """

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

    def search(self, text, limit=10):
        """
        Perform search query to Wikipedia API.

        Args:
            text (str): Query text for the search.
            limit (int): Maximum number of returned results.

        Returns:
            pd.DataFrame: A pandas DataFrame with columns ['PageID', 'PageTitle', 'Searchrank', 'SearchScore'],
            unique in 'PageID', with the wikisearch results the given keywords set.
        """

        # Make request
        try:
            self.params['srsearch'] = text
            self.params['srlimit'] = limit
            r = requests.get(self.url, params=self.params, headers=self.headers, timeout=6).json()

            top_pages = r['query']['search']

            table = [[top_pages[i]['pageid'], top_pages[i]['title'], (i + 1), 1] for i in range(len(top_pages))]

            return pd.DataFrame(table, columns=['PageID', 'PageTitle', 'Searchrank', 'SearchScore'])
        except Exception:
            # If something goes wrong, avoid crashing and return empty DataFrame
            return pd.DataFrame(columns=['PageID', 'PageTitle', 'Searchrank', 'SearchScore'])
