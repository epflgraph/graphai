from requests import post


class OldApi:
    def __init__(self):
        self.url = 'http://cedegemac34.epfl.ch:38800/Wikify'

        self.snake_map = {
            'RawText': 'raw_text',
            'AnchorPageIDs': 'anchor_page_ids',
            'Keywords': 'keywords',
            'PageID': 'page_id',
            'PageTitle': 'page_title',
            'SearchRank': 'searchrank',
            'MedianGraphScore': 'median_graph_score',
            'GraphRankRatio': 'searchrank_graph_ratio',
            'LevenshteinScore': 'levenshtein_score',
            'MixedScore': 'mixed_score',
            'SourcePageIDs': 'source_page_ids',
            'TargetPageIDs': 'target_page_ids'
        }
        self.unsnake_map = {v: k for k, v in self.snake_map.items()}

    def snake(self, d):
        snaked = {}
        for key in d:
            snaked[self.snake_map.get(key, key)] = d[key]
        return snaked

    def unsnake(self, d):
        unsnaked = {}
        for key in d:
            unsnaked[self.unsnake_map.get(key, key)] = d[key]
        return unsnaked

    def wikify(self, raw_text, anchor_page_ids):
        params = {
            'raw_text': raw_text,
            'anchor_page_ids': anchor_page_ids
        }

        return [self.snake(d) for d in post(self.url, json=self.unsnake(params)).json()]
