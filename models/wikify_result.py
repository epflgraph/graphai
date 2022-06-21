class WikifyResult:
    def __init__(self, keywords, page_id, page_title, page_title_0='', searchrank=0, median_graph_score=0, search_graph_ratio=0, levenshtein_score=0, mixed_score=0):
        self.keywords = keywords
        self.page_id = page_id
        self.page_title = page_title
        self.page_title_0 = page_title_0
        self.searchrank = searchrank
        self.median_graph_score = median_graph_score
        self.search_graph_ratio = search_graph_ratio
        self.levenshtein_score = levenshtein_score
        self.mixed_score = mixed_score

    def __repr__(self):
        return f'WikifyResult({self.keywords}, {self.page_id}, {self.page_title}, {self.median_graph_score:.4f}, {self.mixed_score:.4f})'

    def equivalent(self, other):
        return self.keywords == other.keywords and self.page_id == other.page_id

    def equal(self, other):
        return self.equivalent(other) and abs(self.median_graph_score - other.median_graph_score) < 10e-4

    @staticmethod
    def from_dict(d):
        return WikifyResult(
            keywords=d.get('keywords'),
            page_id=d.get('page_id'),
            page_title=d.get('page_title'),
            page_title_0=d.get('page_title_0'),
            searchrank=d.get('searchrank'),
            median_graph_score=d.get('median_graph_score'),
            search_graph_ratio=d.get('search_graph_ratio'),
            levenshtein_score=d.get('levenshtein_score'),
            mixed_score=d.get('mixed_score')
        )
