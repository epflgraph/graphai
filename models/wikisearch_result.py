class WikisearchResult:
    """
    Class representing the result of a wikisearch for a given set of keywords.
    """

    def __init__(self, keywords, pages):
        self.keywords = keywords
        self.pages = pages

    def __repr__(self):
        return f'WikisearchResult({self.keywords}, {len(self.pages)})'
