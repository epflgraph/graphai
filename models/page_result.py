class PageResult:
    """
    Class representing a wikipage in the context of a search result.
    """

    def __init__(self, page_id, page_title, searchrank, score):
        self.page_id = page_id
        self.page_title = page_title
        self.searchrank = searchrank
        self.score = score

    def __repr__(self):
        return f'PageResult({self.page_id}, {self.page_title}, {self.searchrank}. {self.score})'

    def equivalent(self, other):
        """
        Return whether self and other results have the same wikipage id.
        """

        return self.page_id == other.page_id

    def equal(self, other):
        """
        Return whether self and other results have the same wikipage id and score.
        """

        return self.equivalent(other) and self.score == other.score

    def to_tuple(self):
        """
        Convert result to tuple.
        """

        return self.page_id, self.page_title, self.searchrank, self.score
