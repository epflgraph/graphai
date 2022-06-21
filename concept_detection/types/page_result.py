class PageResult:

    def __init__(self, page_id, page_title, searchrank, score):
        self.page_id = page_id
        self.page_title = page_title
        self.searchrank = searchrank
        self.score = score

    def __repr__(self):
        return f'PageResult({self.page_id}, {self.page_title}, {self.searchrank}. {self.score})'

    def equivalent(self, other):
        return self.page_id == other.page_id

    def equal(self, other):
        return self.equivalent(other) and self.score == other.score

    def to_tuple(self):
        return self.page_id, self.page_title, self.searchrank, self.score
