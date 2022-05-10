class PageResult:

    def __init__(self, page_id, page_title, searchrank, score):
        self.page_id = page_id
        self.page_title = page_title
        self.searchrank = searchrank
        self.score = score

    def __repr__(self):
        return f'PageResult({self.page_id}, {self.page_title}, {self.searchrank}. {self.score})'
