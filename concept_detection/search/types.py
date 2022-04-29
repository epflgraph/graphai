class PageResult:

    def __init__(self, page_id, page_title, searchrank, score):
        self.page_id = page_id
        self.page_title = page_title
        self.searchrank = searchrank
        self.score = score


class WikisearchResult:

    def __init__(self, keywords, pages):
        self.keywords = keywords
        self.pages = pages