from requests import post


class NewApi:
    def __init__(self):
        self.url = 'http://86.119.27.90:28800'
        self.name = '/wikify'

    def keywords(self, raw_text):
        params = {
            'raw_text': raw_text
        }

        return post(f'{self.url}/keywords', json=params).json()

    def keywords_nltk(self, raw_text):
        params = {
            'raw_text': raw_text
        }

        return post(f'{self.url}/keywords_nltk', json=params).json()

    def wikify(self, raw_text, anchor_page_ids=None):
        if anchor_page_ids is None:
            params = {
                'raw_text': raw_text
            }
        else:
            params = {
                'raw_text': raw_text,
                'anchor_page_ids': anchor_page_ids
            }

        return post(f'{self.url}/wikify', json=params).json()

    def wikify_keywords(self, keyword_list, anchor_page_ids=None):
        if anchor_page_ids is None:
            params = {
                'keyword_list': keyword_list
            }
        else:
            params = {
                'keyword_list': keyword_list,
                'anchor_page_ids': anchor_page_ids
            }

        return post(f'{self.url}/wikify_keywords', json=params).json()
