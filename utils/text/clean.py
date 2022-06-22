import cleantext as ct

from utils.text.html_cleaner import HTMLCleaner


def normalize(text):
    # Clean text of encoding problems and other rubbish
    text = ct.clean(text, lower=False, to_ascii=False, no_line_breaks=False, no_urls=True, replace_with_url='', no_emails=True, replace_with_email='')

    # Clean text of HTML code
    c = HTMLCleaner()
    c.feed(text)
    text = c.get_data()

    return text.lower().strip()
