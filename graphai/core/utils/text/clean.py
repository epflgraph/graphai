import re
import unicodedata
from html.parser import HTMLParser

import cleantext as ct

text_tags = ['b', 'strong', 'i', 'em', 'mark', 'small', 'del', 'ins', 'sub', 'sup']

symbols = {
    'squote': "'",
    'dquote': '"',
    'intermark': '?',
    'exclmark': '!',
    'dot': '.',
    'comma': ',',
    'colon': ':',
    'semicolon': ';',
    'lpar': '(',
    'rpar': ')',
    'lbrace': '{',
    'lcbracket': '{',
    'rbrace': '}',
    'rcbracket': '}',
    'lbracket': '[',
    'lsbracket': '[',
    'rbracket': ']',
    'rsbracket': ']',
    'backslash': '\\',
    'tab': '\t',
    'linebreak': '\n',
}


class HTMLCleaner(HTMLParser):
    """
    Class to parse and clean HTML tags from raw text.
    """

    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.pieces = []

    def handle_starttag(self, tag, attrs):
        if tag in symbols:
            self.pieces.append(symbols[tag])
            return

        if tag in text_tags:
            self.pieces.append(' ')
            return

        self.pieces.append('\n')

    def handle_endtag(self, tag):
        if tag in symbols:
            return

        if tag in text_tags:
            self.pieces.append(' ')
            return

        self.pieces.append('\n')

    def handle_data(self, d):
        # Clean manual unordered lists with leading *, - or ·
        lines = d.split('\n')
        lines = [re.sub(r'^\s*[\*+|\-+|•+]\s*', '', line) for line in lines]
        d = '\n'.join(lines)

        self.pieces.append(d)

    def get_data(self):
        # Concatenate all pieces of text
        s = ''.join(self.pieces)

        # Normalize text
        unicodedata.normalize('NFKC', s)

        # Remove punctuation
        for p in [':', ';', '|', '/', '=', '“', '”', '«', '»', '§']:
            s = s.replace(p, ' ')

        # Collapse multiple whitespaces
        s = re.sub(' {2,}', ' ', s)
        s = re.sub('\t{2,}', ' ', s)
        s = re.sub('\n{2,}', '\n', s)
        s = re.sub('\r{2,}', '\n', s)
        s = re.sub(' \n', '\n', s)
        s = re.sub('\n ', '\n', s)

        return s


def normalize(text):
    """
    Normalizes the given text by solving encoding problems, deleting URLs, emails, cleaning HTML tags and
    converting to lowercase.

    Args:
        text (str): Text to be normalized.

    Returns:
        str: Normalized text.
    """

    # Clean text of encoding problems and other rubbish
    text = ct.clean(text, lower=False, to_ascii=False, no_line_breaks=False, no_urls=True, replace_with_url='', no_emails=True, replace_with_email='')

    # Remove patterns known to create issues with HTMLCleaner
    text = text.replace('<![', '[')

    ################################################################
    # HTML cleaning                                                #
    ################################################################

    # Instantiate parser
    c = HTMLCleaner()

    # Feed text and process it
    c.feed(text)

    # Close parser (forces processing of unfinished text and prevents issue when text doesn't have '<' but has a '&' near the end)
    c.close()

    # Retrieve clean text from parser
    clean_text = c.get_data()

    # Replace text only if it is not empty after HTML stripping (it happens in some circumstances when the text is poor HTML)
    if clean_text:
        text = clean_text

    # Return lowercased and stripped text
    return text.lower().strip()


if __name__ == '__main__':
    text = r"""Finally, R&D"""
    text = normalize(text)

    print(text)
