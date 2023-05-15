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

    Examples:
        Use as follows:

        >>> text = ' '.join([
        ...     "<p>You get a <i>shiver</i> in the <strong>dark</strong>",
        ...     "<br/>",
        ...     "It's a raining in the <a>park</a> but meantime</p>"
        ... ])
        >>> c = HTMLCleaner()
        >>> c.feed(text)
        >>> print(c.get_data())
        <BLANKLINE>
        You get a shiver in the dark
        It's a raining in the
        park
        but meantime
        <BLANKLINE>
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

    Examples:
        >>> text = ' '.join([
        ...     "<p>You get a <i>shiver</i> in the <strong>dark</strong>",
        ...     "<br/>",
        ...     "It's a \\u2018raining\\u2019 in the <a>park</a> but »meantime«</p>",
        ...     "&lt;3"
        ... ])
        >>> print(normalize(text))
        you get a shiver in the dark
        it's a 'raining' in the
        park
        but meantime
        <3
    """

    # Clean text of encoding problems and other rubbish
    text = ct.clean(text, lower=False, to_ascii=False, no_line_breaks=False, no_urls=True, replace_with_url='', no_emails=True, replace_with_email='')

    # Clean text of HTML code
    c = HTMLCleaner()
    c.feed(text)
    text = c.get_data()

    return text.lower().strip()
