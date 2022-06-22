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
        for line in d.split('\n'):
            # Don't include manual unordered list markers: -
            split_line = line.split('-', 1)
            if len(split_line) >= 2 and (not split_line[0] or split_line[0].isspace()):
                self.pieces.append(split_line[1])
                continue

            # Don't include manual unordered list markers: •
            split_line = line.split('•', 1)
            if len(split_line) >= 2 and (not split_line[0] or split_line[0].isspace()):
                self.pieces.append(split_line[1])
                continue

            # Don't include manual unordered list markers: *
            split_line = line.split('*', 1)
            if len(split_line) >= 2 and (not split_line[0] or split_line[0].isspace()):
                self.pieces.append(split_line[1])
                continue

            self.pieces.append(line)

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
    # Clean text of encoding problems and other rubbish
    text = ct.clean(text, lower=False, to_ascii=False, no_line_breaks=False, no_urls=True, replace_with_url='', no_emails=True, replace_with_email='')

    # Clean text of HTML code
    c = HTMLCleaner()
    c.feed(text)
    text = c.get_data()

    return text.lower().strip()


def decode_url_title(url_title):
    """
    Decodes free text title from wikipedia URL title:
    * Convert to lowercase.
    * Replace punctuation with spaces.
    * Collapse multiple consecutive spaces and remove leading/trailing spaces.
    * Replace single quotes with dashes.

    Args:
        url_title (str): Title to decode.

    Returns:
        str: Decoded title.
    """

    # Convert to lowercase
    decoded_title = url_title.lower()

    # Replace punctuation with spaces
    for c in ['_', '-', '–', '"', ';']:
        decoded_title = decoded_title.replace(c, ' ')

    # Collapse multiple consecutive spaces
    while '  ' in decoded_title:
        decoded_title = decoded_title.replace('  ', ' ')

    # Remove leading and trailing spaces
    decoded_title = decoded_title.strip()

    # Replace single quotes with dashes
    decoded_title = decoded_title.replace("'", '-')

    return decoded_title
