import re
import unicodedata

from html.parser import HTMLParser

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
