from html.parser import HTMLParser

text_tags = ['b', 'strong', 'i', 'em', 'mark', 'small', 'del', 'ins', 'sub', 'sup']


class HTMLCleaner(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.pieces = []

    def handle_starttag(self, tag, attrs):
        if tag in text_tags:
            self.pieces.append(' ')
        else:
            self.pieces.append('\n')

    def handle_endtag(self, tag):
        if tag in text_tags:
            self.pieces.append(' ')
        else:
            self.pieces.append('\n')

    def handle_data(self, d):
        d = d.replace(u'\xa0', ' ')

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

            self.pieces.append(line)

    def get_data(self):
        # Concatenate all pieces of text
        s = ''.join(self.pieces)

        # Add spaces around punctuation
        for p in ['.', ',', ';', '|', '/', '\n-']:
            s = s.replace(p, ' %s ' % p)

        # Remove freestyle list markers
        # s.replace()

        # Collapse multiple spaces
        while '  ' in s:
            s = s.replace('  ', ' ')

        # Collapse multiple tabs and line breaks
        while '\t\t' in s:
            s = s.replace('\t\t', '\t')
        while '\n\n' in s:
            s = s.replace('\n\n', '\n')
        while '\r\r' in s:
            s = s.replace('\r\r', '\r')

        # Remove single quotes
        s = s.replace("'", '')

        return s
