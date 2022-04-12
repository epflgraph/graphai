import json
import unicodedata
import logging
import re
from html.parser import HTMLParser

emoji_pattern = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\uD83E[\uDD00-\uDDFF])|"
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83c[\udde0-\uddff])|"  # flags (iOS)
    u"([\u2934\u2935]\uFE0F?)|"
    u"([\u3030\u303D]\uFE0F?)|"
    u"([\u3297\u3299]\uFE0F?)|"
    u"([\u203C\u2049]\uFE0F?)|"
    u"([\u00A9\u00AE]\uFE0F?)|"
    u"([\u2122\u2139]\uFE0F?)|"
    u"(\uD83C\uDC04\uFE0F?)|"
    u"(\uD83C\uDCCF\uFE0F?)|"
    u"([\u0023\u002A\u0030-\u0039]\uFE0F?\u20E3)|"
    u"(\u24C2\uFE0F?|[\u2B05-\u2B07\u2B1B\u2B1C\u2B50\u2B55]\uFE0F?)|"
    u"([\u2600-\u26FF]\uFE0F?)|"
    u"([\u2700-\u27BF]\uFE0F?)"
    "+", flags=re.UNICODE
)


def remove_emojis(text):
    text = text.decode('utf8')
    return emoji_pattern.sub(r'', text).encode('utf8')


safe_type = {
    "'": '<squote/>',
    '"': '<dquote/>',
    '?': '<intermark/>',
    '!': '<exclmark/>',
    '.': '<dot/>',
    ',': '<comma/>',
    ':': '<colon/>',
    ';': '<semicolon/>',
    '(': '<lpar/>',
    ')': '<rpar/>',
    '{': '<lbrace/>',
    '}': '<rbrace/>',
    '[': '<lbracket/>',
    ']': '<rbracket/>',
    '\\': '<backslash/>',
    '\t': '<tab/>',
    '\n': '<linebreak/>',
    '\r': '<linebreak/>',
}

unsafe_type = {
    '<squote/>': "'",
    '<dquote/>': '"',
    '<intermark/>': '?',
    '<exclmark/>': '!',
    '<dot/>': '.',
    '<comma/>': ',',
    '<colon/>': ':',
    '<semicolon/>': ';',
    '<lpar/>': '(',
    '<rpar/>': ')',
    '<lbrace/>': '{',
    '<lcbracket/>': '{',
    '<rbrace/>': '}',
    '<rcbracket/>': '}',
    '<lbracket/>': '[',
    '<lsbracket/>': '[',
    '<rbracket/>': ']',
    '<rsbracket/>': ']',
    '<backslash/>': '\\',
    '<tab/>': '\t',
    '<linebreak/>': '\n',
}


def printjson(s):
    print(json.dumps(s, sort_keys=True, indent=4, separators=(',', ': ')))


def is_safetype(text):
    for symbol in safe_type.keys():
        if symbol in text:
            return False
    return True


def xmlify(text, exceptions=[]):
    if type(text) is str or type(text) is unicode:
        for symbol in safe_type.keys():
            if symbol not in exceptions:
                text = text.replace(symbol, safe_type[symbol])
    return text


def unxmlify(text, exceptions=[]):
    if type(text) is str or type(text) is unicode:
        for escaped_symbol in unsafe_type.keys():
            if unsafe_type[escaped_symbol] not in exceptions:
                text = text.replace(escaped_symbol, unsafe_type[escaped_symbol])
    return text


def force_decode(text, codecs=['utf8', 'cp1252']):
    for i in codecs:
        try:
            if type(text) is unicode:
                return text
            elif type(text) is float or type(text) is int:
                return unicode(text)
            else:
                return text.decode(i)
        except UnicodeDecodeError:
            pass
    logging.warning("Cannot decode %s" % ([text]))


def unicode_to_ascii(text):
    return unicodedata.normalize('NFKD', force_decode(text)).encode('ascii', 'ignore')

    # if type(text) is unicode:
    #     return unicodedata.normalize('NFKD', text).encode('ascii','ignore')
    # else:
    #     print text.decode("utf8", "replace")
    #     return str(text)
    #     # return unicodedata.normalize('NFKD', unicode(text).encode('utf-8') ).encode('ascii','ignore')


def remove_special_chars(text, exceptions=[]):
    if type(text) is str or type(text) is unicode:
        for symbol in safe_type.keys():
            if symbol not in exceptions:
                text = text.replace(symbol, '')
    return text


def parse_to_sql_column_name(text, separator=''):
    if type(text) is str or type(text) is unicode:
        text_list = [text]
    elif type(text) is list:
        text_list = text
    else:
        return text

    parsed_text_list = []
    for i in text_list:
        if i == 'TableName':
            parsed_text_list += [i]
        else:
            i = unxmlify(i)
            i = remove_special_chars(i)
            i = unicode_to_ascii(i)
            i = i.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
            i = i.replace('/', ' ')
            parsed_text_list += [separator.join([s.title() for s in i.split(' ')])[0:64]]

    if type(text) is str or type(text) is unicode:
        return parsed_text_list[0]
    elif type(text) is list:
        return parsed_text_list


def remove_double_spaces(s):
    s_prev = s
    for k in range(100):
        s = s.replace('  ', ' ')
        if s == s_prev:
            return s.strip()
        else:
            s_prev = s


# Class that removes XML tags
class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        s = '. '.join(self.fed)
        s = s.replace('\n', '. ').replace('\t', ' ').replace('\xa0', ' ').replace('\xad', '').replace('\u2009', ' ')
        s = s.replace('.', ' . ')
        while s.find('  ') != -1:
            s = s.replace('  ', ' ')
        while s.find('. .') != -1:
            s = s.replace('. .', '.')
        while s.find('..') != -1:
            s = s.replace('..', '.')
        s = s.replace(' . ', '. ')
        return s


# Function that removes XML tags
def strip_xml_tags(text):
    if text is None:
        return None
    mls = MLStripper()
    mls.feed(text)
    return mls.get_data()




