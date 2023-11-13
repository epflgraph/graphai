import numpy as np
import re

# Constants for ASCII codes
codeBackslash = 0x5c  # "\"
codeSlash = 0x2f  # "/"
codeAsterisk = 0x2a  # "*"
codeOpeningBrace = 0x7b  # "{"
codeClosingBrace = 0x7d  # "}"
codeOpeningBracket = 0x5b  # "["
codeClosingBracket = 0x5d  # "]"
codeOpenParenthesis = 0x28  # "("
codeCloseParenthesis = 0x29  # ")"
codeSpace = 0x20  # " "
codeNewline = 0xa  # "\n"
codeTab = 0x9  # "\t"
codeReturn = 0xd  # "\r"
codeBackspace = 0x08  # "\b"
codeFormFeed = 0x0c  # "\f"
codeDoubleQuote = 0x0022  # """
codePlus = 0x2b  # "+"
codeMinus = 0x2d  # "-"
codeQuote = 0x27  # "'"
codeZero = 0x30
codeNine = 0x39
codeComma = 0x2c  # ","
codeDot = 0x2e  # "."
codeColon = 0x3a  # ":"
codeSemicolon = 0x3b  # ";"
codeUppercaseA = 0x41  # "A"
codeLowercaseA = 0x61  # "a"
codeUppercaseE = 0x45  # "E"
codeLowercaseE = 0x65  # "e"
codeUppercaseF = 0x46  # "F"
codeLowercaseF = 0x66  # "f"
codeNonBreakingSpace = 0xa0
codeEnQuad = 0x2000
codeHairSpace = 0x200a
codeNarrowNoBreakSpace = 0x202f
codeMediumMathematicalSpace = 0x205f
codeIdeographicSpace = 0x3000
codeDoubleQuoteLeft = 0x201c  # “
codeDoubleQuoteRight = 0x201d  # ”
codeQuoteLeft = 0x2018  # ‘
codeQuoteRight = 0x2019  # ’
codeGraveAccent = 0x0060  # `
codeAcuteAccent = 0x00b4  # ´


def is_hex(code):
    return (codeZero <= code <= codeNine) or \
           (codeUppercaseA <= code <= codeUppercaseF) or \
           (codeLowercaseA <= code <= codeLowercaseF)


def is_digit(code):
    return codeZero <= code <= codeNine


def is_valid_string_character(code):
    return 0x20 <= code <= 0x10ffff


regex_delimiter = re.compile(r'[\[\],:{}()\n+]')


def is_delimiter(char):
    return regex_delimiter.match(char) or (char and is_quote(ord(char)))


regex_start_of_value = re.compile(r'[\[{\w-]')


def is_start_of_value(char):
    if char is None:
        return False
    return regex_start_of_value.match(char) or (char and is_quote(ord(char)))


def is_control_character(code):
    return code in [codeNewline, codeReturn, codeTab, codeBackspace, codeFormFeed]


def is_whitespace(code):
    return code in [codeSpace, codeNewline, codeTab, codeReturn]


def is_special_whitespace(code):
    return (code == codeNonBreakingSpace or (codeEnQuad <= code <= codeHairSpace)
            or code == codeNarrowNoBreakSpace or code == codeMediumMathematicalSpace or code == codeIdeographicSpace)


def is_quote(code):
    return is_double_quote_like(code) or is_single_quote_like(code)


def is_double_quote_like(code):
    return code in [codeDoubleQuote, codeDoubleQuoteLeft, codeDoubleQuoteRight]


def is_double_quote(code):
    return code == codeDoubleQuote


def is_single_quote_like(code):
    return code in [codeQuote, codeQuoteLeft, codeQuoteRight, codeGraveAccent, codeAcuteAccent]


def is_single_quote(code):
    return code == codeQuote


def strip_last_occurrence(text, text_to_strip, strip_remaining_text=False):
    index = text.rfind(text_to_strip)
    return text[:index] + ('' if strip_remaining_text else text[index + 1:]) if index != -1 else text


def insert_before_last_whitespace(text, text_to_insert):
    index = len(text)
    while is_whitespace(ord(text[index - 1])):
        index -= 1
    return text[:index] + text_to_insert + text[index:]


def remove_at_index(text, start, count):
    return text[:start] + text[start + count:]


regex_ends_with_comma_or_newline = re.compile(r'[,\n][ \t\r]*')


def ends_with_comma_or_newline(text):
    return regex_ends_with_comma_or_newline.search(text) is not None


def next_non_white_space_character(text, start):
    i = start
    while is_whitespace(ord(text[i])):
        i += 1
    return text[i]


def char_code_at(text, ind):
    try:
        return ord(text[ind])
    except Exception:
        return np.NaN


def char_at(text, ind, default=''):
    try:
        return text[ind]
    except Exception:
        return default
