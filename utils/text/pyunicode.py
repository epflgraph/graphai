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


def unxmlify(text, exceptions=[]):
    if type(text) is str:
        for escaped_symbol in unsafe_type.keys():
            if unsafe_type[escaped_symbol] not in exceptions:
                text = text.replace(escaped_symbol, unsafe_type[escaped_symbol])
    return text
