from graphai.core.common.json_repair.json_repair_error import JSONRepairError
from graphai.core.common.json_repair.string_utils import (
    codeAsterisk,
    codeBackslash,
    codeCloseParenthesis,
    codeClosingBrace,
    codeClosingBracket,
    codeColon,
    codeComma,
    codeDot,
    codeDoubleQuote,
    codeLowercaseE,
    codeMinus,
    codeNewline,
    codeOpeningBrace,
    codeOpeningBracket,
    codeOpenParenthesis,
    codePlus,
    codeSemicolon,
    codeSlash,
    codeUppercaseE,
    ends_with_comma_or_newline,
    insert_before_last_whitespace,
    is_control_character,
    is_delimiter,
    is_digit,
    is_double_quote,
    is_double_quote_like,
    is_hex,
    is_quote,
    is_single_quote,
    is_single_quote_like,
    is_special_whitespace,
    is_start_of_value,
    is_valid_string_character,
    is_whitespace,
    next_non_white_space_character,
    remove_at_index,
    strip_last_occurrence,
    char_code_at,
    char_at
)

import re
import json

control_characters = {
    '\b': '\\b',
    '\f': '\\f',
    '\n': '\\n',
    '\r': '\\r',
    '\t': '\\t'
}

# map with all escape characters
escape_characters = {
    '"': '"',
    '\\': '\\',
    '/': '/',
    'b': '\b',
    'f': '\f',
    'n': '\n',
    'r': '\r',
    't': '\t'
    # note that \u is handled separately in parse_string()
}


def repair_json(text):
    i = 0  # current index in text
    output = ''  # generated output

    def parse_value():
        parse_whitespace_and_skip_comments()
        processed = (
                parse_object() or
                parse_array() or
                parse_string() or
                parse_number() or
                parse_keywords() or
                parse_unquoted_string()
        )
        parse_whitespace_and_skip_comments()
        return processed

    def parse_whitespace_and_skip_comments():
        nonlocal i
        start = i
        changed = parse_whitespace()
        while changed:
            changed = parse_comment()
            if changed:
                changed = parse_whitespace()

        return i > start

    def parse_whitespace():
        nonlocal i, output
        print(i, output)
        whitespace = ''
        while is_whitespace(char_code_at(text, i)) or is_special_whitespace(char_code_at(text, i)):
            if is_whitespace(char_code_at(text, i)):
                whitespace += text[i]
            else:
                # repair special whitespace
                whitespace += ' '
            i += 1

        if whitespace:
            output += whitespace
            return True

        return False

    def parse_comment():
        nonlocal i
        # find a block comment '/* ... */'
        if char_code_at(text, i) == codeSlash and char_code_at(text, i + 1) == codeAsterisk:
            # repair block comment by skipping it
            while i < len(text) and not at_end_of_block_comment(text, i):
                i += 1
            i += 2
            return True

        # find a line comment '// ...'
        if char_code_at(text, i) == codeSlash and char_code_at(text, i + 1) == codeSlash:
            # repair line comment by skipping it
            while i < len(text) and char_code_at(text, i) != codeNewline:
                i += 1
            return True

        return False

    def parse_character(code):
        nonlocal i, output
        if char_code_at(text, i) == code:
            output += text[i]
            i += 1
            return True

        return False

    def skip_character(code):
        nonlocal i
        if char_code_at(text, i) == code:
            i += 1
            return True, i

        return False, i

    def skip_escape_character():
        return skip_character(codeBackslash)

    def parse_object():
        nonlocal i, output
        if char_code_at(text, i) == codeOpeningBrace:
            output += '{'
            i += 1
            parse_whitespace_and_skip_comments()

            initial = True
            while i < len(text) and char_code_at(text, i) != codeClosingBrace:
                if not initial:
                    processed_comma = parse_character(codeComma)
                    if not processed_comma:
                        # repair missing comma
                        output = insert_before_last_whitespace(output, ',')
                    parse_whitespace_and_skip_comments()
                else:
                    processed_comma = True
                    initial = False

                processed_key = parse_string() or parse_unquoted_string()
                if not processed_key:
                    if (
                            char_code_at(text, i) == codeClosingBrace or
                            char_code_at(text, i) == codeOpeningBrace or
                            char_code_at(text, i) == codeClosingBracket or
                            char_code_at(text, i) == codeOpeningBracket or
                            char_at(text, i, None) is None
                    ):
                        # repair trailing comma
                        output = strip_last_occurrence(output, ',')
                    else:
                        throw_object_key_expected()
                    break

                parse_whitespace_and_skip_comments()
                processed_colon = parse_character(codeColon)
                truncated_text = i >= len(text)
                if not processed_colon:
                    if is_start_of_value(char_at(text, i, None)) or truncated_text:
                        # repair missing colon
                        output = insert_before_last_whitespace(output, ':')
                    else:
                        throw_colon_expected()
                processed_value = parse_value()
                if not processed_value:
                    if processed_colon or truncated_text:
                        # repair missing object value
                        output += 'null'
                    else:
                        throw_colon_expected()

            if char_code_at(text, i) == codeClosingBrace:
                output += '}'
                i += 1
            else:
                # repair missing end bracket
                output = insert_before_last_whitespace(output, '}')

            return True

        return False

    def parse_array():
        nonlocal i, output
        if char_code_at(text, i) == codeOpeningBracket:
            output += '['
            i += 1
            parse_whitespace_and_skip_comments()

            initial = True
            while i < len(text) and char_code_at(text, i) != codeClosingBracket:
                if not initial:
                    processed_comma = parse_character(codeComma)
                    if not processed_comma:
                        # repair missing comma
                        output = insert_before_last_whitespace(output, ',')
                else:
                    initial = False

                processed_value = parse_value()
                if not processed_value:
                    # repair trailing comma
                    output = strip_last_occurrence(output, ',')
                    break

            if char_code_at(text, i) == codeClosingBracket:
                output += ']'
                i += 1
            else:
                # repair missing closing array bracket
                output = insert_before_last_whitespace(output, ']')

            return True

        return False

    def parse_newline_delimited_json():
        nonlocal output
        # repair NDJSON
        initial = True
        processed_value = True

        while processed_value:
            if not initial:
                # parse optional comma, insert when missing
                processed_comma = parse_character(codeComma)
                if not processed_comma:
                    # repair: add missing comma
                    output = insert_before_last_whitespace(output, ',')
            else:
                initial = False

            processed_value = parse_value()

        if not processed_value:
            # repair: remove trailing comma
            output = strip_last_occurrence(output, ',')

        # repair: wrap the output inside array brackets
        output = f"[\n{output}\n]"
        return output

    def parse_string(stop_at_delimiter=False):
        nonlocal i, output
        skip_escape_chars = char_code_at(text, i) == codeBackslash
        if skip_escape_chars:
            # repair: remove the first escape character
            i += 1
            skip_escape_chars = True

        if is_quote(char_code_at(text, i)):
            is_end_quote = (is_double_quote if is_double_quote(char_code_at(text, i))
                            else is_single_quote if is_single_quote(char_code_at(text, i))
            else is_single_quote_like if is_single_quote_like(char_code_at(text, i))
            else is_double_quote_like)

            i_before = i
            output_before = output

            output += '"'
            i += 1

            is_end_of_string = (lambda i: is_delimiter(char_at(text, i, None))) if stop_at_delimiter else \
                (lambda i: is_end_quote(char_code_at(text, i)))

            while i < len(text) and not is_end_of_string(i):
                if char_code_at(text, i) == codeBackslash:
                    char = char_at(text, i + 1)
                    escape_char = escape_characters.get(char)
                    if escape_char is not None:
                        output += text[i:i + 2]
                        i += 2
                    elif char == 'u':
                        if (is_hex(char_code_at(text, i + 2)) and is_hex(char_code_at(text, i + 3)) and
                                is_hex(char_code_at(text, i + 4)) and is_hex(char_code_at(text, i + 5))):
                            output += text[i:i + 6]
                            i += 6
                        else:
                            return throw_invalid_unicode_character(i)
                    else:
                        # repair invalid escape character: remove it
                        output += char
                        i += 2
                else:
                    char = char_at(text, i)
                    code = char_code_at(text, i)

                    if code == codeDoubleQuote and char_code_at(text, i - 1) != codeBackslash:
                        # repair unescaped double quote
                        output += '\\' + char
                        i += 1
                    elif is_control_character(code):
                        # unescaped control character
                        output += control_characters[char]
                        i += 1
                    else:
                        if not is_valid_string_character(code):
                            return throw_invalid_character(char)
                        output += char
                        i += 1

                if skip_escape_chars:
                    processed = skip_escape_character()
                    if processed:
                        pass

            try:
                has_end_quote = is_quote(char_code_at(text, i))
                valid = (has_end_quote and
                         (i + 1 >= len(text) or is_delimiter(next_non_white_space_character(text, i + 1))))
            except Exception:
                valid = False
                has_end_quote = False
            if not valid and not stop_at_delimiter:
                i = i_before
                output = output_before
                return parse_string(True)

            if has_end_quote:
                output += '"'
                i += 1
            else:
                # repair missing quote
                output = insert_before_last_whitespace(output, '"')

            parse_concatenated_string()
            return True

        return False

    def parse_concatenated_string():
        nonlocal i, output
        processed = False

        parse_whitespace_and_skip_comments()
        while char_code_at(text, i) == codePlus:
            processed = True
            i += 1
            parse_whitespace_and_skip_comments()

            # repair: remove the end quote of the first string
            output = strip_last_occurrence(output, '"', True)
            start = len(output)
            parse_string()

            # repair: remove the start quote of the second string
            output = remove_at_index(output, start, 1)

        return processed

    def parse_number():
        nonlocal i, output
        start = i
        if char_code_at(text, i) == codeMinus:
            i += 1
            result = expect_digit_or_repair(start)
            if result:
                return True

        # Parse integer part
        while is_digit(char_code_at(text, i)):
            i += 1

        # Parse fractional part
        if char_code_at(text, i) == codeDot:
            i += 1
            result = expect_digit_or_repair(start)
            if result:
                return True
            while is_digit(char_code_at(text, i)):
                i += 1

        # Parse exponent
        if char_code_at(text, i) in (codeLowercaseE, codeUppercaseE):
            i += 1
            if char_code_at(text, i) in (codeMinus, codePlus):
                i += 1
            result = expect_digit_or_repair(start)
            if result:
                return True
            while is_digit(char_code_at(text, i)):
                i += 1

        if i > start:
            # repair a number with leading zeros like "00789"
            num = text[start:i]
            has_invalid_leading_zero = bool(re.match(r'^0\d', num))

            output += f'"{num}"' if has_invalid_leading_zero else num
            return True

        return False


    def parse_keywords():
        result = parse_keyword('true', 'true')
        if result:
            return True

        result = parse_keyword('false', 'false')
        if result:
            return True

        result = parse_keyword('null', 'null')
        if result:
            return True

        # repair Python keywords True, False, None
        result = parse_keyword('True', 'true')
        if result:
            return True

        result = parse_keyword('False', 'false')
        if result:
            return True

        result = parse_keyword('None', 'null')
        if result:
            return True

        return False


    def parse_keyword(name, value):
        nonlocal i, output
        if text[i:i + len(name)] == name:
            output += value
            i += len(name)
            return True

        return False


    def parse_unquoted_string():
        nonlocal i, output
        # note that the symbol can end with whitespaces: we stop at the next delimiter
        start = i
        while i < len(text) and not is_delimiter(char_at(text, i, None)):
            i += 1

        if i > start:
            if char_code_at(text, i) == codeOpenParenthesis:
                # repair a MongoDB function call like NumberLong("2")
                # repair a JSONP function call like callback({...});
                i += 1

                processed = parse_value()

                if char_code_at(text, i) == codeCloseParenthesis:
                    # repair: skip close bracket of function call
                    i += 1
                    if char_code_at(text, i) == codeSemicolon:
                        # repair: skip semicolon after JSONP call
                        i += 1

                return True
            else:
                # repair unquoted string
                # also, repair undefined into null

                # first, go back to prevent getting trailing whitespaces in the symbol
                while is_whitespace(char_code_at(text, i - 1)) and i > 0:
                    i -= 1

                symbol = text[start:i]
                output += 'null' if symbol == 'undefined' else json.dumps(symbol)

                if char_code_at(text, i) == codeDoubleQuote:
                    # we had a missing start quote, but now we encountered the end quote, so we can skip that one
                    i += 1

                return True

        return False

    def expect_digit(start):
        nonlocal i
        if not is_digit(char_code_at(text, i)):
            num_so_far = text[start:i]
            raise JSONRepairError(f"Invalid number '{num_so_far}', expecting a digit {got()}", i)

    def expect_digit_or_repair(start):
        nonlocal i, output
        if i >= len(text):
            # repair numbers cut off at the end
            # this will only be called when we end after a '.', '-', or 'e' and does not
            # change the number more than it needs to make it valid JSON
            output += text[start:i] + '0'
            return True
        else:
            expect_digit(start)
            return False

    def throw_invalid_character(char):
        nonlocal i
        raise JSONRepairError('Invalid character ' + repr(char), i)

    def throw_unexpected_character():
        nonlocal i
        raise JSONRepairError('Unexpected character ' + repr(text[i]), i)

    def throw_unexpected_end():
        raise JSONRepairError('Unexpected end of json string', len(text))

    def throw_object_key_expected():
        nonlocal i
        raise JSONRepairError('Object key expected', i)

    def throw_colon_expected():
        nonlocal i
        raise JSONRepairError('Colon expected', i)

    def throw_invalid_unicode_character(start):
        nonlocal i
        end = start + 2
        while re.match(r"\w", text[end]):
            end += 1
        chars = text[start:end]
        raise JSONRepairError(f'Invalid unicode character "{chars}"', i)

    def got():
        nonlocal i
        return f"but got '{text[i]}'" if i < len(text) else 'but reached end of input'

    processed = parse_value()
    if not processed:
        throw_unexpected_end()

    processed_comma = parse_character(codeComma)
    if processed_comma:
        parse_whitespace_and_skip_comments()

    if is_start_of_value(char_at(text, i, None)) and ends_with_comma_or_newline(output):
        # start of a new value after end of the root level object: looks like
        # newline delimited JSON -> turn into a root level array
        if not processed_comma:
            # repair missing comma
            output = insert_before_last_whitespace(output, ',')

        parse_newline_delimited_json()
    elif processed_comma:
        # repair: remove trailing comma
        output = strip_last_occurrence(output, ',')
    # repair redundant end quotes
    while (char_code_at(text, i) == codeClosingBrace or char_code_at(text, i) == codeClosingBracket):
        i += 1
        parse_whitespace_and_skip_comments()

    if i >= len(text):
        # reached the end of the document properly
        return output  # or return output, depending on context
    else:
        throw_unexpected_character()


def at_end_of_block_comment(text, i):
    return text[i] == '*' and text[i + 1:i + 2] == '/'
