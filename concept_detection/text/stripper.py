import re
import unicodedata
import mwparserfromhell
from mwparserfromhell.wikicode import Wikicode
from mwparserfromhell.nodes.argument import Argument
from mwparserfromhell.nodes.comment import Comment
from mwparserfromhell.nodes.external_link import ExternalLink
from mwparserfromhell.nodes.heading import Heading
from mwparserfromhell.nodes.html_entity import HTMLEntity
from mwparserfromhell.nodes.tag import Tag
from mwparserfromhell.nodes.template import Template
from mwparserfromhell.nodes.text import Text
from mwparserfromhell.nodes.wikilink import Wikilink

# Pronunciation & IPA link codes from https://en.wikipedia.org/wiki/Template:IPAc-en
codes = ['lang', 'local', 'ipa', 'also', 'uk', 'us', 'uklang', 'uslang', 'ukalso', 'usalso', 'alsouk', 'alsous']


def parse_template(node):
    # node.name is a Wikicode object, we parse it to a lowercase string
    name = parse(node.name).lower()

    # Full phonetic transcription of a word or expression
    if 'ipac' in name:
        s = ''
        for param in node.params:
            # Exclude named parameters, e.g. "audio=..."
            if param.showkey:
                continue

            # param.value is a Wikicode object, we parse it to string
            value = parse(param.value)

            # Exclude modifiers, e.g. "lang" or "US"
            if value.lower() in codes:
                continue

            s += value
        return s

    # Phonetic transcription of a phoneme
    if 'ipa' in name:
        # First parameter should be the transcription
        if not node.has(1):
            return ''

        return parse(node.get(1).value)

    # Block in different language
    if 'lang' in name:
        # Case {{lang|fr|Bonjour}}
        if node.has(2):
            return parse(node.get(2).value)

        # Case {{lang-fr|Bonjour}}
        if node.has(1):
            return parse(node.get(1).value)

        return ''

    # Nihongo: Kanji/kana segments
    if 'nihongo' in name:
        params = ['', '', '']

        for i in range(len(params)):
            if node.has(i + 1):
                params[i] = parse(node.get(i + 1).value)

        if params[0] and params[1] and params[2]:
            return f'{params[0]} ({params[1]}, {params[2]})'

        if params[0] and params[1]:
            return f'{params[0]} ({params[1]})'

        if params[0] and params[2]:
            return f'{params[0]} ({params[2]})'

        if params[1] and params[2]:
            return f'{params[2]} ({params[1]})'

        if params[2]:
            return params[2]

        if params[1]:
            return params[1]

        if params[0]:
            return params[0]

        return ''

    # Formatted number
    if 'val' in name:
        s = ''

        # Parameter p is the prefix
        if node.has('p'):
            s += parse(node.get('p').value)

        # First parameter is the number
        if node.has(1):
            s += parse(node.get(1).value)

        # Parameter e is the exponent
        if node.has('e'):
            s += 'e' + parse(node.get('e').value)

        # Parameter u or ul is the unit (numerator)
        if node.has('u'):
            s += parse(node.get('u').value)
        elif node.has('ul'):
            s += parse(node.get('ul').value)

        # Parameter up or upl is the unit (denominator)
        if node.has('up'):
            s += parse(node.get('up').value)
        elif node.has('upl'):
            s += parse(node.get('upl').value)

        # Parameter s is the suffix
        if node.has('s'):
            s += parse(node.get('s').value)

        return s

    # Gap-separated values
    if 'gaps' in name:
        # Parameter lhs is the left hand-side
        prefix = ''
        if node.has('lhs'):
            prefix += parse(node.get('lhs').value) + '='

        # Parameter e is the exponent, base is the base
        suffix = ''
        if node.has('e'):
            if node.has('base'):
                suffix += 'x' + parse(node.get('base').value) + '^' + parse(node.get('e').value)
            else:
                suffix += 'e' + parse(node.get('e').value)

        # Parameter u is the unit
        if node.has('u'):
            suffix += parse(node.get('u').value)

        # Unnamed params are to be concatenated with space separators
        unnamed_params = [parse(param.value) for param in node.params if not param.showkey]

        return prefix + ' '.join(unnamed_params) + suffix

    # Nowrap
    if 'nowrap' in name:
        return ''.join([parse(param.value) for param in node.params])

    # Mathematical variable mvar
    if 'mvar' in name:
        # First parameter should be the variable
        if not node.has(1):
            return ''

        return parse(node.get(1).value)

    # Mathematical fraction sfrac
    if 'sfrac' in name:
        params = ['', '', '']

        for i in range(len(params)):
            if node.has(i + 1):
                params[i] = parse(node.get(i + 1).value)

        # {{sfrac|A|B|C}} means AB/C
        if params[2]:
            return params[0] + params[1] + '/' + params[2]

        # {{sfrac|A|B}} means A/B
        if params[1]:
            return params[0] + '/' + params[1]

        # {{sfrac|A}} means 1/A
        if params[0]:
            return '1/' + params[0]

        return ''

    # Main article template
    if 'main' in name:
        # First parameter should be the page title
        if not node.has(1):
            return ''

        return parse(node.get(1).value)

    # Quote template
    if 'quote' in name:
        if node.has('text'):
            return parse(node.get('text').value)

        if node.has(1):
            return parse(node.get(1).value)

        return ''

    # Default behavior for Template
    return ''


def parse(node):
    # node is actually None
    if node is None:
        return ''

    # node is actually a Wikicode
    if isinstance(node, Wikicode):
        return ''.join([parse(child) for child in node.nodes])

    # Argument node
    if isinstance(node, Argument):
        return str(node.__strip__())

    # Comments are removed
    if isinstance(node, Comment):
        return ''

    # External links. We keep the title but not the URL
    if isinstance(node, ExternalLink):
        return parse(node.title)

    # Headings are removed
    if isinstance(node, Heading):
        return ''

    # HTML entities (like &nbsp;)
    if isinstance(node, HTMLEntity):
        return node.normalize()

    # Tag nodes have wikicode as content
    if isinstance(node, Tag):
        # Tag is ex or implicitly self-closing, like <br> or <img/>
        if node.implicit or node.self_closing:
            return ' '

        # Exclude reference, math tags and tables
        tag = parse(node.tag)
        if tag in ['ref', 'math', 'table']:
            return ''

        # Parse tag contents by default
        return parse(node.contents)

    # Template nodes require processing
    if isinstance(node, Template):
        return parse_template(node)

    # Text nodes
    if isinstance(node, Text):
        # Need to check for wrongly matched templates due to a mwparserfromhell limitation
        # concerning template transclusion.
        #
        # e.g. {{DISPLAYTITLE:SL<sub>2</sub>('''R''')}} is parsed as
        #   - Text node {{DISPLAYTITLE:SL
        #   - Tag node <sub>2</sub>
        #   - Text node (
        #   - Tag node '''R'''
        #   - Text node )}}
        #
        # instead of as a Template node. We correct the Text nodes containing unmatched {{ or }}
        # as a patch to this behavior. In the previous example, we return the string
        #   DISPLAYTITLE:SL2(R)
        #
        # More info: https://github.com/earwig/mwparserfromhell#limitations

        text = node.value
        if '{{' in text and '}}' not in text:
            return ''.join(text.split('{{'))
        if '{{' not in text and '}}' in text:
            return ''.join(text.split('}}'))

        return text

    # Wikilink nodes. We exclude some types, otherwise parse inner text
    if isinstance(node, Wikilink):
        for link_type in ['File', 'Image', 'Category']:
            if link_type in node.title:
                return ''

        # If the link has text, parse it and return it
        text = parse(node.text)
        if text:
            return text

        # Otherwise parse link title
        return parse(node.title)

    # Default behavior for unspecified tags
    return ''


def parse_hatnote(node):
    # Hatnotes can only be templates
    if not isinstance(node, Template):
        return ''

    # Ignore redirects, date hatnotes and other irrelevant hatnotes
    name = node.name.lower()
    irrelevant_hatnotes = ['redirect', 'date', 'engvarb', 'coord', 'english']
    for irrelevant_hatnote in irrelevant_hatnotes:
        if irrelevant_hatnote in name:
            return ''

    # For the "About" hatnote only the first child is returned, the rest are disambiguation alternatives
    if 'about' in name and node.params:
        return parse(node.params[0].value)

    # Concatenate all parsed parameters by default. This could be perhaps refined depending on the template.
    return ' '.join([parse(param.value) for param in node.params])


def clean(text):
    # Normalize line breaks and tabs
    text = re.sub('[\r\f\v]', '\n', text)
    text = re.sub('\t', ' ', text)

    # Strip lines
    lines = text.split('\n')
    text = '\n'.join([line.strip() for line in lines])

    # Remove lines with only punctuation
    text = re.sub('\n[,;.:+*·-]+', '\n', text)

    # Collapse consecutive line breaks
    text = re.sub('\n{2,}', '\n', text)

    # Collapse consecutive whitespaces
    text = re.sub(' +', ' ', text)

    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)

    return text


def get_section_headings(section):
    return [clean(parse(heading.title)) for heading in section.filter_headings()]


def get_categories(section):
    # Pattern:
    #   \[\[        - leading [[
    #   Category    - literal
    #   \s*         - any number of whitespaces
    #   :           - literal
    #   \s*         - any number of whitespaces
    #   (.+)        - capturing group, at least one character
    #   \s*         - any number of whitespaces
    #   \]\]        - trailing ]]
    matches = re.findall(r'\[\[Category\s*:\s*(.+)\s*\]\]', section)

    # Some categories have format [[Category:something]], some others have format
    # [[Category:something|something else]]. We clean the second type.
    categories = [re.split(r'\|', match, maxsplit=1)[0] for match in matches]

    categories = [clean(category) for category in categories]

    return categories


def get_tables(section):
    return [clean(parse(table.contents)) for table in section.filter_tags(matches=lambda node: node.tag == 'table')]


def get_captions(section):
    wikilinks = section.filter_wikilinks(matches=lambda node: 'file:' in node.title.lower())

    captions = []
    for wikilink in wikilinks:
        # Split string like 'thumb|right|500px|alt=Exterior of a large English Baroque palace|Blenheim Palace'
        pieces = [piece.strip() for piece in parse(wikilink.text).split('|')]

        # Remove options based on reserved keywords
        reserved_keywords = ['thumb', 'frame', 'border', 'right', 'left', 'center', 'none', 'baseline', 'middle', 'sub', 'super', 'top', 'bottom', 'upright', 'px', 'link', 'page', 'alt', 'lang']
        non_reserved_pieces = []
        for piece in pieces:
            if all([reserved_keyword not in piece.lower() for reserved_keyword in reserved_keywords]) or (' ' in piece and '=' not in piece):
                non_reserved_pieces.append(piece)

        # Extract caption
        if non_reserved_pieces:
            # Last piece is the caption
            caption = non_reserved_pieces[-1]
            caption = mwparserfromhell.parse(caption)
            captions.append(clean(parse(caption)))

        # Extract alt_text piece as well
        for piece in pieces:
            if 'alt=' in piece:
                alt_text = piece.replace('alt=', '')
                alt_text = mwparserfromhell.parse(alt_text)
                captions.append(clean(parse(alt_text)))
                break

    return captions


def split_hatnotes_opening_text(section):
    hatnotes = []
    opening_text = ''

    started = False
    for child in section.nodes:
        if not started and not isinstance(child, Template):
            text = parse(child)
            if text.strip():
                opening_text += text
                started = True
            continue

        if started:
            opening_text += parse(child)
        else:
            hatnote = clean(parse_hatnote(child))
            if hatnote:
                hatnotes.append(hatnote)

    return hatnotes, clean(opening_text)


def strip(page_content):
    """
    Strips wikimarkup from a string and returns a human-readable version by parsing the markup.

    Args:
        page_content (str): String containing the page content in wikimarkup format.

    Returns:
        dict: Dictionary containing several fields derived from the markup parsing, containing a 'text' field which
            represents a human-readable version of the input text.
    """
    
    wikicode = mwparserfromhell.parse(page_content)

    sections = wikicode.get_sections(levels=[2], include_lead=True)

    # Sections not processed for text
    text_excluded_sections = ['See also', 'References', 'References and notes', 'External links', 'Footnotes', 'Further reading']

    # Sections processed for auxiliary text
    auxiliary_text_sections = ['See also']

    stripped_page = {
        'heading': [],
        'opening_text': '',
        'text': '',
        'auxiliary_text': []
    }
    for section in sections:
        # Extract list of section headings
        section_headings = get_section_headings(section)
        stripped_page['heading'].extend([h for h in section_headings if h not in text_excluded_sections])

        # Captions and tables are extracted and stored as auxiliary text
        stripped_page['auxiliary_text'].extend(get_captions(section))
        stripped_page['auxiliary_text'].extend(get_tables(section))

        # Extract section title from headings and proceed depending on the case
        section_title = section_headings[0] if section_headings else ''
        if not section_title:
            # First section, split hatnotes and opening_text and store them as
            # auxiliary text and opening_text/text, respectively.
            hatnotes, opening_text = split_hatnotes_opening_text(section)
            stripped_page['auxiliary_text'].extend(hatnotes)
            stripped_page['opening_text'] = opening_text
            stripped_page['text'] += f'\n{opening_text}'
        else:
            # Named section, may be processed as either text or auxiliary text
            if section_title in auxiliary_text_sections and section_title in text_excluded_sections:
                # Section to be processed only as auxiliary text
                text = clean(parse(section))
                stripped_page['auxiliary_text'].append(text)
            elif section_title in auxiliary_text_sections and section_title not in text_excluded_sections:
                # Section to be processed as both text and auxiliary text
                text = clean(parse(section))
                stripped_page['auxiliary_text'].append(text)
                stripped_page['text'] += f'\n{text}'
            elif section_title not in auxiliary_text_sections and section_title not in text_excluded_sections:
                # Section to be processed only as text
                text = clean(parse(section))
                stripped_page['text'] += f'\n{text}'

    return stripped_page

