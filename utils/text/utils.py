import re
import unicodedata

from utils.text.pyunicode import unxmlify


def clean_text(raw_text):
    """
    Cleans raw text from XML tags and other rubbish

    Args:
        raw_text (str): String containing the text to be cleaned

    Returns:
        str: String containing the clean text
    """

    # Assign raw text
    cleaned_text = raw_text

    # Replace XML tags with respective characters
    cleaned_text = unxmlify(cleaned_text)

    # Remove or replace special characters and rubbish
    cleaned_text = cleaned_text.replace('•', '\n')
    cleaned_text = cleaned_text.replace('*', '\n')

    # Collapse multiple dashes
    cleaned_text = re.sub('-{2,}', '-', cleaned_text)

    # Return cleaned text
    return cleaned_text


def word_tokens(text):
    """
    Generates all possible word tokens from a sentence.

    Args:
        text (str): String containing words separated by spaces. Example: "how are you"

    Returns:
        list: A list with all the possible word tokens for the given sentence.
        Example: ['how', 'are', 'you', 'how are', 'are you', 'how are you']
    """

    # Split text as word list
    word_list = text.split(' ')
    n = len(word_list)

    # Iterate over all possible tokens of all possible word lengths
    output = []
    for i in range(1, n):
        for k in range(n - i + 1):
            output += [' '.join(word_list[k: k + i])]

    # Include sentence itself
    output += [text]

    return output


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
