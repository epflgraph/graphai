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
