import RAKE
import nltk
from rake_nltk import Rake

from graphai.core.utils.text.clean import normalize

# Download nltk resources
nltk.download('stopwords', quiet=True)


# Initialise RAKE model
python_rake_model = RAKE.Rake(RAKE.SmartStopList())

# Initialise nltk-rake
nltk_rake_model = Rake()


def word_tokens(text):
    """
    Generates all possible word tokens from a sentence.

    Args:
        text (str): String containing words separated by spaces.

    Returns:
        list[str]: A list with all the possible word tokens for the given sentence.

    Examples:
        >>> word_tokens('how are you')
        ['how', 'are', 'you', 'how are', 'are you', 'how are you']
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


def rake_extract(text, use_nltk, split_words=False, return_scores=False, threshold='auto', filter_past_tenses_and_adverbs=False):
    """
    Extracts keywords from unconstrained text using python-rake or nltk-rake.

    Args:
        text (str): Text from which to extract the keywords.
        use_nltk (bool): Whether to use nltk-rake for keyword extraction, otherwise python-rake is used.
        split_words (bool): If True, keywords with more than one word are split. Default: False.
        return_scores (bool): If True, keywords are retured in a tuple with their RAKE score. Default: False.
        threshold (float or 'auto'): Minimal RAKE score below which extracted keywords are ignored. Default: 'auto',
            which translates to 10% of the maximum score.
        filter_past_tenses_and_adverbs (bool): Whether to filter out words in keywords which are past tenses, past
            participles or adverbs. Default: False.

    Returns:
        list[str] or list[tuple(str, float)]: A list of
            * str: Keywords, if split_words is True or return_scores is False.
            * tuple(str, float): A pair representing keywords and score, otherwise.

    Examples:
        >>> text = ' '.join([
        >>>     "Then a crowd a young boys they're a foolin' around in the corner",
        >>>     "Drunk and dressed in their best brown baggies and their platform soles",
        >>>     "They don't give a damn about any trumpet playin' band",
        >>>     "It ain't what they call 'rock and roll'"
        >>> ])
        >>> rake_extract(text, use_nltk=False)
        ['brown baggies', 'young boys', 'trumpet playin', 'corner drunk', 'platform soles']
    """

    # Execute RAKE model with given text
    if use_nltk:
        nltk_rake_model.extract_keywords_from_text(text)
        results = nltk_rake_model.get_ranked_phrases_with_scores()
        results = [(keywords, score) for score, keywords in results]
    else:
        results = python_rake_model.run(text)

    if threshold == 'auto':
        threshold = 0.1 * max([score for _, score in results])

    # Iterate over all results and compose keyword_list
    keyword_list = []
    for keywords, score in results:
        # Ignore scores below threshold
        if score <= threshold:
            continue

        # Nothing else to do if filtering is not needed
        if not filter_past_tenses_and_adverbs:
            keyword_list.append((keywords, score))
            continue

        # Split keywords into words and semantically tag them
        words = nltk.word_tokenize(keywords)
        tagged_words = nltk.pos_tag(words)

        # Remove words which are past tenses, past participles and adverbs
        filtered_words = []
        for word, tag in tagged_words:
            if tag not in ['VBD', 'VBN', 'RB']:
                filtered_words.append(word)

        # Append to keyword_list if there are words left
        if len(filtered_words) > 0:
            filtered_keywords = ' '.join(filtered_words)
            keyword_list.append((filtered_keywords, score))

    # If needed, split keywords into words and return
    if split_words:
        words = []
        for keywords, score in keyword_list:
            words.extend(word_tokens(keywords))
        return list(set(words))

    # Return keyword_list with or without scores, as needed
    if return_scores:
        return keyword_list

    return list(set([keywords for keywords, score in keyword_list]))


def get_keywords(text, use_nltk=False):
    """
    Normalize raw text and extract keywords.

    Args:
        text (str): Text to extract keywords from.
        use_nltk (bool): Whether to use nltk-rake for keyword extraction, otherwise python-rake is used. Default: False.

    Returns:
        list[str]: A list containing the keywords extracted from the text.

    Examples:
        >>> text = ' '.join([
        >>>     "<p>",
        >>>     "Then a crowd a young boys they're a foolin' around in the corner",
        >>>     "Drunk and dressed in their best brown baggies and their platform soles",
        >>>     "They don't give a damn about any trumpet playin' band",
        >>>     "It ain't what they call 'rock and roll'",
        >>>     "</p>"
        >>> ])
        >>> get_keywords(text)
        ['brown baggies', 'young boys', 'trumpet playin', 'corner drunk', 'platform soles']
    """

    text = normalize(text)

    # Extract keywords from text. We perform two extractions:
    #   * One with the full text.
    #   * One for each line.
    # This is done to account for slides with unconnected text in different lines.
    keyword_list = rake_extract(text, use_nltk)
    for line in text.split('\n'):
        keyword_list.extend(rake_extract(line, use_nltk))

    # Remove duplicates
    keyword_list = list(set(keyword_list))

    # If no keywords are extracted at all and text is short, use it as a single keyword
    if not keyword_list and len(text) < 50:
        keyword_list = [text]

    return keyword_list
