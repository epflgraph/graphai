import RAKE
import nltk
from rake_nltk import Rake
from models.text_utils import clean_text, word_tokens

# Initialise RAKE model
rake_model = RAKE.Rake(RAKE.SmartStopList())

# Initialise nltk-rake
r = Rake()


def rake_extract(text, split_words=False, return_scores=False, threshold=1):
    """
    Extracts keywords from unconstrained text using RAKE.

    Args:
        text (str): Text from which to extract the keywords.
        split_words (bool): If True, keywords with more than one word are split. Default: False.
        return_scores (bool): If True, keywords are retured in a tuple with their RAKE score. Default: False.
        threshold (float): Minimal RAKE score below which extracted keywords are ignored. Default: 1.

    Returns:
        list: A list of
            * str: Keywords, if split_words is True or return_scores is False.
            * (str, float): A pair representing keywords and score, otherwise.
    """

    # Execute RAKE model with given text
    results = rake_model.run(text)

    # Iterate over all results and compose keyword_list
    keyword_list = []
    for keywords, score in results:
        # Ignore scores below threshold
        if score <= threshold:
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


def rake_extract_nltk(text, split_words=False, return_scores=False, threshold=1):
    """
    Extracts keywords from unconstrained text using nltk-rake.

    Args:
        text (str): Text from which to extract the keywords.
        split_words (bool): If True, keywords with more than one word are split. Default: False.
        return_scores (bool): If True, keywords are retured in a tuple with their RAKE score. Default: False.
        threshold (float): Minimal RAKE score below which extracted keywords are ignored. Default: 1.

    Returns:
        list: A list of
            * str: Keywords, if split_words is True or return_scores is False.
            * (str, float): A pair representing keywords and score, otherwise.
    """

    # Execute RAKE model with given text
    r.extract_keywords_from_text(text)
    results = r.get_ranked_phrases_with_scores()

    # Iterate over all results and compose keyword_list
    keyword_list = []
    for score, keywords in results:
        # Ignore scores below threshold
        if score <= threshold:
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


def get_keyword_list(raw_text):
    """
    Clean raw text and extract keyword list.

    Args:
        raw_text (str): Text to be cleaned and used to extract keywords.

    Returns:
        list[str]: A list of keywords automatically extracted from the given text.
    """

    # Clean raw text of XML tags and other rubbish
    cleaned_text = clean_text(raw_text)

    # Extract keywords from clean text. We perform two extractions:
    #   * One with the full text.
    #   * One for each line.
    # This is done to account for slides with unconnected text in different lines.
    keyword_list = rake_extract(cleaned_text)
    for line in cleaned_text.split('\n'):
        keyword_list.extend(rake_extract(line))

    # Remove duplicates
    return list(set(keyword_list))


def get_keyword_list_nltk(raw_text):
    """
    Clean raw text and extract keyword list using nltk.

    Args:
        raw_text (str): Text to be cleaned and used to extract keywords.

    Returns:
        list[str]: A list of keywords automatically extracted from the given text.
    """

    # Clean raw text of XML tags and other rubbish
    cleaned_text = clean_text(raw_text)

    # Extract keywords from clean text. We perform two extractions:
    #   * One with the full text.
    #   * One for each line.
    # This is done to account for slides with unconnected text in different lines.
    keyword_list = rake_extract_nltk(cleaned_text)
    for line in cleaned_text.split('\n'):
        keyword_list.extend(rake_extract_nltk(line))

    # Remove duplicates
    return list(set(keyword_list))
