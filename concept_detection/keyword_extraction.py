import RAKE
import nltk
from rake_nltk import Rake
from utils.text.clean import clean
from utils.text.utils import word_tokens

# Download nltk resources
nltk.download('stopwords', quiet=True)


# Initialise RAKE model
python_rake_model = RAKE.Rake(RAKE.SmartStopList())

# Initialise nltk-rake
nltk_rake_model = Rake()


def rake_extract(text, use_nltk, split_words=False, return_scores=False, threshold=1):
    """
    Extracts keywords from unconstrained text using python-rake or nltk-rake.

    Args:
        text (str): Text from which to extract the keywords.
        use_nltk (bool): Whether to use nltk-rake for keyword extraction, otherwise python-rake is used.
        split_words (bool): If True, keywords with more than one word are split. Default: False.
        return_scores (bool): If True, keywords are retured in a tuple with their RAKE score. Default: False.
        threshold (float): Minimal RAKE score below which extracted keywords are ignored. Default: 1.

    Returns:
        list: A list of
            * str: Keywords, if split_words is True or return_scores is False.
            * (str, float): A pair representing keywords and score, otherwise.
    """

    # Execute RAKE model with given text
    if use_nltk:
        nltk_rake_model.extract_keywords_from_text(text)
        results = nltk_rake_model.get_ranked_phrases_with_scores()
        results = [(keywords, score) for score, keywords in results]
    else:
        results = python_rake_model.run(text)

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


def get_keyword_list(text, use_nltk=False):
    """
    Clean raw text and extract keyword list.

    Args:
        text (str): Text to be cleaned and used to extract keywords.
        use_nltk (bool): Whether to use nltk-rake for keyword extraction, otherwise python-rake is used. Default: False.

    Returns:
        list[str]: A list of keywords automatically extracted from the given text.
    """

    text = clean(text)

    # Extract keywords from clean text. We perform two extractions:
    #   * One with the full text.
    #   * One for each line.
    # This is done to account for slides with unconnected text in different lines.
    keyword_list = rake_extract(text, use_nltk)
    for line in text.split('\n'):
        keyword_list.extend(rake_extract(line, use_nltk))

    # Remove duplicates
    return list(set(keyword_list))
