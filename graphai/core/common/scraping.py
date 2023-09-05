import numpy as np
import itertools
import requests
import re
from bs4 import BeautifulSoup
from unidecode import unidecode


def compare_strings(string1, string2):
    """
    Compares two input strings and returns array of 0 and 1s
    Args:
        string1: First string
        string2: Second string

    Returns:
        Array of 0s and 1s, with 1 indicating equality of characters at that index
    """
    if len(string1) != len(string2):
        raise ValueError("Both strings must be the same length")
    return np.array([int(char1 == char2) for char1, char2 in zip(string1, string2)])


def find_consecutive_runs(v, min_run=32):
    """
    Finds consecutive runs of equal elements in list (i.e. a sequence of 40 times the same value)
    Args:
        v: The list or string
        min_run: Minimum length of a consecutive run

    Returns:
        List of tuples (k1, k2, Length, Value)
    """
    k1 = 0
    k2 = 1
    consecutive_runs = []
    while True:
        if v[k1] == v[k2] and k2 <= len(v)-2:
            k2 += 1
        else:
            if k2-k1 > min_run:
                consecutive_runs += [(k1, k2-1, k2-k1, v[k1])]
            k1 = k2
            k2 = k1+1
            if k1 >= len(v)-1:
                break
    return consecutive_runs


def find_edge_patterns(content_stack, flip_strings=False):
    """
    Finds repeated patterns at the edge (beginning or end) of strings in a list of strings
    Args:
        content_stack: List of strings (which are the contents of webpages)
        flip_strings: Whether to flip the strings or not. Finds footer patterns if True, header patterns if False.

    Returns:
        List of patterns
    """
    # Remove duplicates and sort content stack in alphabetic order
    # so that content[k] and content[k+1] are as similar as possible
    content_stack = sorted(list(set(content_stack)), reverse=True)

    # Get the maximum length among all strings in the input stack
    # (this is needed to compare the strings later)
    max_length = max([len(c) for c in content_stack])

    # Right-pad all strings with empty space, so that they all have the same length
    # (this is a requirement of the "compare_strings" function)
    padded_content = []
    for c in content_stack:
        # If we're looking for footers instead of headers, just flip the strings
        if flip_strings:
            c = c[::-1]
        padded_content += [c.ljust(max_length)]
    padded_content = sorted(list(set(padded_content)), reverse=True)

    # Initialise output pattern stack
    pattern_stack = []

    # Loop and compare consecutive strings in the stack ... (k,k+1)
    # Since we've sorted the strings alphabetically, it should be (mostly) safe to only compare consecutive strings,
    # which will save a lot of time (compared to performing comparisons between every possible pair)
    for k in range(len(padded_content)-1):

        # Compare two strings and find the indexes are they are equal
        consecutive_runs = find_consecutive_runs(compare_strings(padded_content[k], padded_content[k+1]))

        # Did it find repeated patterns?
        # (note: we only care about the first one)
        if len(consecutive_runs) > 0:

            # Does the first repeated pattern start at index zero in the string?
            # (ie, is it an actual header?)
            if consecutive_runs[0][0] == 0:

                # If so, extract the repeated pattern
                pattern = padded_content[k][consecutive_runs[0][0]:consecutive_runs[0][1]+1].strip()

                # If it's a footer, flip the pattern string
                if flip_strings:
                    pattern = pattern[::-1]

                # Append it to output patterns stack
                pattern_stack += [pattern]

    # Sort output stack by decreasing string length
    pattern_stack = sorted(set(pattern_stack), key=len, reverse=True)

    # Output headers/footers stack
    return pattern_stack


def string_circular_shift(s, shift=1):
    """
    Performs a circular shift on a string by the provided value
    Args:
        s: String to shift
        shift: How many characters to shift the string by

    Returns:
        Shifted string
    """
    shift %= len(s)
    if shift == 0:
        return s
    else:
        return s[-shift:]+s[:len(s)-shift]


def find_spaces(s):
    """
    Finds spaces in string
    Args:
        s: Input string

    Returns:
        List of starting points of every space sequence in the string
    """
    return [m.start() for m in re.finditer(' ', s)]


def shift_to_max_correlation(s1, s2):
    """
    Shifts two strings to find their maximum correlation (as indicated by the positions of spaces in them)
    and the largest-matching string pattern with that shift
    Args:
        s1: First string
        s2: Second string

    Returns:
        (Optimal shift value, number of intersections with optimal shift,
        position of intersections, largest matching string pattern with optimal shift)
    """
    # The method works by first extracting the indexes of all spaces in each string
    s1_spaces = np.array(find_spaces(s1))
    s2_spaces = np.array(find_spaces(s2))

    # Determine direction of the iterative shift
    if sum(s1_spaces) >= sum(s2_spaces):
        d = 1
    else:
        d = -1

    # Get the maximum length of the two input strings
    # (this is needed to compare the shifted strings later)
    max_length = max(len(s1), len(s2))

    # Initialise all loop variables
    max_shift = 0
    shift = 0
    max_n_intersect = 0
    max_intersect_values = None

    # Start infinite loop of circular-shifting the 2nd string
    # and comparing it to 1st string to find the shift value
    # that results in maximum cross-correlation.
    while True:

        # Increase/decrease current shift value by 1
        shift += d

        # For 2nd string, shift all space indexes by 1
        s2_spaces += d

        # Find and count where space indexes intersect between the two strings
        # (this gives an estimation of string cross-correlation)
        intersect_values = np.intersect1d(s1_spaces, s2_spaces)
        n_intersect = len(intersect_values)

        # Did we find a higher correlation than in the previous iteration?
        # If yes, update output values
        if n_intersect > max_n_intersect:
            max_n_intersect = n_intersect
            max_shift = shift
            max_intersect_values = intersect_values

        # Limit the attempted shifts to +/-2048
        if shift > 2048 or shift < -2048:
            break

    # Apply the argmax(shift) found on the 2nd string, compare the two optimally aligned strings,
    # and find the indexes where the chain of characters are actually equal. Unlike find_edge_patterns,
    # we are looking for matches anywhere in the two strings, and not only at the beginning/end.
    consecutive_runs = find_consecutive_runs(
        compare_strings(s1.ljust(max_length), string_circular_shift(s2, shift=max_shift).ljust(max_length)),
        min_run=1024
    )

    # It can happen that two strings are optimally aligned, but there are no actual repeated substrings
    # So we have to check for that...
    if len(consecutive_runs) > 0:

        # If actual repeated substrings were found, keep only the best/longest one
        best_run = sorted(consecutive_runs, key=lambda tup: tup[2], reverse=True)[0]

        # For the best one, extract the actual substring of the repeated pattern
        intersect_pattern = s1[best_run[0]:best_run[1]+1].strip()

    # If no patterns are found...
    else:
        intersect_pattern = ''

    # Return argmax, the max correlation value, the array of common space indexes (for debugging purposes),
    # and the repeated text pattern (the one we care about)
    return max_shift, max_n_intersect, max_intersect_values, intersect_pattern


def find_repeated_patterns(content_stack, min_length=1024):
    """
    Finds repeated patterns in a list of strings, everywhere within the strings
    Args:
        content_stack: List of strings
        min_length: Minimum length of the matching substrings

    Returns:
        List of matching patterns
    """

    # Remove duplicates and sort content stack in aphabetic order
    # so that content[k] and content[k+1] are as similar as possible
    content_stack = sorted(list(set(content_stack)), reverse=True)

    # Initialise repeated text patterns stack
    repeated_patterns_stack = []

    # Loop over all combinations of indexes, unlike the edge pattern detection
    for k1, k2 in list(itertools.combinations(range(len(content_stack)), 2)):

        # Print status
        print('Analysing indices:', k1, k2)

        # For content (k1,k2) pair, shift the 2nd string to the point of maximum correlation with the 1st one
        # (we only care about the output text pattern)
        max_shift, max_n_intersect, intersect_values, intersect_pattern = \
            shift_to_max_correlation(content_stack[k1], content_stack[k2])

        # If the text pattern is long enough (as defined by user input), add to output stack
        if len(intersect_pattern) >= min_length:
            repeated_patterns_stack += [intersect_pattern]

    # Sort output stack by decreasing string length
    repeated_patterns_stack = sorted(set(repeated_patterns_stack), key=len, reverse=True)

    # Output repeated text patterns stack
    return repeated_patterns_stack


def extract_text_from_url(url, request_headers=None, max_length=None, tag_search_sequence=None):
    """
    Extracts text from webpage by URL
    Args:
        url: The url
        request_headers: Request headers for the headless browser
        max_length: Maximum length of the page contents
        tag_search_sequence: Sequence of tags to search for the contents in

    Returns:
        Contents of the page
    """
    if request_headers is None:
        request_headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                                         '(KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
                           'Accept': 'application/json'}
    if tag_search_sequence is None:
        tag_search_sequence = ['main', 'body', 'html']
    # Initialise output text as empty string
    text = ''

    # Fetch webpage from URL
    response = requests.get(url, allow_redirects=True, headers=request_headers)

    # Return if there's no response
    if response.text is None:
        print('Warning: No response from URL:', url)
        return ''

    # Parse and extract text
    soup = BeautifulSoup(response.text, 'lxml')
    soup_tag = None
    tag_name = None
    # Loop through input tag search sequence until one is found
    for tag_name in tag_search_sequence:

        # Find tag in HTML
        soup_tag = soup.find(tag_name)

        # Does tag exist? If so, exit the loop
        if soup_tag is not None:
            break

    # Return if none of the tags are found
    if soup_tag is None:
        print('Warning: None of the tags in the input sequence was found:', tag_search_sequence)
        return ''

    # Show status
    print('Extracting text content from "%s" tag in URL: %s' % (tag_name, url))

    # Fetch text from the first tag that was found in the sequence
    text = soup_tag.get_text(' ', strip=True)

    # Clean up text string
    text = unidecode(text).replace('"', ' ').replace('\\', ' ').replace('/', ' ').replace('\n', ' ')
    for k in range(32):
        text = text.replace('  ', ' ')
    text = text.strip()

    # Truncate text string if requested
    if max_length is not None:
        text = text[:max_length]

    # Return content
    return text

