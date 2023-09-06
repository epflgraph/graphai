import numpy as np
import itertools
import requests
import re
from bs4 import BeautifulSoup
from unidecode import unidecode
import hashlib


REQ_HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                             '(KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
               'Accept': 'application/json'}

PAGE_TYPES_DICT = {
    'homepage'   			: 'homepage',
    'home'   				: 'homepage',
    'about' 				: 'about',
    'about-us' 				: 'about',
    'alumni' 				: 'people',
    'alumni-interns' 		: 'people',
    'completed-projects' 	: 'research',
    'contact' 				: 'contacts',
    'contacts' 				: 'contacts',
    'current-projects' 		: 'research',
    'education' 			: 'teaching',
    'events' 				: 'activities',
    'facilities' 			: 'facilities',
    'former-members' 		: 'people',
    'funding' 				: 'funding',
    'group' 				: 'people',
    'lab-members' 			: 'people',
    'lectures' 				: 'teaching',
    'members' 				: 'people',
    'news' 					: 'news',
    'open-positions' 		: 'jobs',
    'openings' 				: 'jobs',
    'openpositions' 		: 'jobs',
    'our-research' 			: 'research',
    'outreach' 				: 'activities',
    'past-members' 			: 'people',
    'past-research' 		: 'research',
    'people' 				: 'people',
    'previousresearch' 		: 'research',
    'projects' 				: 'research',
    'publications' 			: 'publications',
    'research' 				: 'research',
    'research-projects' 	: 'research',
    'resources' 			: 'resources',
    'scientific-activities' : 'activities',
    'seminars' 				: 'activities',
    'student-projects' 		: 'student projects',
    'student_projects' 		: 'student projects',
    'studentprojects' 		: 'student projects',
    'teaching' 				: 'teaching',
    'teaching-projects' 	: 'teaching',
    'team'					: 'people',
    'technical-reports' 	: 'publications',
}


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
        request_headers = REQ_HEADERS
    if tag_search_sequence is None:
        tag_search_sequence = ['main', 'body', 'html']

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


def check_url(test_url, request_headers=None):
    """
    Checks if a URL is accessible and returns the fully resolved URL if so
    Args:
        test_url: Starting URL
        request_headers: Headers of the request, uses defaults if None

    Returns:
        The validated URL, status message, and status code
    """
    if request_headers is None:
        request_headers = REQ_HEADERS

    validated_url = None

    try:
        # Fetch webpage from URL
        response = requests.get(test_url, allow_redirects=True, headers=request_headers)

        # Get status code and store locally
        status_code = response.status_code

        # Is the URL reachable?
        # Yes.
        if response.status_code == 200:

            # Update status message
            status_msg = f"The URL is reachable. Final URL after redirection (if any) is: {response.url}"

            # Get validated URL
            validated_url = response.url

            # Remove trailing forward slash from URL
            if validated_url.endswith('/'):
                validated_url = validated_url[:-1]

        # No. (all bellow)
        elif response.status_code >= 300 and response.status_code < 400:
            status_msg = "The URL is a redirect"

        elif response.status_code == 400:
            status_msg = "Bad Request"

        elif response.status_code == 401:
            status_msg = "Unauthorized"

        elif response.status_code == 403:
            status_msg = "Forbidden"

        elif response.status_code == 404:
            status_msg = "The URL is not found"

        elif response.status_code == 500:
            status_msg = "Internal Server Error"

        elif response.status_code == 501:
            status_msg = "Not Implemented"

        elif response.status_code == 502:
            status_msg = "Bad Gateway"

        elif response.status_code == 503:
            status_msg = "Service Unavailable"

        else:
            status_msg = f"An error occurred while trying to access the URL. Error Code: {response.status_code}"

    # Parse general errors
    except requests.exceptions.RequestException as err:
        status_msg = f"URL does not exist or the URL request failed. Error: {err}"
        status_code = -1
    # if type(err) is requests.exceptions.SSLError:
    # 	self.ssl_error = True
    # elif type(err) is requests.exceptions.ConnectionError:
    # 	self.connection_error = True

    return validated_url, status_msg, status_code


def initialize_url(url):
    """
    Initializes the provided URL by determining its protocol (http or https) and validating it
    Args:
        url: The URL to initialize

    Returns:
        The validated base URL and the original (corrected) base URL
    """

    # Ignore initial tasks if no input URL is provided
    if url is None:
        return None, None

    # Extract base URL from input
    base_url = url.replace('https://www.','').replace('http://www.','').replace('https://','').replace('http://','')

    # Test for all 4 URL combinations
    for test_url in ['https://www.'+base_url, 'http://www.'+base_url, 'https://'+base_url, 'http://'+base_url]:

        # Check URL for a valid address
        validated_url = check_url(test_url=test_url)

        # Stop the test if a reachable URL was found
        if validated_url is not None:
            return validated_url, base_url
    return None, base_url


def get_sublinks(validated_url, request_headers=None):
    """
    Retrieves all the sublinks of a URL
    Args:
        validated_url: Base validated URL
        request_headers: Headers of the request, uses defaults if None

    Returns:
        List of sublinks, a data dictionary mapping each sublink to a dict to be filled later, and the validated URL
    """
    if request_headers is None:
        request_headers = REQ_HEADERS
    # Return empty list if URL hasn't been validated
    if validated_url is None:
        return []

    # Remove trailing forward slash from URL
    if validated_url.endswith('/'):
        validated_url = validated_url[:-1]

    # Fetch webpage from URL
    response = requests.get(validated_url, allow_redirects=True, headers=request_headers)

    # Parse and extract links
    soup = BeautifulSoup(response.text, 'lxml')
    links = []
    for link in soup.find_all('a'):
        link_url = link.get('href')
        if link_url is not None and (link_url.startswith('http') or link_url.startswith('/')):
            if link_url.startswith('/'):
                if validated_url.endswith('/'):
                    link_url = validated_url[:-1]+link_url
                else:
                    link_url = validated_url+link_url
            if validated_url in link_url:
                if link_url.endswith('/'):
                    link_url = link_url[:-1]
                if not link_url.endswith('.pdf') and not link_url.endswith('.png') and not link_url.endswith('.jpg') and not link_url.endswith('.mp3') and not link_url.endswith('.mp4') and not link_url.endswith('wp-admin'):
                    links.append(link_url)

    # Sort and make sublinks list unique
    sublinks = sorted(list(set([validated_url]+links)))
    data = dict()

    # Initialise data dictionary
    for sublink in sublinks:
        if sublink not in data:
            data.update({sublink : {'id':None, 'content':'', 'pagetype':None}})

    # Return list of sublinks
    return sublinks, data, validated_url


def parse_page_type(url, validated_url):
    """
    Parses the type of a page according to predefined types
    Args:
        url: Given URL
        validated_url: Base validated URL

    Returns:
        Page type
    """

    # Check if it's home page
    if url == validated_url:
        return 'homepage'

    # Search for common keywords in URL
    for keywords in PAGE_TYPES_DICT:
        if keywords in url:
            page_type = PAGE_TYPES_DICT[keywords]
            return page_type

    # Return NoneType if not found
    return None


def process_all_sublinks(data, base_url, validated_url):
    """
    Processes all the sublinks and extracts their contents
    Args:
        data: Data dict (which will be modified)
        base_url: Corrected (but not validated) base URL (used for the id of the sublink)
        validated_url: Validated base URL

    Returns:
        Modified data dict
    """
    # Loop over all sublinks
    for sublink in data:

        # Print status
        print('Extracting content from:', sublink)

        # Generate unique identifier
        sublink_id = base_url.split('/')[0].replace('.', '-') + '-' + \
                     hashlib.md5(sublink.encode('utf-8')).hexdigest()[:8]

        # Parse webpage type from URL
        page_type = parse_page_type(sublink, validated_url)

        # Fetch content from webpage
        content = extract_text_from_url(sublink, max_length=16384)

        # Update data dictionary
        data[sublink].update({'id': sublink_id, 'pagetype': page_type, 'content': content})

    # Return modified data dictionary
    return data


def remove_headers(data):
    """
    Removes all headers and footers from the data dict, which contains the contents of all the sublinks of a base URL
    Args:
        data: Data dict

    Returns:
        Modified data dict with all headers and footers eliminated
    """

    # Return if no data to process
    if len(data) == 0:
        return False

    # Generate list of sublinks (data keys)
    sublinks_list = sorted(list(data.keys()))

    # Generate content stack
    content_stack = [data[k]['content'] for k in data if len(data[k]['content'])>=2]

    # Detect headers
    headers_to_delete = find_edge_patterns(content_stack=content_stack, flip_strings=False)

    # Remove headers from all extracted text
    for h in headers_to_delete:
        print('\nDeleting header:', h)
        for k in range(len(data)):
            if data[sublinks_list[k]]['content'].startswith(h):
                data[sublinks_list[k]]['content'] = data[sublinks_list[k]]['content'].replace(h,'').strip()

    # Generate content stack
    content_stack = [data[k]['content'] for k in data]

    # Detect footers
    footers_to_delete = find_edge_patterns(content_stack=content_stack, flip_strings=True)

    # Remove footers from all extracted text
    for f in footers_to_delete:
        print('\nDeleting footer:', f)
        for k in range(len(data)):
            if data[sublinks_list[k]]['content'].endswith(f):
                data[sublinks_list[k]]['content'] = data[sublinks_list[k]]['content'].replace(f,'').strip()

    # Return modified data dictionary
    return data


def remove_long_patterns(data, min_length=1024):
    """
    Removes all long patterns from the data dict, which contains the contents of all the sublinks of a base URL
    Args:
        data: Data dict
        min_length: Minimum length of a long pattern

    Returns:
        Modified data dict with all long patterns eliminated
    """

    # Return if no data to process
    if len(data)==0:
        return False

    # Generate list of sublinks (data keys)
    sublinks_list = sorted(list(data.keys()))

    # Generate content stack
    content_stack = [data[k]['content'] for k in data if len(data[k]['content'])>=2]

    # Detect long patterns
    patterns_to_delete = find_repeated_patterns(content_stack, min_length=min_length)

    # Remove footers from all extracted text
    for p in patterns_to_delete:
        print('Deleting pattern of length', len(p), ' ---> ', p)
        for k in range(len(data)):
            data[sublinks_list[k]]['content'] = data[sublinks_list[k]]['content'].replace(p,'').strip()

    # Return modified data dictionary
    return data
