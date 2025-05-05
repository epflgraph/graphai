import errno
import json
import os
import random
import sys
import time
from datetime import datetime
import shutil
from collections import Counter

import wget


def make_sure_path_exists(path, file_at_the_end=False, full_perm=True):
    """
    Recursively creates the folders in a path.
    Args:
        path: The path that needs to exist (and will thus be created if it doesn't)
        file_at_the_end: Whether there is a filename at the end of the path
        full_perm: If set, the function will assign full permission (chmod 777) to each newly created folder
    Returns:
        None
    """
    if path == '/' or path == '':
        return
    if file_at_the_end:
        path = '/'.join(path.split('/')[:-1])
    if os.path.isdir(path):
        return
    try:
        parent_path = '/'.join(path.split('/')[:-1])
        make_sure_path_exists(parent_path)
        os.mkdir(path)
        if full_perm:
            os.chmod(path, 0o777)
    except OSError as exception:
        if exception.errno != errno.EEXIST and exception.errno != errno.EPERM:
            raise


def file_exists(file_path):
    """
    Checks whether a given file exists
    Args:
        file_path: Path of the file

    Returns:
        True if file exists, False otherwise
    """
    return os.path.exists(file_path)


def write_text_file(filename_with_path, contents):
    """
    Writes contents to text file
    Args:
        filename_with_path: Full path of the file
        contents: Textual contents

    Returns:
        None
    """
    with open(filename_with_path, 'w') as f:
        f.write(contents)


def write_json_file(filename_with_path, d):
    """
    Writes dictionary to JSON file
    Args:
        filename_with_path: Full path of the file
        d: Dictionary to write

    Returns:
        None
    """
    with open(filename_with_path, 'w') as f:
        json.dump(d, f)


def read_text_file(filename_with_path):
    """
    Opens and reads the contents of a text file
    Args:
        filename_with_path: Full path of the file

    Returns:
        Contents of the file
    """
    with open(filename_with_path, 'r') as f:
        return f.read()


def read_json_file(filename_with_path):
    """
    Reads the contents of a JSON file
    Args:
        filename_with_path: Full path of the file

    Returns:
        Dictionary containing contents of the JSON file
    """
    with open(filename_with_path, 'r') as f:
        return json.load(f)


def format_datetime_for_mysql(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_mysql_datetime(dt):
    return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")


def get_current_datetime():
    """
    Returns current datetime formatted for MySQL
    Returns:
        Datetime string
    """
    current_datetime = format_datetime_for_mysql(datetime.now())
    return current_datetime


def invert_dict(d):
    return {d[k]: k for k in d}


def strtobool(val):
    """Convert a string representation of truth to True or False.
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; False values
    are 'n', 'no', 'f', 'false', 'off', and '0'. Raises ValueError if
    'val' is anything else. Case-insensitive.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0', ''):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


def copy_file_within_folder(folder_name, src_file, dest_file):
    shutil.copyfile(os.path.join(folder_name, src_file), os.path.join(folder_name, dest_file))


TEXT_LIST_TO_STRING_SEPARATOR = ' [{[!!SEP!!]}] '


def convert_list_to_text(str_or_list):
    if not isinstance(str_or_list, list):
        return str_or_list
    str_or_list = [x if x is not None else '' for x in str_or_list]
    return TEXT_LIST_TO_STRING_SEPARATOR.join(str_or_list)


def convert_text_back_to_list(s, return_list=False):
    results = s.split(TEXT_LIST_TO_STRING_SEPARATOR)
    if len(results) == 1 and not return_list:
        return results[0]
    return results


def generate_random_token():
    """
    Generates a random string using the current time and a random number to be used as a token.
    Returns:
        Random token
    """
    return ('%.06f' % time.time()).replace('.', '') + '%08d' % random.randint(0, int(1e7))


def get_file_size(file_path):
    if file_path is None:
        return None
    try:
        return os.path.getsize(file_path)
    except OSError:
        return None


def retrieve_generic_file_from_generic_url(url, output_filename_with_path, output_token):
    """
    Retrieves a generic file from a given URL using WGET and stores it locally.
    Args:
        url: the URL
        output_filename_with_path: Path of output file
        output_token: Token of output file

    Returns:
        Output token if successful, None otherwise
    """
    try:
        wget.download(url, output_filename_with_path)
    except Exception as e:
        print(e, file=sys.stderr)
        return None
    if file_exists(output_filename_with_path):
        return output_token
    else:
        return None


def is_url(s):
    if s.startswith('http://') or s.startswith('https://'):
        return True
    return False


def is_effective_url(s):
    if s.startswith('gdrive://'):
        return True
    return is_url(s)


def is_token(s):
    return not is_url(s)


def is_pdf(s):
    return s.endswith('.pdf')


def get_most_common_element(l, remove_nulls=True):
    if remove_nulls:
        l = [x for x in l if x is not None]
    if len(l) == 0:
        return None
    return Counter(l).most_common(1)[0][0]
