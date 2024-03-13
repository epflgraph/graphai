import errno
import json
import os
from datetime import datetime


def make_sure_path_exists(path, file_at_the_end=False, full_perm=True):
    """
    Recursively creates the folders in a path.
    Args:
        path: The path that needs to exist (and will thus be created if it doesn't)
        file_at_the_end: Whether there is a filename at the end of the path

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
