import datetime


def now():
    """
    Return current timestamp as a datetime object that prints as "2022-06-28 09:46:21.677968"
    """

    return datetime.datetime.now()


def rescale(date, min_date, max_date):
    """
    Maps a date between min_date and max_date linearly to a number in [0, 1].
    Dates below or above the given range are mapped to 0 or 1, respectively.
    Dates are assumed to be in format yyyy-mm-dd.
    """

    assert min_date < max_date, f'min_date ({min_date}) should be strictly less than max_date ({max_date})'

    if date <= min_date:
        return 0.

    if date >= max_date:
        return 1.

    # Convert string dates to datetime.date objects
    a = datetime.datetime.strptime(min_date, '%Y-%m-%d').date()
    b = datetime.datetime.strptime(max_date, '%Y-%m-%d').date()
    x = datetime.datetime.strptime(date, '%Y-%m-%d').date()

    return (x-a)/(b-a)
