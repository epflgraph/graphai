import datetime

import pandas as pd


def now():
    """
    Return current timestamp as a datetime object that prints as "2022-06-28 09:46:21.677968"
    """

    return datetime.datetime.now()


def rescale(dates, min_date, max_date):
    """
    Maps a pandas Series between min_date and max_date linearly to [0, 1].
    Dates below or above the given range are mapped to 0 or 1, respectively.
    Dates are assumed to be in format yyyy-mm-dd.
    """

    a = pd.to_datetime(min_date, format='%Y-%m-%d')
    b = pd.to_datetime(max_date, format='%Y-%m-%d')
    x = pd.to_datetime(dates, format='%Y-%m-%d')

    return ((x - a) / (b - a)).clip(0, 1)

    # # Convert string dates to datetime.date objects
    # a = datetime.datetime.strptime(min_date, '%Y-%m-%d').date()
    # b = datetime.datetime.strptime(max_date, '%Y-%m-%d').date()
    # x = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    #
    # return (x-a)/(b-a)


def add_years(date, n_years):
    [year, month, day] = date.split('-')
    year = str(int(year) + n_years)
    return '-'.join([year, month, day])
