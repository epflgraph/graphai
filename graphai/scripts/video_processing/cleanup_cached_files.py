from graphai.core.common.caching import (
    VideoConfig,
    VideoDBCachingManager,
    AudioDBCachingManager,
    SlideDBCachingManager
)
from graphai.definitions import CONFIG_DIR
from graphai.core.common.video import format_datetime_for_mysql

from datetime import datetime, timedelta
import configparser


def get_cleanup_interval():
    config_contents = configparser.ConfigParser()
    try:
        print('Reading cache cleanup interval configuration from file')
        config_contents.read(f'{CONFIG_DIR}/cache.ini')
        n_days = int(config_contents['CLEANUP'].get('interval', fallback='30'))
    except Exception:
        print(f'Could not read file {CONFIG_DIR}/cache.ini or '
              f'file does not have section [CLEANUP], falling back to defaults.')
        n_days = 30
    return n_days


def get_oldest_acceptable_date(cleanup_interval):
    current_date = datetime.now()
    td = timedelta(days=cleanup_interval)
    oldest_acceptable_date = current_date - td
    oldest_acceptable_date_str = format_datetime_for_mysql(oldest_acceptable_date)
    return oldest_acceptable_date_str


def find_old_videos(oldest_acceptable_date):
    db_manager = VideoDBCachingManager()
    results = db_manager.get_all_details(['date_added'], latest_date=oldest_acceptable_date)
    if results is None:
        return []
    id_list = list(results.keys())
    return id_list
