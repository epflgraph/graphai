from graphai.core.common.caching import (
    VideoConfig,
    VideoDBCachingManager,
    AudioDBCachingManager,
    SlideDBCachingManager,
    delete_file
)
from graphai.definitions import CONFIG_DIR
from graphai.core.common.video import format_datetime_for_mysql

from datetime import datetime, timedelta
import configparser
import numpy as np


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
    print('Finding old videos')
    # We only get the rows whose origin_token is not null,
    # because those are the ones that have not been cleaned up.
    results = db_manager.get_all_details(['origin_token', 'date_added'], allow_nulls=False,
                                         latest_date=oldest_acceptable_date)
    if results is None:
        return []
    id_list = list(results.keys())
    return id_list


def remove_origin_urls(id_list):
    video_db_manager = VideoDBCachingManager()
    print('Erasing `origin_token` (URL) column for old videos')
    for id_token in id_list:
        video_db_manager.insert_or_update_details(id_token, values_to_insert={'origin_token': None})


def remove_old_files(id_list):
    storage_config = VideoConfig()
    slide_db_manager = SlideDBCachingManager()
    audio_db_manager = AudioDBCachingManager()
    for id_token in id_list:
        current_video_path = storage_config.generate_filepath(id_token)
        print("Deleting video file")
        delete_file(current_video_path)

        print("Finding slide tokens")
        slide_tokens = slide_db_manager.get_details_using_origin(id_token, [])
        if slide_tokens is not None:
            slide_tokens = [x['id_token'] for x in slide_tokens]
            print("Deleting slide files")
            for slide_token in slide_tokens:
                current_slide_path = storage_config.generate_filepath(slide_token)
                delete_file(current_slide_path)

        print("Finding audio tokens")
        audio_tokens = audio_db_manager.get_details_using_origin(id_token, [])
        if audio_tokens is not None:
            audio_token = audio_tokens[0]['id_token']
            print("Deleting audio file")
            current_audio_path = storage_config.generate_filepath(audio_token)
            delete_file(current_audio_path)

