import os
from datetime import datetime, timedelta

from graphai.core.common.caching import (
    VideoConfig,
    VideoDBCachingManager,
    AudioDBCachingManager,
    SlideDBCachingManager,
    delete_file
)
from graphai.core.common.common_utils import format_datetime_for_mysql
from graphai.core.common.config import config


def get_cleanup_interval():
    try:
        print("Reading cache cleanup interval from config")
        n_days = int(config['cleanup'].get('interval', 60))
    except Exception:
        print(
            "The cache cleanup interval could not be found in the config file, using 60 days as default. "
            "To use a different one, make sure to add a [cleanup] section with the interval parameter in days."
        )
        n_days = 60

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
                                         latest_date=oldest_acceptable_date, use_date_modified_col=True)
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
        print(f"Working on token {id_token}")
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


def cleanup_broken_symlinks():
    print('Removing broken symlinks')
    storage_config = VideoConfig()
    image_dir = storage_config.get_image_dir()
    # This command finds all the broken symlinks in image_dir and then deletes them
    os.system("find -L %s -type l -exec rm {} \\;" % image_dir)
    audio_dir = storage_config.get_audio_dir()
    # Same as above but for the audio_dir, since it's the other directory that has symlinks
    os.system("find -L %s -type l -exec rm {} \\;" % audio_dir)


def cleanup_old_cached_files():
    cleanup_interval = get_cleanup_interval()
    print(f"Deleting videos older than {cleanup_interval} days")
    oldest_acceptable_date = get_oldest_acceptable_date(cleanup_interval)
    old_video_ids = find_old_videos(oldest_acceptable_date)
    remove_old_files(old_video_ids)
    remove_origin_urls(old_video_ids)
    cleanup_broken_symlinks()


if __name__ == '__main__':
    cleanup_old_cached_files()
