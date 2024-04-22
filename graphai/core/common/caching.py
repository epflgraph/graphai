import os
from datetime import datetime

from db_cache_manager.db import DBCachingManagerBase

from graphai.core.common.common_utils import make_sure_path_exists, file_exists
from graphai.core.common.config import config

ROOT_VIDEO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../Storage/'))
# Formats with a . in their name indicate single files, whereas formats without a . indicate folders (e.g. '_slides')
VIDEO_SUBFOLDER = 'Video'
VIDEO_FORMATS = ['.mkv', '.mp4', '.avi', '.mov', '.flv']
AUDIO_SUBFOLDER = 'Audio'
AUDIO_FORMATS = ['.mp3', '.flac', '.wav', '.aac', '.ogg']
IMAGE_SUBFOLDER = 'Image'
IMAGE_FORMATS = ['.png', '.tiff', '.jpg', '.jpeg', '.bmp', '_slides', '_all_frames']
OTHER_SUBFOLDER = 'Other'
TRANSCRIPT_SUBFOLDER = 'Transcripts'
TRANSCRIPT_FORMATS = ['_transcript.txt', '_subtitle_segments.json']
TEMP_SUBFOLDER = 'Temp'

# Cache config parameters
DEFAULT_SCHEMA = 'cache_graphai'
try:
    cache_schema = config['cache']['schema']
except Exception:
    cache_schema = DEFAULT_SCHEMA


def delete_file(file_path):
    """
    Deletes a file
    Args:
        file_path: Full path of the file

    Returns:
        None
    """
    if file_exists(file_path):
        os.remove(file_path)


def create_symlink_between_paths(old_path, new_path):
    """
    Creates a symlink from new_path to old_path
    Args:
        old_path: Path to the old (real) file
        new_path: Path to the new (symlink) file

    Returns:
        None
    """
    if not file_exists(new_path):
        os.symlink(old_path, new_path)


class VideoDBCachingManager(DBCachingManagerBase):
    def __init__(self, initialize_database=False):
        super().__init__(
            db_config=config['database'],
            cache_table='Video_Main',
            most_similar_table='Video_Most_Similar',
            schema=cache_schema,
            cache_date_modified_col='date_modified',
            initialize_database=initialize_database
        )

    def init_db(self):
        # Ensuring the schema's existence
        self.db.execute_query(
            f"""
            CREATE DATABASE IF NOT EXISTS `{self.schema}`
            DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci
            DEFAULT ENCRYPTION='N';
            """
        )

        # Creating the cache table if it does not exist
        self.db.execute_query(
            f"""
            CREATE TABLE IF NOT EXISTS `{self.schema}`.`{self.cache_table}` (
              `id_token` VARCHAR(255),
              `origin_token` LONGTEXT,
              `fingerprint` VARCHAR(255) DEFAULT NULL,
              `date_added` DATETIME DEFAULT NULL,
              `date_modified` DATETIME DEFAULT NULL,
              PRIMARY KEY id_token (id_token)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
            """
        )

        # Creating an origin_token index (since it's LONGTEXT, we need to specify index length)
        try:
            self.db.execute_query(
                f"""
                CREATE INDEX `video_main_origin_token_index` ON `{self.schema}`.`{self.cache_table}` (`origin_token`(512));
                """
            )
        except Exception:
            pass

        # Creating the closest match table
        self.db.execute_query(
            f"""
            CREATE TABLE IF NOT EXISTS `{self.schema}`.`{self.most_similar_table}` (
              `id_token` VARCHAR(255),
              `most_similar_token` VARCHAR(255) DEFAULT NULL,
              PRIMARY KEY id_token (id_token),
              KEY most_similar_token (most_similar_token)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
            """
        )


class AudioDBCachingManager(DBCachingManagerBase):
    def __init__(self, initialize_database=False):
        super().__init__(
            db_config=config['database'],
            cache_table='Audio_Main',
            most_similar_table='Audio_Most_Similar',
            schema=cache_schema,
            initialize_database=initialize_database
        )

    def init_db(self):
        # Ensuring the schema's existence
        self.db.execute_query(
            f"""
            CREATE DATABASE IF NOT EXISTS `{self.schema}`
            DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci
            DEFAULT ENCRYPTION='N';
            """
        )

        # Creating the cache table if it does not exist
        self.db.execute_query(
            f"""
            CREATE TABLE IF NOT EXISTS `{self.schema}`.`{self.cache_table}` (
              `id_token` VARCHAR(255),
              `origin_token` VARCHAR(255),
              `fingerprint` LONGTEXT DEFAULT NULL,
              `duration` FLOAT,
              `transcript_token` VARCHAR(255) DEFAULT NULL,
              `subtitle_token` VARCHAR(255) DEFAULT NULL,
              `transcript_results` LONGTEXT DEFAULT NULL,
              `subtitle_results` LONGTEXT DEFAULT NULL,
              `nosilence_token` VARCHAR(255) DEFAULT NULL,
              `nosilence_duration` FLOAT DEFAULT NULL,
              `language` VARCHAR(10) DEFAULT NULL,
              `fp_nosilence` INT DEFAULT NULL,
              `date_added` DATETIME DEFAULT NULL,
              PRIMARY KEY id_token (id_token)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
            """
        )

        # Creating an origin_token index
        try:
            self.db.execute_query(
                f"""
                CREATE INDEX `audio_main_origin_token_index` ON `{self.schema}`.`{self.cache_table}` (`origin_token`);
                """
            )
        except Exception:
            pass

        # Creating the closest match table
        self.db.execute_query(
            f"""
            CREATE TABLE IF NOT EXISTS `{self.schema}`.`{self.most_similar_table}` (
              `id_token` VARCHAR(255),
              `most_similar_token` VARCHAR(255) DEFAULT NULL,
              PRIMARY KEY id_token (id_token),
              KEY most_similar_token (most_similar_token)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
            """
        )


class SlideDBCachingManager(DBCachingManagerBase):
    def __init__(self, initialize_database=False):
        super().__init__(
            db_config=config['database'],
            cache_table='Slide_Main',
            most_similar_table='Slide_Most_Similar',
            schema=cache_schema,
            initialize_database=initialize_database
        )

    def init_db(self):
        # Ensuring the schema's existence
        self.db.execute_query(
            f"""
            CREATE DATABASE IF NOT EXISTS `{self.schema}`
            DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci
            DEFAULT ENCRYPTION='N';
            """
        )

        # Creating the cache table if it does not exist
        self.db.execute_query(
            f"""
            CREATE TABLE IF NOT EXISTS `{self.schema}`.`{self.cache_table}` (
              `id_token` VARCHAR(255),
              `origin_token` VARCHAR(255),
              `fingerprint` LONGTEXT DEFAULT NULL,
              `timestamp` FLOAT,
              `slide_number` INT UNSIGNED,
              `ocr_tesseract_token` VARCHAR(255) DEFAULT NULL,
              `ocr_google_1_token` VARCHAR(255) DEFAULT NULL,
              `ocr_google_2_token` VARCHAR(255) DEFAULT NULL,
              `ocr_tesseract_results` LONGTEXT DEFAULT NULL,
              `ocr_google_1_results` LONGTEXT DEFAULT NULL,
              `ocr_google_2_results` LONGTEXT DEFAULT NULL,
              `language` VARCHAR(10) DEFAULT NULL,
              `date_added` DATETIME DEFAULT NULL,
              PRIMARY KEY id_token (id_token)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
            """
        )

        # Creating the fingerprint index if it doesn't exist
        try:
            self.db.execute_query(
                f"""
                CREATE INDEX `slide_main_fp_index` ON `{self.schema}`.`{self.cache_table}` (`fingerprint`(64));
                """
            )
        except Exception:
            pass

        # Creating an origin_token index
        try:
            self.db.execute_query(
                f"""
                CREATE INDEX `slide_main_origin_token_index` ON `{self.schema}`.`{self.cache_table}` (`origin_token`);
                """
            )
        except Exception:
            pass

        # Creating the closest match table
        self.db.execute_query(
            f"""
            CREATE TABLE IF NOT EXISTS `{self.schema}`.`{self.most_similar_table}` (
              `id_token` VARCHAR(255),
              `most_similar_token` VARCHAR(255) DEFAULT NULL,
              PRIMARY KEY id_token (id_token),
              KEY most_similar_token (most_similar_token)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
            """
        )


class TextDBCachingManager(DBCachingManagerBase):
    def __init__(self, initialize_database=False):
        super().__init__(
            db_config=config['database'],
            cache_table='Text_Main',
            most_similar_table='Text_Most_Similar',
            schema=cache_schema,
            initialize_database=initialize_database
        )

    def init_db(self):
        # Making sure the schema exists
        self.db.execute_query(
            f"""
            CREATE DATABASE IF NOT EXISTS `{self.schema}`
            DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci
            DEFAULT ENCRYPTION='N';
            """
        )

        # Creating the cache table if it does not exist
        self.db.execute_query(
            f"""
            CREATE TABLE IF NOT EXISTS `{self.schema}`.`{self.cache_table}` (
              `id_token` VARCHAR(255),
              `fingerprint` VARCHAR(255) DEFAULT NULL,
              `source` LONGTEXT DEFAULT NULL,
              `target` LONGTEXT DEFAULT NULL,
              `source_lang` VARCHAR(10) DEFAULT NULL,
              `target_lang` VARCHAR(10) DEFAULT NULL,
              `date_added` DATETIME DEFAULT NULL,
              PRIMARY KEY id_token (id_token)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
            """
        )

        # Creating the fingerprint index if it doesn't exist
        try:
            self.db.execute_query(
                f"""
                CREATE INDEX `text_main_fp_index` ON `{self.schema}`.`{self.cache_table}` (`fingerprint`(64));
                """
            )
        except Exception:
            pass

        # Creating the closest match table
        self.db.execute_query(
            f"""
            CREATE TABLE IF NOT EXISTS `{self.schema}`.`{self.most_similar_table}` (
              `id_token` VARCHAR(255),
              `most_similar_token` VARCHAR(255) DEFAULT NULL,
              PRIMARY KEY id_token (id_token),
              KEY most_similar_token (most_similar_token)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
            """
        )


class ScrapingDBCachingManager(DBCachingManagerBase):
    def __init__(self, initialize_database=False):
        super().__init__(
            db_config=config['database'],
            cache_table='Scraping_Main',
            most_similar_table='Scraping_Most_Similar',
            schema=cache_schema,
            initialize_database=initialize_database
        )
        # Expiration period in days
        self.expiration_period = 7

    def init_db(self):
        # Making sure the schema exists
        self.db.execute_query(
            f"""
            CREATE DATABASE IF NOT EXISTS `{self.schema}`
            DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci
            DEFAULT ENCRYPTION='N';
            """
        )

        # Creating the cache table if it does not exist
        self.db.execute_query(
            f"""
            CREATE TABLE IF NOT EXISTS `{self.schema}`.`{self.cache_table}` (
              `id_token` VARCHAR(255),
              `origin_token` VARCHAR(255),
              `fingerprint` VARCHAR(255) DEFAULT NULL,
              `link` LONGTEXT,
              `content` LONGTEXT DEFAULT NULL,
              `page_type` VARCHAR(255) DEFAULT NULL,
              `headers_removed` INT DEFAULT NULL,
              `long_patterns_removed` INT DEFAULT NULL,
              `date_added` DATETIME,
              PRIMARY KEY id_token (id_token)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
            """
        )

        # Creating the fingerprint index if it doesn't exist
        try:
            self.db.execute_query(
                f"""
                CREATE INDEX `scraping_main_fp_index` ON `{self.schema}`.`{self.cache_table}` (`fingerprint`);
                """
            )
        except Exception:
            pass

        # Creating the parent_token index if it doesn't exist
        try:
            self.db.execute_query(
                f"""
                CREATE INDEX `scraping_main_origin_index` ON `{self.schema}`.`{self.cache_table}` (`origin_token`);
                """
            )
        except Exception:
            pass

        # Creating the closest match table
        self.db.execute_query(
            f"""
            CREATE TABLE IF NOT EXISTS `{self.schema}`.`{self.most_similar_table}` (
              `id_token` VARCHAR(255),
              `most_similar_token` VARCHAR(255) DEFAULT NULL,
              PRIMARY KEY id_token (id_token),
              KEY most_similar_token (most_similar_token)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
            """
        )

    def get_details_using_origin(self, origin_token, cols):
        """
        Gets details of cache row(s) using origin token instead of id token
        Args:
            origin_token: Origin token
            cols: List of columns to retrieve

        Returns:
            Cache row detail dict
        """
        if 'date_added' not in cols:
            cols = cols + ['date_added']
        results = self._get_details_using_origin(self.cache_table, origin_token, cols, has_date_col=True)
        if results is None:
            return None
        current_time = datetime.now()
        # Only keep the results that are no older than the expiration period
        correct_results = [x for x in results if (current_time - x['date_added']).days < self.expiration_period]
        correct_ids = [x['id_token'] for x in correct_results]
        # See if there are outdated results in the table
        incorrect_ids = [x['id_token'] for x in results if x['id_token'] not in correct_ids]
        # If so, delete their cache rows (since we are not the Internet Archive)
        if len(incorrect_ids) > 0:
            self.delete_cache_rows(incorrect_ids)
        # Return None in case of an empty results list to be consistent with caching logic elsewhere
        if len(correct_results) == 0:
            return None
        return correct_results


class VideoConfig:
    def __init__(self):
        try:
            print("Reading cache storage directory from config")
            self.root_dir = config['cache']['root']
        except Exception:
            print(
                f"The cache storage directory could not be found in the config file, using {ROOT_VIDEO_DIR} as the default. "
                "To use a different one, make sure to add a [cache] section with the root parameter."
            )
            self.root_dir = ROOT_VIDEO_DIR

    def get_root_dir(self):
        return self.root_dir

    def get_image_dir(self):
        return self.concat_file_path('', IMAGE_SUBFOLDER)

    def get_audio_dir(self):
        return self.concat_file_path('', AUDIO_SUBFOLDER)

    def get_video_dir(self):
        return self.concat_file_path('', VIDEO_SUBFOLDER)

    def get_transcript_dir(self):
        return self.concat_file_path('', TRANSCRIPT_SUBFOLDER)

    def concat_file_path(self, filename, subfolder):
        """
        Concatenates the root dir with the given subfolder and file name.
        Args:
            filename: Name of the file
            subfolder: Subfolder it is/should be in

        Returns:
            Full path of the file
        """
        result = os.path.join(self.root_dir, subfolder, filename)
        make_sure_path_exists(result, file_at_the_end=True)
        return result

    def generate_filepath(self, filename, force_dir=None):
        """
        Generates the full path of a given file based on its name
        Args:
            filename: Name of the file
            force_dir: Whether to force a particular subfolder

        Returns:
            The full file path
        """
        if force_dir is not None:
            # If the directory is being forced, we don't check the name of the file to know where it should go
            filename_with_path = self.concat_file_path(filename, force_dir)
        else:
            # If the "file" is really a file or a folder, this will give us the unchanged file name.
            # However, if it's actually in a `folder/file` form, it'll give us the folder, which is how we
            # figure out where it's supposed to go. The full path still involves the full file name,
            # not just the folder part.
            filename_first_part = filename.split('/')[0]
            # If it ends in a video format, it goes into the video subfolder, etc.
            if any([filename_first_part.endswith(x) for x in VIDEO_FORMATS]):
                filename_with_path = self.concat_file_path(filename, VIDEO_SUBFOLDER)
            elif any([filename_first_part.endswith(x) for x in AUDIO_FORMATS]):
                filename_with_path = self.concat_file_path(filename, AUDIO_SUBFOLDER)
            elif any([filename_first_part.endswith(x) for x in IMAGE_FORMATS]):
                filename_with_path = self.concat_file_path(filename, IMAGE_SUBFOLDER)
            elif any([filename_first_part.endswith(x) for x in TRANSCRIPT_FORMATS]):
                filename_with_path = self.concat_file_path(filename, TRANSCRIPT_SUBFOLDER)
            else:
                filename_with_path = self.concat_file_path(filename, OTHER_SUBFOLDER)
        return filename_with_path

    def create_symlink(self, old_path, new_filename):
        """
        Creates a symlink between a new filename (for which the path is generated automatically) and an old full path
        Args:
            old_path: Old full path
            new_filename: New filename

        Returns:
            None
        """
        new_path = self.generate_filepath(new_filename)
        # Only creating the symlink if it doesn't already exist
        create_symlink_between_paths(old_path, new_path)


class FingerprintParameters:
    def __init__(self):
        self.min_similarity = dict()
        self.load_values()

    def load_values(self):
        """
        Loads the values of fingerprinting similarity thresholds from file, or failing that, sets them to defaults
        Returns:
            None
        """
        defaults = {
            'text': '1.0',
            'image': '1.0',
            'audio': '1.0',
            'video': '1.0'
        }
        try:
            print('Reading fingerprint min similarity values from config')
            self.min_similarity['text'] = float(config['fingerprint'].get('text', defaults['text']))
            self.min_similarity['image'] = float(config['fingerprint'].get('image', defaults['image']))
            self.min_similarity['audio'] = float(config['fingerprint'].get('audio', defaults['audio']))
            self.min_similarity['video'] = float(config['fingerprint'].get('video', defaults['video']))
        except Exception:
            self.min_similarity = {k: float(v) for k, v in defaults.items()}

    def get_min_sim_text(self):
        return self.min_similarity['text']

    def get_min_sim_image(self):
        return self.min_similarity['image']

    def get_min_sim_audio(self):
        return self.min_similarity['audio']

    def get_min_sim_video(self):
        return self.min_similarity['video']


def is_fingerprinted(token, db_manager):
    values = db_manager.get_details(token, ['fingerprint'], using_most_similar=False)[0]
    exists = values is not None
    return exists, exists and values['fingerprint'] is not None


def get_token_file_status(token, file_manager):
    if token is None:
        raise Exception("Token is null")
    return os.path.isfile(file_manager.generate_filepath(token))


def get_video_token_status(token):
    video_config = VideoConfig()
    video_db_manager = VideoDBCachingManager()
    try:
        active = get_token_file_status(token, video_config)
    except Exception:
        return None

    exists, has_fingerprint = is_fingerprinted(token, video_db_manager)
    return {
        'active': active and exists,
        'fingerprinted': has_fingerprint
    }


def get_image_token_status(token):
    video_config = VideoConfig()
    image_db_manager = SlideDBCachingManager()
    try:
        active = get_token_file_status(token, video_config)
    except Exception:
        return None

    exists, has_fingerprint = is_fingerprinted(token, image_db_manager)
    return {
        'active': active and exists,
        'fingerprinted': has_fingerprint
    }


def get_audio_token_status(token):
    video_config = VideoConfig()
    audio_db_manager = AudioDBCachingManager()
    try:
        active = get_token_file_status(token, video_config)
    except Exception:
        return None

    exists, has_fingerprint = is_fingerprinted(token, audio_db_manager)
    return {
        'active': active and exists,
        'fingerprinted': has_fingerprint
    }
