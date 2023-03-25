import os
import errno
from graphai.core.interfaces.db import DB

ROOT_VIDEO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../Storage/'))
# Formats with a . in their name indicate single files, whereas formats without a . indicate folders (e.g. '_slides')
VIDEO_SUBFOLDER = 'Video'
VIDEO_FORMATS = ['.mkv', '.mp4', '.avi', '.mov', '.flv']
AUDIO_SUBFOLDER = 'Audio'
AUDIO_FORMATS = ['.mp3', '.flac', '.wav', '.aac']
IMAGE_SUBFOLDER = 'Image'
IMAGE_FORMATS = ['.png', '.tiff', '.jpg', '.jpeg', '.bmp', '_slides']
OTHER_SUBFOLDER = 'Other'
SIGNATURE_SUBFOLDER = 'Signatures'
SIGNATURE_FORMATS = ['_sig.xml']
TEMP_SUBFOLDER = 'Temp'

SLIDE_OUTPUT_FORMAT = '_slides/slide_%05d.png'

STANDARD_FPS = 30


def make_sure_path_exists(path, file_at_the_end=False):
    try:
        if not file_at_the_end:
            os.makedirs(path)
        else:
            os.makedirs('/'.join(path.split('/')[:-1]))
        return
    except OSError as exception:
        if exception.errno != errno.EEXIST and exception.errno != errno.EPERM:
            raise


def file_exists(file_path):
    return os.path.exists(file_path)


def get_dir_files(root_dir):
    return [sub_path for sub_path in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, sub_path))]


def count_files(root_dir):
    count = 0
    for sub_path in os.listdir(root_dir):
        if os.path.isfile(os.path.join(root_dir, sub_path)):
            count += 1
    return count


class VideoConfig():
    def __init__(self, root_dir=ROOT_VIDEO_DIR):
        self.root_dir = root_dir


    def concat_file_path(self, filename, subfolder):
        result = os.path.join(self.root_dir, subfolder, filename)
        make_sure_path_exists(result, file_at_the_end=True)
        return result


    def set_root_dir(self, new_root_dir):
        self.root_dir = new_root_dir
        make_sure_path_exists(new_root_dir)

    def generate_filename(self, filename, force_dir=None):
        if force_dir is not None:
            filename_with_path = self.concat_file_path(filename, force_dir)
        elif any([filename.endswith(x) for x in VIDEO_FORMATS]):
            filename_with_path = self.concat_file_path(filename, VIDEO_SUBFOLDER)
        elif any([filename.endswith(x) for x in AUDIO_FORMATS]):
            filename_with_path = self.concat_file_path(filename, AUDIO_SUBFOLDER)
        elif any([filename.endswith(x) for x in SIGNATURE_FORMATS]):
            filename_with_path = self.concat_file_path(filename, SIGNATURE_SUBFOLDER)
        elif any([filename.endswith(x) for x in IMAGE_FORMATS]):
            filename_with_path = self.concat_file_path(filename, IMAGE_SUBFOLDER)
        else:
            filename_with_path = self.concat_file_path(filename, OTHER_SUBFOLDER)
        return filename_with_path


def surround_with_character(s, c="'"):
    return c + s + c


class DBCachingManager():
    def __init__(self):
        self.schema = 'cache_graphai'
        self.audio_cache_table = 'Audio_Main'
        self.audio_most_similar_table = 'Audio_Most_Similar'
        self.db = DB()

    def init_db(self):
        self.db.execute_query(
            f"""
            CREATE DATABASE IF NOT EXISTS `{self.schema}` 
            DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci 
            DEFAULT ENCRYPTION='N';
            """
        )
        self.db.execute_query(
            f"""
            CREATE TABLE IF NOT EXISTS `{self.schema}`.`{self.audio_cache_table}` (
              `id_token` VARCHAR(255),
              `origin_token` VARCHAR(255),
              `fingerprint` LONGTEXT DEFAULT NULL,
              `duration` FLOAT,
              `transcript_token` VARCHAR(255) DEFAULT NULL,
              `nosilence_token` VARCHAR(255) DEFAULT NULL,
              `nosilence_duration` FLOAT DEFAULT NULL,
              `fp_nosilence` INT DEFAULT NULL,
              PRIMARY KEY id_token (id_token)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
            """
        )
        self.db.execute_query(
            f"""
            CREATE TABLE IF NOT EXISTS `{self.schema}`.`{self.audio_most_similar_table}` (
              `id_token` VARCHAR(255),
              `most_similar_token` VARCHAR(255) DEFAULT NULL,
              PRIMARY KEY id_token (id_token),
              KEY most_similar_token (most_similar_token)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
            """
        )


    def _insert_or_update_details(self, schema, table_name, id_token, values_to_insert=None):
        if values_to_insert is None:
            values_to_insert = dict()
        values_to_insert = {
            x: surround_with_character(values_to_insert[x], "'") if isinstance(values_to_insert[x], str)
            else str(values_to_insert[x])
            for x in values_to_insert
        }
        existing = self.db.execute_query(
            f"""
            SELECT COUNT(*) FROM `{schema}`.`{table_name}`
            WHERE id_token={surround_with_character(id_token, "'")}
            """
        )[0][0]
        if existing > 0:
            cols = [surround_with_character(x, "`") for x in values_to_insert.keys()]
            values = list(values_to_insert.values())
            cols_and_values = [cols[i] + ' = ' + values[i] for i in range(len(cols))]
            self.db.execute_query(
                f"""
                UPDATE `{schema}`.`{table_name}`
                SET
                {', '.join(cols_and_values)}
                WHERE id_token={surround_with_character(id_token, "'")};
                """
            )
        else:
            cols = ['id_token'] + [x for x in values_to_insert.keys()]
            cols = [surround_with_character(x, "`") for x in cols]
            values = [surround_with_character(id_token, "'")] + list(values_to_insert.values())

            self.db.execute_query(
                f"""
                INSERT INTO `{schema}`.`{table_name}`
                    ({', '.join(cols)})
                    VALUES
                    ({', '.join(values)});
                """
            )

    def insert_or_update_audio_details(self, id_token, values_to_insert=None):
        self._insert_or_update_details(self.schema, self.audio_cache_table, id_token, values_to_insert)

    def get_audio_details(self, id_token, cols):
        column_list = ['id_token'] + cols
        results = self.db.execute_query(
            f"""
            SELECT {', '.join(column_list)} FROM `{self.schema}`.`{self.audio_cache_table}`
            WHERE id_token={surround_with_character(id_token, "'")}
            """
        )
        if len(results) > 0:
            results = {column_list[i]: results[0][i] for i in range(len(column_list))}
        else:
            results = None
        return results

    def get_all_audio_details(self, cols):
        column_list = ['id_token'] + cols
        results = self.db.execute_query(
            f"""
            SELECT {', '.join(column_list)} FROM `{self.schema}`.`{self.audio_cache_table}`
            """
        )
        if len(results) > 0:
            results = {row[0]: {column_list[i]: row[i] for i in range(len(column_list))} for row in results}
        else:
            results = None
        return results