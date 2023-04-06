import os

from .video import ROOT_VIDEO_DIR, make_sure_path_exists, VIDEO_FORMATS, VIDEO_SUBFOLDER, AUDIO_FORMATS, \
    AUDIO_SUBFOLDER, SIGNATURE_FORMATS, SIGNATURE_SUBFOLDER, IMAGE_FORMATS, IMAGE_SUBFOLDER, TRANSCRIPT_FORMATS, \
    TRANSCRIPT_SUBFOLDER, OTHER_SUBFOLDER
from ..interfaces.db import DB
import abc


def surround_with_character(s, c="'"):
    return c + s + c


class DBCachingManagerBase(abc.ABC):
    def __init__(self, cache_table, most_similar_table):
        # Only three values are hardcoded into this class and need to be respected by its child classes:
        # 1. The schema, 'cache_graphai', should not be changed
        # 2. The name of the id column for both the main and the most-similar tables is 'id_token'
        # 3. The name of the second column in the most-similar table is 'most_similar_token'
        self.schema = 'cache_graphai'
        self.cache_table = cache_table
        self.most_similar_table = most_similar_table
        self.db = DB()
        self.init_db()

    @abc.abstractmethod
    def init_db(self):
        pass

    def resolve_most_similar_chain(self, token):
        if token is None:
            return None
        prev_most_similar = self.get_closest_match(token)
        if prev_most_similar is None or prev_most_similar==token:
            return token
        return self.resolve_most_similar_chain(prev_most_similar)


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

    def _get_details(self, schema, table_name, id_token, cols):
        column_list = ['id_token'] + cols
        results = self.db.execute_query(
            f"""
            SELECT {', '.join(column_list)} FROM `{schema}`.`{table_name}`
            WHERE id_token={surround_with_character(id_token, "'")}
            """
        )
        if len(results) > 0:
            results = {column_list[i]: results[0][i] for i in range(len(column_list))}
        else:
            results = None
        return results

    def _get_all_details(self, schema, table_name, cols):
        column_list = ['id_token'] + cols
        results = self.db.execute_query(
            f"""
            SELECT {', '.join(column_list)} FROM `{schema}`.`{table_name}`
            """
        )
        if len(results) > 0:
            results = {row[0]: {column_list[i]: row[i] for i in range(len(column_list))} for row in results}
        else:
            results = None
        return results

    def insert_or_update_details(self, id_token, values_to_insert=None):
        self._insert_or_update_details(self.schema, self.cache_table, id_token, values_to_insert)

    def get_details(self, id_token, cols, using_most_similar=False):
        if using_most_similar:
            print(id_token)
            id_token_to_use = self.get_closest_match(id_token)
            if id_token_to_use is None:
                id_token_to_use = id_token
        else:
            id_token_to_use = id_token
        return self._get_details(self.schema, self.cache_table, id_token_to_use, cols)

    def get_all_details(self, cols, using_most_similar=False):
        results = self._get_all_details(self.schema, self.cache_table, cols)
        if using_most_similar:
            most_similar_map = self.get_all_closest_matches()
            results = {x: results[most_similar_map.get(x, x)] for x in results}
        return results

    def insert_or_update_closest_match(self, id_token, values_to_insert):
        self._insert_or_update_details(self.schema, self.most_similar_table, id_token, values_to_insert)

    def get_closest_match(self, id_token):
        results = self._get_details(self.schema, self.most_similar_table, id_token, ['most_similar_token'])
        if results is not None:
            return results['most_similar_token']
        return None

    def get_all_closest_matches(self):
        results = self._get_all_details(self.schema, self.most_similar_table, ['most_similar_token'])
        if results is not None:
            return {x: results[x]['most_similar_token'] for x in results
                    if results[x]['most_similar_token'] is not None}
        return None


class AudioDBCachingManager(DBCachingManagerBase):
    def __init__(self):
        super().__init__(cache_table='Audio_Main', most_similar_table='Audio_Most_Similar')

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
            CREATE TABLE IF NOT EXISTS `{self.schema}`.`{self.cache_table}` (
              `id_token` VARCHAR(255),
              `origin_token` VARCHAR(255),
              `fingerprint` LONGTEXT DEFAULT NULL,
              `duration` FLOAT,
              `transcript_token` VARCHAR(255) DEFAULT NULL,
              `subtitle_token` VARCHAR(255) DEFAULT NULL,
              `nosilence_token` VARCHAR(255) DEFAULT NULL,
              `nosilence_duration` FLOAT DEFAULT NULL,
              `language` VARCHAR(10) DEFAULT NULL,
              `fp_nosilence` INT DEFAULT NULL,
              PRIMARY KEY id_token (id_token)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
            """
        )
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
        elif any([filename.endswith(x) for x in TRANSCRIPT_FORMATS]):
            filename_with_path = self.concat_file_path(filename, TRANSCRIPT_SUBFOLDER)
        else:
            filename_with_path = self.concat_file_path(filename, OTHER_SUBFOLDER)
        return filename_with_path