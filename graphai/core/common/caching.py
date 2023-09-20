import os
import configparser
import abc
from datetime import datetime

from graphai.core.common.common_utils import make_sure_path_exists, file_exists
from graphai.definitions import CONFIG_DIR
from graphai.core.interfaces.db import DB

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

DEFAULT_SCHEMA = 'cache_graphai'


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


def surround_with_character(s, c="'"):
    """
    Surrounds a string with a character
    Args:
        s: The string
        c: The character

    Returns:
        Resulting string
    """
    return c + s + c


def escape_single_quotes(s):
    """
    Escapes single quotes for SQL queries
    Args:
        s: The original string

    Returns:
        Original string with single quotes escaped
    """
    return s.replace("'", "''")


def add_where_or_and(query):
    """
    Prepares an SQL query for a new condition by adding a WHERE (if the query doesn't already have one)
    or an AND (if it does).
    Args:
        query: The original query

    Returns:
        The query with WHERE or AND added to it
    """
    if 'WHERE' in query:
        return ' AND '
    else:
        return '\nWHERE '


def add_equality_conditions(conditions):
    """
    Generates equality conditions that can be added to a query
    Args:
        conditions: A dictionary where the conditions would be of the form "key=value"

    Returns:
        A string containing the conditions.
    """
    return " AND ".join([f"{k}='{escape_single_quotes(v)}'" for k, v in conditions.items()])


def add_non_null_conditions(cols):
    """
    Generates non-null conditions that can be added to a query
    Args:
        cols: List of columns that cannot be null, which would turn into conditions of the form "col IS NOT NULL"

    Returns:
        A string containing the conditions
    """
    return " AND ".join([col + " IS NOT NULL" for col in cols])


class DBCachingManagerBase(abc.ABC):
    def __init__(self, cache_table, most_similar_table):
        # Only four values are hardcoded into this class and need to be respected by its child classes:
        # 1. The schema, 'cache_graphai', should not be changed
        # 2. The name of the id column for both the main and the most-similar tables is 'id_token'
        # 3. The cache tables must have a "date_added" column of the data type DATETIME,
        #    which has the following format: YYYY-MM-DD hh:mm:ss
        # 4. The name of the second column in the most-similar table is 'most_similar_token'

        config_contents = configparser.ConfigParser()
        try:
            print('Reading cache storage configuration from file')
            config_contents.read(f'{CONFIG_DIR}/cache.ini')
            self.schema = config_contents['CACHE'].get('schema', fallback=DEFAULT_SCHEMA)
        except Exception:
            print(f'Could not read file {CONFIG_DIR}/cache.ini or '
                  f'file does not have section [CACHE] with parameter "schema", '
                  f'falling back to default.')
            self.schema = DEFAULT_SCHEMA
        self.cache_table = cache_table
        self.most_similar_table = most_similar_table
        self.db = DB()
        self.init_db()

    @abc.abstractmethod
    def init_db(self):
        pass

    def _resolve_most_similar_chain(self, token):
        """
        Internal method that resolves the chain of most similar token edges for a given token.
        Args:
            token: The starting token

        Returns:
            The final token of the chain starting from `token`
        """
        if token is None:
            return None
        prev_most_similar = self._resolve_closest_match_edge(token)
        if prev_most_similar is None or prev_most_similar == token:
            return token
        return self._resolve_most_similar_chain(prev_most_similar)

    def _insert_or_update_details(self, schema, table_name, id_token, values_to_insert=None):
        """
        Internal method that inserts a new row or updates an existing row.
        Args:
            schema: The schema of the table
            table_name: Name of the table
            id_token: The id token
            values_to_insert: Dictionary of column names and values to be inserted/updated for `id_token`

        Returns:
            None
        """
        if values_to_insert is None:
            values_to_insert = dict()
        values_to_insert = {
            x: surround_with_character(escape_single_quotes(values_to_insert[x]), "'") if isinstance(values_to_insert[x], str)
            else str(values_to_insert[x]) if values_to_insert[x] is not None
            else 'null'
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
        """
        Internal method that retrieves the details of a given id_token.
        Args:
            schema: Schema name
            table_name: Table name
            id_token: The identifier token
            cols: Columns to retrieve

        Returns:
            A dictionary mapping each column name to its corresponding value for the `id_token` row
        """
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

    def _resolve_closest_match_edge(self, id_token):
        """
        Internal method. Resolves one single edge in the closest match graph
        Args:
            id_token: The starting token

        Returns:
            Closest match of `id_token` if one exists in the corresponding table, None otherwise
        """
        results = self._get_details(self.schema, self.most_similar_table, id_token, ['most_similar_token'])
        if results is not None:
            return results['most_similar_token']
        return None

    def _get_details_using_origin(self, schema, table_name, origin_token, cols, has_date_col=True):
        """
        Internal method that gets details using an origin token (e.g. the video an audio file originated from).
        Args:
            schema: Schema name
            table_name: Table name
            origin_token: Token of the origin file
            cols: Columns to retrieve

        Returns:
            Dictionary mapping column names to values
        """
        column_list = ['origin_token', 'id_token'] + cols
        query = f"""
            SELECT {', '.join(column_list)} FROM `{schema}`.`{table_name}`
            WHERE origin_token={surround_with_character(origin_token, "'")}
            """
        if has_date_col:
            query += '\nORDER BY date_added'
        results = self.db.execute_query(query)
        if len(results) > 0:
            results = [{column_list[i]: result[i] for i in range(len(column_list))} for result in results]
        else:
            results = None
        return results

    def _get_all_details(self, schema, table_name, cols, start=0, limit=-1,
                         exclude_token=None, allow_nulls=True, earliest_date=None, latest_date=None,
                         equality_conditions=None, has_date_col=False):
        """
        Internal method. Gets the details of all rows in a table, with some conditions.
        Args:
            schema: Schema name
            table_name: Table name
            cols: Columns to retrieve
            start: The offset parameter of the LIMIT clause
            limit: the limit parameter of the LIMIT clause
            exclude_token: List of tokens to exclude
            allow_nulls: Whether to allow null values or to exclude rows where any of the required columns is null
            earliest_date: The earliest date to include
            equality_conditions: Equality conditions
            has_date_col: Whether or not the table has a date_added column, which would be used to sort the results.

        Returns:
            Dictionary mapping each id_token to a dictionary of column name : values.
        """
        column_list = ['id_token'] + cols
        query = f"""
            SELECT {', '.join(column_list)} FROM `{schema}`.`{table_name}`
            """
        if exclude_token is not None:
            if isinstance(exclude_token, str):
                query += f"""
                WHERE id_token != {surround_with_character(exclude_token, "'")}
                """
            else:
                query += f"""
                WHERE id_token NOT IN ({','.join([surround_with_character(t, "'") for t in exclude_token])})
                """
        if not allow_nulls:
            query += add_where_or_and(query)
            query += add_non_null_conditions(cols)
        if earliest_date is not None and has_date_col:
            query += add_where_or_and(query)
            query += f" date_added >= '{earliest_date}'"
        if latest_date is not None and has_date_col:
            query += add_where_or_and(query)
            query += f" date_added <= '{latest_date}'"
        if equality_conditions is not None:
            query += add_where_or_and(query)
            query += add_equality_conditions(equality_conditions)
        # ORDER BY comes before LIMIT but after WHERE
        if has_date_col:
            query += "\nORDER BY date_added"
        else:
            query += "\nORDER BY id_token"
        if limit != -1:
            query += f"""
            LIMIT {start},{limit}
            """
        results = self.db.execute_query(query)
        if len(results) > 0:
            results = {row[0]: {column_list[i]: row[i] for i in range(len(column_list))} for row in results}
        else:
            results = None
        return results

    def _delete_rows(self, schema, table_name, id_tokens):
        """
        Internal method. Deletes rows using a list of ids to delete
        Args:
            schema: Schema name
            table_name: Table name
            id_tokens: List of ids to delete

        Returns:
            None
        """
        id_tokens_str = '(' + ', '.join([surround_with_character(id_token, "'") for id_token in id_tokens]) + ')'
        self.db.execute_query(
            f"""
            DELETE FROM `{schema}`.`{table_name}` WHERE id_token IN {id_tokens_str}
            """
        )

    def _get_count(self, schema, table_name, non_null_cols=None, equality_conditions=None):
        """
        Internal method. Gets the number of rows in a table, possibly with conditions on the rows.
        Args:
            schema: Schema name
            table_name: Table name
            non_null_cols: List of columns that have a non-null condition
            equality_conditions: Dictionary of equality conditions

        Returns:
            Number of rows with the given conditions
        """
        query = f"""
        SELECT COUNT(*) FROM `{schema}`.`{table_name}`
        """
        if non_null_cols is not None:
            query += f"""
            WHERE {add_non_null_conditions(non_null_cols)}
            """
        if equality_conditions is not None:
            query += add_where_or_and(query)
            query += add_equality_conditions(equality_conditions)
        results = self.db.execute_query(query)
        return results[0][0]

    def _row_exists(self, schema, table_name, id_token):
        """
        Internal method. Checks whether a row with a given id token exists.
        Args:
            schema: Schema name
            table_name: Table name
            id_token: Identifier token of the row

        Returns:
            True if the row exists, False otherwise
        """
        query = f"""
        SELECT COUNT(*) FROM `{schema}`.`{table_name}`
        WHERE id_token = '{id_token}'
        """
        results = self.db.execute_query(query)
        if len(results) > 0:
            return True
        return False

    def delete_cache_rows(self, id_tokens):
        """
        Deletes rows from the cache table
        Args:
            id_tokens: List of id tokens to delete the rows of

        Returns:
            None
        """
        self._delete_rows(self.schema, self.cache_table, id_tokens)

    def insert_or_update_details(self, id_token, values_to_insert=None):
        """
        Inserts or updates values in the cache table
        Args:
            id_token: Identifier token
            values_to_insert: Dictionary of column to value mappings

        Returns:
            None
        """
        self._insert_or_update_details(self.schema, self.cache_table, id_token, values_to_insert)

    def update_details_if_exists(self, id_token, values_to_insert):
        """
        Only updates values if the row exists and does not insert otherwise
        Args:
            id_token: Identifier token
            values_to_insert: Column to value dict

        Returns:
            None
        """
        if not self._row_exists(self.schema, self.cache_table, id_token):
            return
        self._insert_or_update_details(self.schema, self.cache_table, id_token, values_to_insert)

    def get_details(self, id_token, cols, using_most_similar=False):
        """
        Gets details from the cache table for a given id token.
        Args:
            id_token: Identifier token
            cols: Columns to retrieve
            using_most_similar: Whether to resolve the most similar chain or not

        Returns:
            A list of two results: the token's own results and the results of the token's closest match. If there
            is no closest match, or the closest match is the token itself, or using_most_similar is False, then the
            second result is None.
        """
        own_results = self._get_details(self.schema, self.cache_table, id_token, cols)
        if not using_most_similar:
            return [own_results, None]
        closest_token = self.get_closest_match(id_token)
        if closest_token is None or closest_token == id_token:
            return [own_results, None]
        closest_match_results = self._get_details(self.schema, self.cache_table, closest_token, cols)
        return [own_results, closest_match_results]

    def get_origin(self, id_token):
        """
        Gets the origin of a given id token
        Args:
            id_token: Identifier token

        Returns:
            Origin token if applicable
        """
        results = self._get_details(self.schema, self.cache_table, id_token, ['origin_token'])
        if results is not None:
            return results['origin_token']
        return None

    def get_details_using_origin(self, origin_token, cols):
        """
        Gets details of cache row(s) using origin token instead of id token
        Args:
            origin_token: Origin token
            cols: List of columns to retrieve

        Returns:
            Cache row detail dict
        """
        return self._get_details_using_origin(self.schema, self.cache_table, origin_token, cols, has_date_col=True)

    def get_all_details(self, cols, start=0, limit=-1, exclude_token=None,
                        allow_nulls=True, earliest_date=None, latest_date=None, equality_conditions=None):
        """
        Gets details of all rows in cache table, possibly with constraints
        Args:
            cols: Columns to retrieve
            start: Offset of LIMIT clause
            limit: count of LIMIT clause
            exclude_token: List of tokens to exclude
            allow_nulls: Whether to allow null values for requested cols
            earliest_date: Earliest date to allow
            latest_date: Latest date to allow
            equality_conditions: Dict of equality conditions

        Returns:
            Dict mapping id tokens to colname->value dicts
        """
        # If we want to exclude a token, all the tokens whose closest match is the former should also be excluded
        # in order not to create cycles.
        if exclude_token is not None:
            all_closest_matches = self.get_all_closest_matches()
            if all_closest_matches is not None:
                exclude_tokens = {k for k, v in all_closest_matches.items() if v == exclude_token}
            else:
                exclude_tokens = set()
            exclude_tokens.add(exclude_token)
        else:
            exclude_tokens = None
        results = self._get_all_details(self.schema, self.cache_table, cols,
                                        start=start, limit=limit, exclude_token=exclude_tokens,
                                        allow_nulls=allow_nulls, earliest_date=earliest_date, latest_date=latest_date,
                                        equality_conditions=equality_conditions, has_date_col=True)
        return results

    def get_cache_count(self, non_null_cols=None, equality_conditions=None):
        """
        Gets number of rows in cache table, possibly with constraints
        Args:
            non_null_cols: Columns to enforce a non-null constraint on
            equality_conditions: Equality conditions dict

        Returns:
            Number of rows that satisfy the given conditions from the cache table
        """
        return self._get_count(self.schema, self.cache_table, non_null_cols, equality_conditions)

    def insert_or_update_closest_match(self, id_token, values_to_insert):
        """
        Inserts or updates the value of a row in the closest match table
        Args:
            id_token: Identifier token
            values_to_insert: Dict of values to insert

        Returns:
            None
        """
        self._insert_or_update_details(self.schema, self.most_similar_table, id_token, values_to_insert)

    def get_closest_match(self, id_token):
        """
        Gets closest match of given token by resolving the closest match chain.
        Args:
            id_token: Starting identifier token

        Returns:
            Final id token in the chain
        """
        return self._resolve_most_similar_chain(id_token)

    def get_all_closest_matches(self):
        """
        Retrieves all the rows in the closest match table
        Returns:
            All rows in most similar token table
        """
        results = self._get_all_details(self.schema, self.most_similar_table,
                                        ['most_similar_token'], has_date_col=False)
        if results is not None:
            return {x: results[x]['most_similar_token'] for x in results
                    if results[x]['most_similar_token'] is not None}
        return None


class VideoDBCachingManager(DBCachingManagerBase):
    def __init__(self):
        super().__init__(cache_table='Video_Main', most_similar_table='Video_Most_Similar')

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
    def __init__(self):
        super().__init__(cache_table='Audio_Main', most_similar_table='Audio_Most_Similar')

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
    def __init__(self):
        super().__init__(cache_table='Slide_Main', most_similar_table='Slide_Most_Similar')

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
    def __init__(self):
        super().__init__(cache_table='Text_Main', most_similar_table='Text_Most_Similar')

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


class SummaryDBCachingManager(DBCachingManagerBase):
    def __init__(self):
        super().__init__(cache_table='Summary_Main', most_similar_table='Summary_Most_Similar')

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
              `input_text` LONGTEXT DEFAULT NULL,
              `input_type` VARCHAR(255) DEFAULT NULL,
              `summary` LONGTEXT DEFAULT NULL,
              `summary_type` VARCHAR(10) DEFAULT NULL,
              `summary_len_class` VARCHAR(10) DEFAULT NULL,
              `summary_tone` VARCHAR(10) DEFAULT NULL,
              `summary_length` INT DEFAULT NULL,
              `summary_token_total` INT DEFAULT NULL,
              `date_added` DATETIME DEFAULT NULL,
              PRIMARY KEY id_token (id_token)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
            """
        )

        # Creating the fingerprint index if it doesn't exist
        try:
            self.db.execute_query(
                f"""
                CREATE INDEX `summary_main_fp_index` ON `{self.schema}`.`{self.cache_table}` (`fingerprint`(64));
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
    def __init__(self):
        super().__init__(cache_table='Scraping_Main', most_similar_table='Scraping_Most_Similar')
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
        results = self._get_details_using_origin(self.schema, self.cache_table, origin_token, cols, has_date_col=True)
        if results is None:
            return results
        current_time = datetime.now()
        # Only keep the results that are no older than the expiration period
        results = [x for x in results if (current_time - x['date_added']).days < self.expiration_period]
        return results


class VideoConfig():
    def __init__(self):
        config_contents = configparser.ConfigParser()
        try:
            print('Reading cache storage configuration from file')
            config_contents.read(f'{CONFIG_DIR}/cache.ini')
            self.root_dir = config_contents['CACHE'].get('root', fallback=ROOT_VIDEO_DIR)
        except Exception:
            print(f'Could not read file {CONFIG_DIR}/cache.ini or '
                  f'file does not have section [CACHE], falling back to defaults.')
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
