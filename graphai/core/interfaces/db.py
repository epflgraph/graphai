import abc
import sys

import pandas as pd

import mysql.connector


def quote_value(v):
    if isinstance(v, str):
        return f'"{v}"'
    else:
        return v


class DB:
    """
    Base class to communicate with the EPFLGraph database.
    """

    def __init__(self, db_config):
        # Read db config from file and open connection
        # db_config is a dictionary containing four keys: "host", "port", "user", and "pass"
        # All except port (which is an int) are strings

        self.host = db_config['host']
        self.port = db_config['port']
        self.user = db_config['user']
        self.password = db_config['pass']

        self.cnx = mysql.connector.connect(host=self.host, port=self.port, user=self.user, password=self.password)

    def __del__(self):
        if hasattr(self, 'cnx'):
            self.cnx.close()

    ################
    # BASE METHODS #
    ################

    def execute_query(self, query, values=None):
        """
        Execute custom query.
        """

        # Refresh connection
        self.cnx.ping(reconnect=True)
        cursor = self.cnx.cursor()

        try:
            if values:
                cursor.execute(query, values)
            else:
                cursor.execute(query)
        except mysql.connector.Error as e:
            print("Error", e)
            raise e

        results = list(cursor)

        self.cnx.commit()

        return results

    def build_conditions_list(self, conditions=None, values=None):
        if conditions is None:
            return []

        if values is None:
            values = []

        conditions_list = []
        for key in conditions:
            if isinstance(conditions[key], dict):
                if key == 'NOT':
                    subconditions_list, subvalues = self.build_conditions_list(conditions[key])
                    conditions_list.append('NOT (' + ' AND '.join(subconditions_list) + ')')
                    values.extend(subvalues)
                elif key in ['AND', 'OR']:
                    subconditions_list, subvalues = self.build_conditions_list(conditions[key])
                    conditions_list.append('(' + f' {key} '.join(subconditions_list) + ')')
                    values.extend(subvalues)
                else:
                    for operator in conditions[key]:
                        conditions_list.append(f'{key} {operator} {quote_value(conditions[key][operator])}')
            elif isinstance(conditions[key], list):
                conditions_list.append(f'{key} IN ({", ".join(["%s"] * len(conditions[key]))})')
                values.extend(conditions[key])
            else:
                conditions_list.append(f'{key} = {quote_value(conditions[key])}')

        return conditions_list, values

    def find(self, table_name, fields=None, conditions=None, print_query=False):
        if fields:
            fields_str = ', '.join(fields)
        else:
            fields_str = '*'

        conditions_str = ''
        values = []
        if conditions:
            conditions_list, values = self.build_conditions_list(conditions)
            conditions_str = 'WHERE ' + ' AND '.join(conditions_list)

        query = f"""
            SELECT {fields_str}
            FROM {table_name}
            {conditions_str}
        """

        if print_query:
            print(query)

        return self.execute_query(query, values)

    def find_or_split(self, table_name, fields, columns, filter_field, filter_ids):
        try:
            conditions = {filter_field: filter_ids} if filter_ids else {}
            return pd.DataFrame(self.find(table_name, fields=fields, conditions=conditions), columns=columns)
        except mysql.connector.Error:
            n = len(filter_ids)
            print(f'Failed fetching df filtering {n} ids. Splitting in two and retrying...')
            df1 = self.find_or_split(table_name, fields, columns, filter_field, filter_ids[: n // 2])
            df2 = self.find_or_split(table_name, fields, columns, filter_field, filter_ids[n // 2:])
            return pd.concat([df1, df2]).reset_index(drop=True)

    def drop_table(self, table_name):
        query = f"""
            DROP TABLE IF EXISTS {table_name};
        """
        self.execute_query(query)

    def create_table(self, table_name, definition):
        query = f"""
            CREATE TABLE {table_name} (
                {', '.join([line for line in definition])}
            ) ENGINE=InnoDB DEFAULT CHARSET ascii;
        """
        self.execute_query(query)

    def insert_dataframe(self, table_name, df):
        tuples = list(df.itertuples(index=False, name=None))
        values = [value for line in tuples for value in line]

        # placeholder for the row of values, e.g. "(%s, %s, %s)"
        placeholder = f'({", ".join(["%s"] * len(df.columns))})'

        query = f"""INSERT INTO {table_name} VALUES {', '.join([placeholder] * len(tuples))}"""

        try:
            self.execute_query(query, values)
        except mysql.connector.Error as e:
            handled_error_codes = [
                mysql.connector.errorcode.CR_SERVER_LOST_EXTENDED,  # Broken pipe error (connection closed by server)
                mysql.connector.errorcode.ER_NET_PACKET_TOO_LARGE   # Packet bigger than max_allowed_packet
            ]

            if e.errno in handled_error_codes:
                n = len(df)
                payload_size_bytes = sys.getsizeof(query)
                print(f'Failed inserting df with {n} rows (query payload size: {payload_size_bytes / (2**20) :.2f} MB). Splitting in two and retrying...')

                df1 = df.iloc[:(n // 2)]
                df2 = df.iloc[(n // 2):]
                self.insert_dataframe(table_name, df1)
                self.insert_dataframe(table_name, df2)
            else:
                raise e

    def drop_create_insert_table(self, table_name, definition, df):
        self.drop_table(table_name)
        self.create_table(table_name, definition)
        self.insert_dataframe(table_name, df)

    def check_if_table_exists(self, schema, table_name):
        query = f"""
        SELECT COUNT(TABLE_NAME)
        FROM
           information_schema.TABLES
        WHERE
           TABLE_SCHEMA LIKE '{schema}' AND
           TABLE_NAME = '{table_name}';
        """
        res = self.execute_query(query)
        if res[0][0] > 0:
            return True
        return False


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


def escape_backslashes(s):
    """
    Escapes backslashes for SQL queries
    Args:
        s: Input string

    Returns:
    Original string with backslashes escaped
    """
    return s.replace("\\", "\\\\")


def escape_everything(s):
    """
    Escapes both single quotes and backslashes
    Args:
        s: Input string

    Returns:
    Original string with everything escaped
    """
    return escape_backslashes(escape_single_quotes(s))


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
    return " AND ".join([f"{k}='{escape_everything(v)}'" for k, v in conditions.items()])


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
    def __init__(self, db_config, cache_table, most_similar_table, schema='cache_graphai',
                 cache_date_added_col='date_added',
                 cache_date_modified_col='date_added'):
        # Only three values are hardcoded into this class and need to be respected by its child classes:
        # 1. The name of the id column for both the main and the most-similar tables is 'id_token'
        # 2. The cache tables must have a "date_added" column of the data type DATETIME,
        #    which has the following format: YYYY-MM-DD hh:mm:ss
        # 3. The name of the second column in the most-similar table is 'most_similar_token'
        self.schema = schema
        self.cache_table = cache_table
        self.most_similar_table = most_similar_table
        self.cache_date_added_col = cache_date_added_col
        self.cache_date_modified_col = cache_date_modified_col
        self.db = DB(db_config)
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

    def _insert_or_update_details(self, table_name, id_token, values_to_insert=None):
        """
        Internal method that inserts a new row or updates an existing row.
        Args:
            table_name: Name of the table
            id_token: The id token
            values_to_insert: Dictionary of column names and values to be inserted/updated for `id_token`

        Returns:
            None
        """
        if values_to_insert is None:
            values_to_insert = dict()
        values_to_insert = {
            x: surround_with_character(escape_everything(values_to_insert[x]), "'")
            if isinstance(values_to_insert[x], str)
            else str(values_to_insert[x]) if values_to_insert[x] is not None
            else 'null'
            for x in values_to_insert
        }
        existing = self.db.execute_query(
            f"""
            SELECT COUNT(*) FROM `{self.schema}`.`{table_name}`
            WHERE id_token={surround_with_character(id_token, "'")}
            """
        )[0][0]
        if existing > 0:
            cols = [surround_with_character(x, "`") for x in values_to_insert.keys()]
            values = list(values_to_insert.values())
            cols_and_values = [cols[i] + ' = ' + values[i] for i in range(len(cols))]
            self.db.execute_query(
                f"""
                UPDATE `{self.schema}`.`{table_name}`
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
                INSERT INTO `{self.schema}`.`{table_name}`
                    ({', '.join(cols)})
                    VALUES
                    ({', '.join(values)});
                """
            )

    def _get_details(self, table_name, id_token, cols):
        """
        Internal method that retrieves the details of a given id_token.
        Args:
            table_name: Table name
            id_token: The identifier token
            cols: Columns to retrieve

        Returns:
            A dictionary mapping each column name to its corresponding value for the `id_token` row
        """
        column_list = ['id_token'] + cols
        results = self.db.execute_query(
            f"""
            SELECT {', '.join(column_list)} FROM `{self.schema}`.`{table_name}`
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
        results = self._get_details(self.most_similar_table, id_token, ['most_similar_token'])
        if results is not None:
            return results['most_similar_token']
        return None

    def _get_details_using_origin(self, table_name, origin_token, cols, has_date_col=True):
        """
        Internal method that gets details using an origin token (e.g. the video an audio file originated from).
        Args:
            table_name: Table name
            origin_token: Token of the origin file
            cols: Columns to retrieve

        Returns:
            Dictionary mapping column names to values
        """
        column_list = ['origin_token', 'id_token'] + cols
        query = f"""
            SELECT {', '.join(column_list)} FROM `{self.schema}`.`{table_name}`
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

    def _get_all_details(self, table_name, cols, start=0, limit=-1, exclude_token=None, allow_nulls=True,
                         earliest_date=None, latest_date=None, equality_conditions=None, has_date_col=False,
                         sort_by_date_col=True, use_date_modified_col=False):
        """
        Internal method. Gets the details of all rows in a table, with some conditions.
        Args:
            table_name: Table name
            cols: Columns to retrieve
            start: The offset parameter of the LIMIT clause
            limit: the limit parameter of the LIMIT clause
            exclude_token: List of tokens to exclude
            allow_nulls: Whether to allow null values or to exclude rows where any of the required columns is null
            earliest_date: The earliest date to include
            equality_conditions: Equality conditions
            has_date_col: Whether the table has a date_added column, which would be used to sort the results.
            use_date_modified_col: Whether to use the date_modified or the date_added column for comparisons
                with `earliest_date` or `latest_date`.
        Returns:
            Dictionary mapping each id_token to a dictionary of column name : values.
        """
        column_list = ['id_token'] + cols
        query = f"""
            SELECT {', '.join(column_list)} FROM `{self.schema}`.`{table_name}`
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
        if has_date_col:
            date_col_to_use_for_comp = \
                self.cache_date_modified_col if use_date_modified_col else self.cache_date_added_col
            if earliest_date is not None:
                query += add_where_or_and(query)
                query += f" {date_col_to_use_for_comp} >= '{earliest_date}'"
            if latest_date is not None:
                query += add_where_or_and(query)
                query += f" {date_col_to_use_for_comp} <= '{latest_date}'"
        if equality_conditions is not None:
            query += add_where_or_and(query)
            query += add_equality_conditions(equality_conditions)
        # ORDER BY comes before LIMIT but after WHERE
        if has_date_col and sort_by_date_col:
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

    def _delete_rows(self, table_name, id_tokens):
        """
        Internal method. Deletes rows using a list of ids to delete
        Args:
            table_name: Table name
            id_tokens: List of ids to delete

        Returns:
            None
        """
        if id_tokens is None or len(id_tokens) == 0:
            return
        id_tokens_str = '(' + ', '.join([surround_with_character(id_token, "'") for id_token in id_tokens]) + ')'
        self.db.execute_query(
            f"""
            DELETE FROM `{self.schema}`.`{table_name}` WHERE id_token IN {id_tokens_str}
            """
        )

    def _get_count(self, table_name, non_null_cols=None, equality_conditions=None):
        """
        Internal method. Gets the number of rows in a table, possibly with conditions on the rows.
        Args:
            table_name: Table name
            non_null_cols: List of columns that have a non-null condition
            equality_conditions: Dictionary of equality conditions

        Returns:
            Number of rows with the given conditions
        """
        query = f"""
        SELECT COUNT(*) FROM `{self.schema}`.`{table_name}`
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

    def _row_exists(self, table_name, id_token):
        """
        Internal method. Checks whether a row with a given id token exists.
        Args:
            table_name: Table name
            id_token: Identifier token of the row

        Returns:
            True if the row exists, False otherwise
        """
        query = f"""
        SELECT COUNT(*) FROM `{self.schema}`.`{table_name}`
        WHERE id_token = '{id_token}'
        """
        results = self.db.execute_query(query)
        if len(results) > 0:
            return True
        return False

    def add_columns(self, table_name, column_names, column_types, defaults=None):
        if defaults is None:
            defaults = ["NULL"] * len(column_names)
        for i in range(len(column_names)):
            query = f"""
            ALTER TABLE `{self.schema}`.`{table_name}`
            ADD {column_names[i]} {column_types[i]} DEFAULT {defaults[i]};
            """
            self.db.execute_query(query)

    def remove_columns(self, table_name, column_names):
        for i in range(len(column_names)):
            query = f"""
            ALTER TABLE `{self.schema}`.`{table_name}`
            DROP COLUMN {column_names[i]};
            """
            self.db.execute_query(query)

    def delete_cache_rows(self, id_tokens):
        """
        Deletes rows from the cache table
        Args:
            id_tokens: List of id tokens to delete the rows of

        Returns:
            None
        """
        self._delete_rows(self.cache_table, id_tokens)

    def insert_or_update_details(self, id_token, values_to_insert=None):
        """
        Inserts or updates values in the cache table
        Args:
            id_token: Identifier token
            values_to_insert: Dictionary of column to value mappings

        Returns:
            None
        """
        self._insert_or_update_details(self.cache_table, id_token, values_to_insert)

    def update_details_if_exists(self, id_token, values_to_insert):
        """
        Only updates values if the row exists and does not insert otherwise
        Args:
            id_token: Identifier token
            values_to_insert: Column to value dict

        Returns:
            None
        """
        if not self._row_exists(self.cache_table, id_token):
            return
        self._insert_or_update_details(self.cache_table, id_token, values_to_insert)

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
        own_results = self._get_details(self.cache_table, id_token, cols)
        if not using_most_similar:
            return [own_results, None]
        closest_token = self.get_closest_match(id_token)
        if closest_token is None or closest_token == id_token:
            return [own_results, None]
        closest_match_results = self._get_details(self.cache_table, closest_token, cols)
        return [own_results, closest_match_results]

    def get_origin(self, id_token):
        """
        Gets the origin of a given id token
        Args:
            id_token: Identifier token

        Returns:
            Origin token if applicable
        """
        results = self._get_details(self.cache_table, id_token, ['origin_token'])
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
        return self._get_details_using_origin(self.cache_table, origin_token, cols, has_date_col=True)

    def get_all_details(self, cols, start=0, limit=-1, exclude_token=None,
                        allow_nulls=True, earliest_date=None, latest_date=None, equality_conditions=None,
                        do_date_sort=True, use_date_modified_col=False):
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
            do_date_sort: If False, the results will NOT be sorted by `date_added`, but by `id_token` (FASTER)
            use_date_modified_col: If True, date_modified will be used for earliest/latest date comparisons (
                instead of date_added)
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
        results = self._get_all_details(self.cache_table, cols, start=start, limit=limit, exclude_token=exclude_tokens,
                                        allow_nulls=allow_nulls, earliest_date=earliest_date, latest_date=latest_date,
                                        equality_conditions=equality_conditions, has_date_col=True,
                                        sort_by_date_col=do_date_sort, use_date_modified_col=use_date_modified_col)
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
        return self._get_count(self.cache_table, non_null_cols, equality_conditions)

    def insert_or_update_closest_match(self, id_token, values_to_insert):
        """
        Inserts or updates the value of a row in the closest match table
        Args:
            id_token: Identifier token
            values_to_insert: Dict of values to insert

        Returns:
            None
        """
        self._insert_or_update_details(self.most_similar_table, id_token, values_to_insert)

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
        results = self._get_all_details(self.most_similar_table, ['most_similar_token'], has_date_col=False)
        if results is not None:
            return {x: results[x]['most_similar_token'] for x in results
                    if results[x]['most_similar_token'] is not None}
        return None


class ExampleDBCachingManager(DBCachingManagerBase):
    # This class demonstrates how to extend the DBCachingManagerBase class for a concrete case
    # For most purposes, the default methods defined in the parent class should be appropriate for any and all tasks.
    # Feel free to overwrite any if you have special mechanisms (e.g. if some results expire after a certain amount
    # of time, you may have to overwrite `get_details`, `get_details_using_origin`, etc.)
    def __init__(self):
        """
        db_config: Parameters for the database connection
        cache_table: Name of the main cache table, where the actual results are stored
        most_similar_table: Name of the similarity table, where the similarity relationships between different rows
            of the cache table are stored
        schema: Name of the database schema
        """
        super().__init__(
            db_config={
                'host': 'localhost',
                'port': 3306,
                'user': 'root',
                'pass': ''
            },
            cache_table='Example_Main', most_similar_table='Example_Most_Similar',
            schema='cache_graphai'
        )

    def init_db(self):
        # This method implements an abstract method in the parent.
        # This example demonstrates the general layout of this method, and generally these are all the lines you need.
        # All you have to do is to add your own columns to the definition of the cache table. See more below.

        # Making sure the schema exists
        self.db.execute_query(
            f"""
            CREATE DATABASE IF NOT EXISTS `{self.schema}`
            DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci
            DEFAULT ENCRYPTION='N';
            """
        )

        # Creating the cache table if it does not exist
        # The columns "id_token", "fingerprint", and "date_added" should not have their names changed, since the
        # methods in the parent class rely on their names.
        # The types of "id_token" and "date_added" should not be changed, but "fingerprint" can be LONGTEXT if need be.
        # Aside from these three columns, add any columns you need. For this example, the task could be text
        # classification: there is an "input" and an "output" column, where "output" is the name of the class that
        # "input" has been classified as. "fingerprint" is the fingerprint of "input", and can be null.
        self.db.execute_query(
            f"""
            CREATE TABLE IF NOT EXISTS `{self.schema}`.`{self.cache_table}` (
              `id_token` VARCHAR(255),
              `fingerprint` VARCHAR(255) DEFAULT NULL,
              `input` LONGTEXT DEFAULT NULL,
              `output` VARCHAR(255) DEFAULT NULL,
              `date_added` DATETIME DEFAULT NULL,
              PRIMARY KEY id_token (id_token)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
            """
        )

        # To speed up fingerprint lookups, it is advisable to create a fingerprint index
        try:
            self.db.execute_query(
                f"""
                CREATE INDEX `example_main_fp_index` ON `{self.schema}`.`{self.cache_table}` (`fingerprint`(64));
                """
            )
        except Exception:
            pass

        # Creating the closest match table. This table represents a DAG of most-similar relationships (although it
        # can also have self-loops), where the row with "id_token" as its id is found to be (almost or exactly)
        # identical to the row with "most_similar_token" as its id.
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
