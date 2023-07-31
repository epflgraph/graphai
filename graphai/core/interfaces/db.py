import sys

import pandas as pd

import mysql.connector
import configparser

from graphai.definitions import CONFIG_DIR


def quote_value(v):
    if isinstance(v, str):
        return f'"{v}"'
    else:
        return v


class DB:
    """
    Base class to communicate with the EPFLGraph database.
    """

    def __init__(self):
        # Read db config from file and open connection
        db_config = configparser.ConfigParser()
        db_config.read(f'{CONFIG_DIR}/db.ini')

        self.host = db_config['DB'].get('host')
        self.port = db_config['DB'].getint('port')
        self.user = db_config['DB'].get('user')
        self.password = db_config['DB'].get('password')

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
