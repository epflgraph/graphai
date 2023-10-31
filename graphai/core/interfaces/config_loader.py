import configparser

from graphai.definitions import CONFIG_DIR

DEFAULT_SCHEMA = 'cache_graphai'


def load_db_config(config_dir=None):
    if config_dir is None:
        config_dir = CONFIG_DIR
    db_config = configparser.ConfigParser()
    db_config.read(f'{config_dir}/db.ini')

    config_params = dict()
    config_params['host'] = db_config['DB'].get('host')
    config_params['port'] = db_config['DB'].getint('port')
    config_params['user'] = db_config['DB'].get('user')
    config_params['pass'] = db_config['DB'].get('password')
    return config_params


def load_schema_name(config_dir=None):
    if config_dir is None:
        config_dir = CONFIG_DIR
    config_contents = configparser.ConfigParser()
    try:
        print('Reading cache storage configuration from file')
        config_contents.read(f'{config_dir}/cache.ini')
        schema = config_contents['CACHE'].get('schema', fallback=DEFAULT_SCHEMA)
    except Exception:
        print(f'Could not read file {config_dir}/cache.ini or '
              f'file does not have section [CACHE] with parameter "schema", '
              f'falling back to default.')
        schema = DEFAULT_SCHEMA
    return schema
