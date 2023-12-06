from configparser import ConfigParser

from graphai.definitions import ROOT_DIR

# Load the configuration file
parser = ConfigParser()
parser.read(f'{ROOT_DIR}/config.ini')

config = {section: dict(parser[section]) for section in parser.sections()}
