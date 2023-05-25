import os
import platform

# Directories
PKG_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(PKG_DIR)

CONFIG_DIR = f'{ROOT_DIR}/config'
DOCS_DIR = f'{ROOT_DIR}/docs'
DATA_DIR = f'{ROOT_DIR}/data'
TESTS_DIR = f'{ROOT_DIR}/tests'

# IPs
GRAPH_AI_TEST_IP = '192.168.142.120'
GRAPH_AI_PROD_IP = '192.168.140.120'

# URLs
node = platform.node()
if node == 'graph-ai-test':
    LOCAL_API_URL = f'http://{GRAPH_AI_TEST_IP}:28800'
elif node == 'graph-ai-prod':
    LOCAL_API_URL = f'http://{GRAPH_AI_PROD_IP}:28800'
else:
    LOCAL_API_URL = 'http://localhost:28800'

TEST_API_URL = 'http://86.119.27.90:28800'
