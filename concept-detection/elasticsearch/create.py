import time
import configparser
from elasticsearch import Elasticsearch
import mysql.connector
from concurrent.futures import as_completed
from requests_futures.sessions import FuturesSession

start_time = time.time()

# URL for markup strip entry point
STRIP_URL = 'http://86.119.27.90:30010/strip'

# Read es config and instantiate elasticsearch client
es_config = configparser.ConfigParser()
es_config.read('config/es.ini')
es = Elasticsearch([f'{es_config["ES"].get("host")}:{es_config["ES"].get("port")}'])
index = 'wikimath'
es.indices.delete(index=index, ignore_unavailable=True)

# Read db config from file and open connection
db_config = configparser.ConfigParser()
db_config.read('config/db.ini')
cnx = mysql.connector.connect(host=db_config['DB'].get('host'), port=db_config['DB'].getint('port'), user=db_config['DB'].get('user'), password=db_config['DB'].get('password'))
cursor = cnx.cursor()

# Execute query on db
query = f"""
    SELECT PageID, PageTitle, PageContent FROM graph.Nodes_N_Concept
    WHERE PageContent LIKE "%mathematics%"
"""
cursor.execute(query)

# Launch API calls and store futures
print('Launching markup strip calls')
session = FuturesSession()
futures = []
for page_id, page_title, page_content in cursor:
    future = session.post(STRIP_URL, json={'markup_code': page_content})
    future.id = page_id
    future.title = page_title
    futures.append(future)

    if len(futures) % 10e3 == 0:
        if len(futures) % 10e4 == 0:
            print('+', end='')
        else:
            print('.', end='')

print()
print('Indexing documents as responses arrive')
# Add documents to elasticsearch index as responses arrive
i = 0
for future in as_completed(futures):
    page_content = future.result().json()['stripped_code']
    doc = {
        'id': future.id,
        'title': future.title,
        'content': page_content
    }
    es.index(index=index, document=doc, id=future.id)

    i += 1
    if i % 10e3 == 0:
        if i % 10e4 == 0:
            print('+', end='')
        else:
            print('.', end='')

# Refresh index
es.indices.refresh(index=index)

print()
print(f'Finished! Took {time.time() - start_time:.2f}s.')
