import time
import configparser
import mysql.connector
from elasticsearch import Elasticsearch
from wikimarkup_stripper.models.stripper import strip

# Read es config and instantiate elasticsearch client
es_config = configparser.ConfigParser()
es_config.read('config/es.ini')
es = Elasticsearch([f'{es_config["ES"].get("host")}:{es_config["ES"].get("port")}'])

# Read db config from file and open connection
db_config = configparser.ConfigParser()
db_config.read('config/db.ini')
cnx = mysql.connector.connect(host=db_config['DB'].get('host'), port=db_config['DB'].getint('port'), user=db_config['DB'].get('user'), password=db_config['DB'].get('password'))
cursor = cnx.cursor()

# Execute query on db
query = f"""
    SELECT PageID, PageTitle, PageContent FROM graph.Nodes_N_Concept
    WHERE PageContent LIKE "%mathematics%"
    LIMIT 100
"""
cursor.execute(query)

start_time = time.time()

i = 0
for page_id, page_title, page_content in cursor:
    stripped_page_content = strip(page_content)
    doc = {
        'id': page_id,
        'title': page_title,
        'content': stripped_page_content
    }
    es.index(index=es_config['ES'].get('index'), document=doc, id=page_id)

    i += 1
    if i % 1e3 == 0:
        if i % 1e4 == 0:
            print('+', end='')
        else:
            print('.', end='')

# Refresh index
es.indices.refresh(index=es_config['ES'].get('index'))

print()
print(f'Finished! Took {time.time() - start_time:.2f}s.')
