from api.common.log import log

from core.common.ontology import Ontology

# Create an Ontology instance to hold ontology graph in memory
log(f'Fetching ontology from database...')
ontology = Ontology()
