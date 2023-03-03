from graphai.api.common import log

from graphai.core.common.ontology import Ontology

# Create an Ontology instance to hold ontology graph in memory
log(f'Fetching ontology from database...')
ontology = Ontology()
