from graphai.api.common.log import log

from graphai.core.common.graph import ConceptsGraph

# Create a ConceptsGraph instance to hold concepts graph in memory
log(f'Fetching concepts graph from database...')
graph = ConceptsGraph()
