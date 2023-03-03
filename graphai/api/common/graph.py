from graphai.api.common import log

from graphai.core.common import ConceptsGraph

# Create a ConceptsGraph instance to hold concepts graph in memory
log(f'Fetching concepts graph from database...')
graph = ConceptsGraph()
