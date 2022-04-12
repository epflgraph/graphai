import configparser
from elasticsearch import Elasticsearch

es_config = configparser.ConfigParser()
es_config.read('config/es.ini')
es = Elasticsearch([f'{es_config["ES"].get("host")}:{es_config["ES"].get("port")}'])

query = {
    'match': {
        'content': """
            We are interested in various aspects of spectral rigidity of Cayley and Schreier graphs of finitely generated groups. For each pair of integers d≥2 and m≥1, we consider an uncountable family of groups of automorphisms of the rooted d-regular tree which provide examples of the following interesting phenomena. For d=2 and any m≥2, we get an uncountable family of non quasi-isometric Cayley graphs with the same Laplacian spectrum, absolutely continuous on the union of two intervals, that we compute explicitly. Some of the groups provide examples where the spectrum of the Cayley graph is connected for one generating set and has a gap for another.
            For each d≥3,m≥1, we exhibit infinite Schreier graphs of these groups with the spectrum a Cantor set of Lebesgue measure zero union a countable set of isolated points accumulating on it. The Kesten spectral measures of the Laplacian on these Schreier graphs are discrete and concentrated on the isolated points. We construct moreover a complete system of eigenfunctions which are strongly localized.
        """
    }
}
r = es.search(index=es_config['ES'].get('index'), query=query, highlight={
    "fields": {
      "content": {}
    }
  })
print(r)

# query = {
#     'match': {
#         'content': """
#             skldjweflkjnrelkjnfrwlknflkdajskhdjklsahdklsahdkjlashdkljhakljahdkljh
#         """
#     }
# }
# r = es.search(index=index, query=query, highlight={
#     "fields": {
#       "content": {}
#     }
#   })
# print(r)

r = es.cat.indices(index=es_config['ES'].get('index'), format='json', v=True)
print(r)
