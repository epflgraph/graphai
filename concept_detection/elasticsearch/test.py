from concept_detection.text.io import pprint
from concept_detection.interfaces.es import ES

es = ES()

# query = {
#     'match': {
#         'content': """
#             We are interested in various aspects of spectral rigidity of Cayley and Schreier graphs of finitely generated groups. For each pair of integers d≥2 and m≥1, we consider an uncountable family of groups of automorphisms of the rooted d-regular tree which provide examples of the following interesting phenomena. For d=2 and any m≥2, we get an uncountable family of non quasi-isometric Cayley graphs with the same Laplacian spectrum, absolutely continuous on the union of two intervals, that we compute explicitly. Some of the groups provide examples where the spectrum of the Cayley graph is connected for one generating set and has a gap for another.
#             For each d≥3,m≥1, we exhibit infinite Schreier graphs of these groups with the spectrum a Cantor set of Lebesgue measure zero union a countable set of isolated points accumulating on it. The Kesten spectral measures of the Laplacian on these Schreier graphs are discrete and concentrated on the isolated points. We construct moreover a complete system of eigenfunctions which are strongly localized.
#         """
#     }
# }

query = {
    'bool': {
        'must_not': {
            'exists': {
                'field': 'content'
            }
        }
    }
}
r = es._search(query=query).body
pprint(r)

r = es.indices().body
pprint(r)
