# Wikitext
The `wikitext` module provides a simple way to interact with the Concepts Detection API via a python class.

### Basic usage
We can wikify a text as follows:
```
from wikitext import *

text = "The Pythagorean theorem does not hold in hyperbolic geometry."
wt = Wikitext(text)
print(wt.results)
```

This prints the following list:
```
[
  {
    'keywords': 'hyperbolic geometry',
    'page_id': 241291,
    'page_title': 'Hyperbolic_geometry',
    'searchrank': 1,
    'median_graph_score': 0.5477285971014862,
    'searchrank_graph_ratio': 0.5477285971014862,
    'levenshtein_score': 1.0,
    'mixed_score': 0.537877035551777
  },
  {
    'keywords': 'hyperbolic geometry',
    'page_id': 58610,
    'page_title': 'Non-euclidean_geometry',
    'searchrank': 2,
    'median_graph_score': 0.5527719516844014,
    'searchrank_graph_ratio': 0.2763859758422007,
    'levenshtein_score': 0.5853658536585366,
    'mixed_score': 0.18362852678827307
  },
  {
    'keywords': 'pythagorean theorem',
    'page_id': 26513034,
    'page_title': 'Pythagorean_theorem',
    'searchrank': 1,
    'median_graph_score': 0.5412036441689434,
    'searchrank_graph_ratio': 0.5412036441689434,
    'levenshtein_score': 1.0,
    'mixed_score': 0.5314694417926717
  },
  {
    'keywords': 'pythagorean theorem',
    'page_id': 24172,
    'page_title': 'Pythagorean_triple',
    'searchrank': 2,
    'median_graph_score': 0.47864712416838334,
    'searchrank_graph_ratio': 0.23932356208419167,
    'levenshtein_score': 0.8108108108108109,
    'mixed_score': 0.2209408861503205
  }
]
```

### Anchor Pages
Anchor pages are a list of Wikipedia pages that are used to define a search space in the concepts graph. In the previous example, we did not provide any, so they were automatically selected. To fine-tune the results, we may provide a list of anchor page ids, as follows:
```
from wikitext import *

text = "The Pythagorean theorem does not hold in hyperbolic geometry."
wt = Wikitext(text, anchor_page_ids=[9417])
print(wt.results)
```

This prints the following list:
```
[
  {
    'keywords': 'hyperbolic geometry',
    'page_id': 241291,
    'page_title': 'Hyperbolic_geometry',
    'searchrank': 1,
    'median_graph_score': 0.4992023450539821,
    'searchrank_graph_ratio': 0.4992023450539821,
    'levenshtein_score': 1.0,
    'mixed_score': 0.4902235868622727
  },
  {
    'keywords': 'hyperbolic geometry',
    'page_id': 58610,
    'page_title': 'Non-euclidean_geometry',
    'searchrank': 2,
    'median_graph_score': 0.5289605348648821,
    'searchrank_graph_ratio': 0.2644802674324411,
    'levenshtein_score': 0.5853658536585366,
    'mixed_score': 0.17571847386683573
  },
  {
    'keywords': 'pythagorean theorem',
    'page_id': 26513034,
    'page_title': 'Pythagorean_theorem',
    'searchrank': 1,
    'median_graph_score': 0.5622701472459408,
    'searchrank_graph_ratio': 0.5622701472459408,
    'levenshtein_score': 1.0,
    'mixed_score': 0.5521570383221592
  },
  {
    'keywords': 'pythagorean theorem',
    'page_id': 24172,
    'page_title': 'Pythagorean_triple',
    'searchrank': 2,
    'median_graph_score': 0.37122929222091594,
    'searchrank_graph_ratio': 0.18561464611045797,
    'levenshtein_score': 0.8108108108108109,
    'mixed_score': 0.17135740433152946
  },
  {
    'keywords': 'pythagorean theorem',
    'page_id': 9417,
    'page_title': 'Euclidean_geometry',
    'searchrank': 9,
    'median_graph_score': 1.0,
    'searchrank_graph_ratio': 0.1111111111111111,
    'levenshtein_score': 0.3783783783783784,
    'mixed_score': 0.03047648026684824
  }
]
``` 
Notice the difference in some scores and the extra result.

### Attributes and methods
A Wikitext has the following attributes:
* `raw_text`: The string passed as parameter when creating the object.
* `anchor_page_ids`: The list of anchor page ids passed as parameter when creating the object. Default: None.
* `results`: The list of results after wikifying the text.

And the following methods are available:
* `keywords()`: Returns a list of the unique keywords present in the results. 
* `page_ids()`: Returns a list of the unique page ids present in the results.
* `unique_pairs()`: Returns a list of the unique pairs (keywords, page id) present in the results.
* `page_titles()`: Returns a list of the unique page titles present in the results.
* `keywords_results(keywords)`: Returns a list of the results matching the provided keywords.
* `page_results(page_id)`: Returns a list of the results matching the provided page id.
* `pair_results(keywords, page_id)`: Returns a list of the results matching the provided pair (keywords, page id).
* `keywords_aggregated()`: Aggregates results with the same keywords averaging their scores.
* `page_aggregated()`: Aggregates results with the same page id averaging their scores.

The `wikitext` module also provides the following function: 
* `combine(wikitexts, f:None)`: Returns all results for all wikitexts with the scores aggregated according to f.
    * Args:
        * `wikitexts (list[Wikitext])`: A list of Wikitext objects.
        * `f (Callable[list[Number], Number])`: A function to aggregate the scores of the common pages. Default: None (numpy.mean)

    * Returns:
        * `list[dict]`: A list of the common results aggregated according to f.

Please check the [[https://c4science.ch/source/wikitext/browse/master/example.py | example file]] for more details.