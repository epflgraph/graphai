from wikitext import *

euclid_5 = """
    5. That, if a straight line falling on two straight lines makes the interior angles on the same side less than two
    right angles, the two straight lines, if produced indefinitely, meet on that side on which are the angles less
    than the two right angles.
"""

wt = Wikitext(euclid_5)

print(f'results ({len(wt.results)})')
print(wt.results)

print(f'keywords ({len(wt.keywords())})')
print(wt.keywords())

print(f'page_ids ({len(wt.page_ids())})')
print(wt.page_ids())

print(f'unique_pairs ({len(wt.unique_pairs())})')
print(wt.unique_pairs())

print(f'page_titles ({len(wt.page_titles())})')
print(wt.page_titles())

keywords = "straight line falling"
print(f'keywords_results {keywords} ({len(wt.keywords_results(keywords))})')
print(wt.keywords_results(keywords))

page_id = 33731493
print(f'page_results {page_id} ({len(wt.page_results(page_id))})')
print(wt.page_results(page_id))

print(f'pair_results {keywords}, {page_id} ({len(wt.pair_results(keywords, page_id))})')
print(wt.pair_results(keywords, page_id))

print(f'keywords_aggregated ({len(wt.keywords_aggregated())})')
print(wt.keywords_aggregated())

print(f'page_aggregated ({len(wt.page_aggregated())})')
print(wt.page_aggregated())

euclid_1_4 = """
    1. To draw a straight line from any point to any point.
    2. To produce a finite straight line continuously in a straight line.
    3. To describe a circle with any center and radius.
    4. That all right angles equal one another.
"""

wt2 = Wikitext(euclid_1_4)
print(combine([wt, wt2]))
