import json
import numpy as np


def f(x):
    return 1 / (1 + np.exp(-8 * (x - 1 / 2)))


def g(x, a=1):
    if x <= 0 or x >= 2:
        return 0
    return np.exp(a + a/(x*(x-2)))


def h(x):
    if x <= 0.5:
        return 2 * x**2
    else:
        return 1 - 2 * (x - 1)**2


def zipsorted(a, b, c):
    zip_sorted = sorted(zip(a, b, c))

    a = [x for x, _, _ in zip_sorted]
    b = [y for _, y, _ in zip_sorted]
    c = [z for _, _, z in zip_sorted]

    return a, b, c


with open('results/course-descriptions-keywords-to-results-results.json') as file:
    results = json.load(file)

t = {
    'n_keywords': 0,
    'nonempty': 0,
    'present': 0,
    'hits': {
        'sr': 0,
        'mgs': 0,
        'sgr': 0,
        'lev': 0,
        'mix': 0,
        'own1': 0,
        'own2': 0,
        'ownp':0
    }
}
scores = {
    'mix': [],
    'own1': [],
    'own2': [],
    'ownp': []
}
for result in results:
    for keywords in result['keywords_results']:
        t['n_keywords'] += 1

        [db_result] = result['keywords_results'][keywords]['db']
        db_page_id = db_result['page_id']

        new_results = result['keywords_results'][keywords]['new']

        if len(new_results) == 0:
            continue

        t['nonempty'] += 1

        matching_results = [new_result for new_result in new_results if new_result['page_id'] == db_page_id]

        if not matching_results:
            continue

        t['present'] += 1

        matching_result = matching_results[0]

        for i in range(len(new_results)):
            new_results[i]['own1_score'] = new_results[i]['searchrank_graph_ratio'] * g(new_results[i]['levenshtein_score'], 1)
            new_results[i]['own2_score'] = new_results[i]['searchrank_graph_ratio'] * g(new_results[i]['levenshtein_score'], 2)
            new_results[i]['ownp_score'] = new_results[i]['searchrank_graph_ratio'] * h(new_results[i]['levenshtein_score'])

        best_results = {
            'sr': sorted(new_results, key=lambda x: x['searchrank'])[0],
            'mgs': sorted(new_results, key=lambda x: x['median_graph_score'], reverse=True)[0],
            'sgr': sorted(new_results, key=lambda x: x['searchrank_graph_ratio'], reverse=True)[0],
            'lev': sorted(new_results, key=lambda x: x['levenshtein_score'], reverse=True)[0],
            'mix': sorted(new_results, key=lambda x: x['mixed_score'], reverse=True)[0],
            'own1': sorted(new_results, key=lambda x: x['own1_score'], reverse=True)[0],
            'own2': sorted(new_results, key=lambda x: x['own2_score'], reverse=True)[0],
            'ownp': sorted(new_results, key=lambda x: x['ownp_score'], reverse=True)[0]
        }

        for key in best_results:
            if best_results[key]['page_id'] == db_page_id:
                t['hits'][key] += 1

        scores['mix'].append(matching_result['mixed_score'])
        scores['own1'].append(matching_result['own1_score'])
        scores['own2'].append(matching_result['own2_score'])
        scores['ownp'].append(matching_result['ownp_score'])


print(f'Number of keywords: {t["n_keywords"]}')
print(f'-- with results: {t["nonempty"]} ({t["nonempty"] / t["n_keywords"] :.4f})')
print(f'---- with matching page: {t["present"]} ({t["present"] / t["nonempty"] :.4f}, {t["present"] / t["n_keywords"] :.4f} from total)')
for key, value in sorted(t['hits'].items(), key=lambda x: x[1], reverse=True):
    print(f'------ matching {key}: {t["hits"][key]} ({t["hits"][key] / t["present"] :.4f}, {t["hits"][key] / t["n_keywords"] :.4f} from total)')
print(f'---- without matching page: {t["nonempty"] - t["present"]} ({1 - t["present"] / t["nonempty"] :.4f}, {(t["nonempty"] - t["present"]) / t["n_keywords"] :.4f} from total)')
print(f'-- without results: {t["n_keywords"] - t["nonempty"]} ({1 - t["nonempty"] / t["n_keywords"] :.4f})')

# import matplotlib.pyplot as plt
# m, o1, o2 = zipsorted(scores['mix'], scores['own1'], scores['own2'])
#
# fig, ax = plt.subplots()
# ax.plot(o1, label='Own1', alpha=0.6)
# ax.plot(o2, label='Own2', alpha=0.6)
# ax.plot(m, label='Mixed', alpha=0.6)
# ax.legend()
# plt.show()

