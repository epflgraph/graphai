import json
import matplotlib.pyplot as plt


def pprint(d):
    print(json.dumps(d, indent=4))


def equivalent(result1, result2):
    return result1['keywords'] == result2['keywords'] and result1['page_id'] == result2['page_id']


def equal(result1, result2):
    return equivalent(result1, result2) and abs(result1['median_graph_score'] - result2['median_graph_score']) < 10e-4


def compare(results1, results2):
    n_equivalent = 0
    n_equal = 0
    differences = []
    for result1 in results1:
        for result2 in results2:
            if equivalent(result1, result2):
                n_equivalent += 1

                if equal(result1, result2):
                    n_equal += 1
                else:
                    differences.append({
                        '1': result1,
                        '2': result2
                    })

    return {
        'results_1': results1,
        'results_2': results2,
        'n_results_1': len(results1),
        'n_results_2': len(results2),
        'n_equivalent': n_equivalent,
        'n_equal': n_equal,
        'ok': n_equivalent == n_equal,
        'differences': differences
    }


def confusion_matrix(predicted, actual):
    predicted = set(predicted)
    actual = set(actual)

    pred = len(predicted)
    act = len(actual)
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)

    p = tp / pred if pred else 0
    r = tp / act if act else 0

    f_score = 2 / (1/p + 1/r) if p > 0 and r > 0 else 0

    return {
        'pred': pred,
        'act': act,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'p': p,
        'r': r,
        'f-score': f_score
    }


def new_page_combinations(source_page_ids, anchor_page_ids, already_seen):
    new_combs = []

    for source_page_id in source_page_ids:
        comb = {
            'source_page_id': source_page_id,
            'anchor_page_ids': anchor_page_ids
        }
        if comb not in already_seen:
            new_combs.append(comb)

    return new_combs


def save_failed(t, filename):
    failed_reqs = []
    for diff in t['different']:
        failed_reqs.append(diff['req'])

    with open(filename, 'w') as f:
        f.write(json.dumps(failed_reqs, indent=4))


def save(d, filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(d, indent=4))


def slide_extract_venn(slide_id, comp):
    r = extract_venn(comp)
    r['slide_id'] = slide_id
    return r


def course_extract_venn(course_id, comp):
    r = extract_venn(comp)
    r['course_id'] = course_id
    return r


def extract_venn(comp):
    return {
        '1': comp['n_results_1'],
        '2': comp['n_results_2'],
        'intersection': comp['n_equivalent'],
        'only_1': comp['n_results_1'] - comp['n_equivalent'],
        'only_2': comp['n_results_2'] - comp['n_equivalent']
    }


def plot_venns(venns, label_name, filename):
    labels = [venn[label_name] for venn in venns]

    only1 = [venn['only_1'] for venn in venns]
    intersection = [venn['intersection'] for venn in venns]
    only2 = [venn['only_2'] for venn in venns]

    sorted_zip_lists = sorted(zip(only1, intersection, only2), key=lambda t: (-t[1], t[0]))

    only1 = [x for x, _, _ in sorted_zip_lists]
    intersection = [y for _, y, _ in sorted_zip_lists]
    only2 = [z for _, _, z in sorted_zip_lists]

    fig, ax = plt.subplots()
    ax.bar(labels, only1, label='Only 1', alpha=0.8)
    ax.bar(labels, intersection, label='Intersection', alpha=0.8, bottom=only1)
    ax.bar(labels, only2, label='Only 2', alpha=0.8, bottom=[sum(x) for x in zip(only1, intersection)])
    ax.axes.xaxis.set_ticklabels([])
    h, l = ax.get_legend_handles_labels()
    ax.legend(reversed(h), reversed(l))
    ax.set_title(filename)

    plt.savefig(filename, dpi=500)


def plot_venns_simple(venns, label_name, filename):
    labels = [venn[label_name] for venn in venns]

    intersection = [venn['intersection'] for venn in venns]
    only2 = [venn['only_2'] for venn in venns]

    sorted_zip_lists = sorted(zip(intersection, only2))

    intersection = [x for x, _ in sorted_zip_lists]
    only2 = [y for _, y in sorted_zip_lists]

    fig, ax = plt.subplots()
    ax.bar(labels, only2, label='Only 2', alpha=0.8, bottom=intersection)
    ax.bar(labels, intersection, label='Intersection', alpha=0.8)
    ax.axes.xaxis.set_ticklabels([])
    h, l = ax.get_legend_handles_labels()
    ax.legend(reversed(h), reversed(l))
    ax.set_title(filename)

    plt.savefig(filename, dpi=500)


def plot_confs(confs, label_name, filename):
    labels = [conf[label_name] for conf in confs]

    tp = [conf['tp'] for conf in confs]
    fn = [conf['fn'] for conf in confs]

    sorted_zip_lists = sorted(zip(tp, fn))

    tp = [x for x, _ in sorted_zip_lists]
    fn = [y for _, y in sorted_zip_lists]

    fig, ax = plt.subplots()
    ax.bar(labels, fn, label='FN', alpha=0.8, bottom=tp)
    ax.bar(labels, tp, label='TP', alpha=0.8)
    ax.axes.xaxis.set_ticklabels([])
    h, l = ax.get_legend_handles_labels()
    ax.legend(reversed(h), reversed(l))
    ax.set_title(filename)

    plt.savefig(filename, dpi=500)
