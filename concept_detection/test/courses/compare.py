
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