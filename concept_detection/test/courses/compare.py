



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