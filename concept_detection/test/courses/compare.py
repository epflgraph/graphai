import matplotlib.pyplot as plt

from concept_detection.test.courses.db import DB
from concept_detection.test.courses.api import Api


def confusion_stats(predicted, actual):
    predicted = set(predicted)
    actual = set(actual)

    pred = len(predicted)
    act = len(actual)
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)

    p = tp / pred if pred else 0
    r = tp / act if act else 0

    f_score = 2 / (1 / p + 1 / r) if p > 0 and r > 0 else 0

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


def plot_confs(confs, label_name, title):
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
    ax.set_title(title)

    plt.show()


def compare_result_lists(results_1, results_2):
    n_equivalent = 0
    n_equal = 0
    differences = []
    for result_1 in results_1:
        for result_2 in results_2:
            if result_1.equivalent(result_2):
                n_equivalent += 1

                if result_1.equal(result_2):
                    n_equal += 1
                else:
                    differences.append({
                        '1': result_1,
                        '2': result_2
                    })

    return {
        'results_1': results_1,
        'results_2': results_2,
        'n_results_1': len(results_1),
        'n_results_2': len(results_2),
        'n_equivalent': n_equivalent,
        'n_equal': n_equal,
        'ok': n_equivalent == n_equal,
        'differences': differences
    }


def compare_course_descriptions_api_db(limit=None):
    db = DB()
    api = Api()

    course_descriptions = db.query_course_descriptions()
    course_ids = list(course_descriptions.keys())
    n_courses = len(course_ids)
    print(f'Got {n_courses} courses')

    db_wikified_course_descriptions = db.query_wikified_course_descriptions()

    comparative = {
        'results': [],
        'pair_confs': [],
        'page_confs': []
    }
    i = 0
    for course_id in course_ids:
        i += 1
        print(i)

        raw_text = course_descriptions[course_id]
        anchor_page_ids = db.course_anchor_page_ids(course_id)

        api_wikified_course_description = api.wikify(raw_text, anchor_page_ids)
        comp = compare_result_lists(api_wikified_course_description, db_wikified_course_descriptions[course_id])

        comparative['results'].append({
            'course_id': course_id,
            'raw_text': raw_text,
            'new': comp['results_1'],
            'db': comp['results_2']
        })

        new_pairs = [(result.keywords, result.page_id) for result in comp['results_1']]
        db_pairs = [(result.keywords, result.page_id) for result in comp['results_2']]

        conf = confusion_stats(new_pairs, db_pairs)
        conf['course_id'] = course_id
        comparative['pair_confs'].append(conf)

        new_page_ids = [result.page_id for result in comp['results_1']]
        db_page_ids = [result.page_id for result in comp['results_2']]

        conf = confusion_stats(new_page_ids, db_page_ids)
        conf['course_id'] = course_id
        comparative['page_confs'].append(conf)

        if i == limit:
            break

    return comparative

