import matplotlib.pyplot as plt

from concept_detection.interfaces.db import DB
from concept_detection.interfaces.api import Api

plt.style.use('seaborn-dark')


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


def plot_stats(stats, label_name):
    n = len(stats)
    m = len(next(iter(stats.values())))

    fig, axs = plt.subplots(n, m, figsize=(18, 10))
    i = 0
    for how in stats:
        j = 0
        for method in stats[how]:
            confusions = stats[how][method]
            labels = [confusion[label_name] for confusion in confusions]

            tp = [confusion['tp'] for confusion in confusions]
            fn = [confusion['fn'] for confusion in confusions]

            sorted_zip_lists = sorted(zip(tp, fn))

            tp = [x for x, _ in sorted_zip_lists]
            fn = [y for _, y in sorted_zip_lists]

            axs[i, j].bar(labels, fn, label='FN', alpha=0.8, bottom=tp)
            axs[i, j].bar(labels, tp, label='TP', alpha=0.8)

            axs[i, j].set_xlabel(label_name)

            axs[i, j].axes.xaxis.set_ticklabels([])

            h, l = axs[i, j].get_legend_handles_labels()
            axs[i, j].legend(reversed(h), reversed(l))
            axs[i, j].set_title(f'{how} - {method}')

            j += 1
        i += 1

    plt.show()


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
        'n_equivalent': n_equivalent,
        'n_equal': n_equal,
        'ok': n_equivalent == n_equal,
        'differences': differences
    }


def compare_wikify_course_descriptions_api_db(methods, limit=None):
    db = DB()
    api = Api()

    # Fetch courses and their descriptions from db
    course_descriptions = db.query_course_descriptions()
    course_ids = list(course_descriptions.keys())
    n_courses = len(course_ids)
    print(f'Got {n_courses} courses')

    # Fetch wikify results from older executions from db
    db_wikified_course_descriptions = db.query_wikified_course_descriptions()

    # Prepare data structure to store how well db and api results compare
    comparative = {
        'results': [],
        'stats': {
            'pair': {method: [] for method in methods},
            'page': {method: [] for method in methods}
        }
    }

    i = 0
    for course_id in course_ids:
        i += 1
        print(i)

        # Get course description and its anchor page ids to use in the API call
        raw_text = course_descriptions[course_id]
        anchor_page_ids = db.course_anchor_page_ids(course_id)

        # Perform api calls to the service running on the server
        api_wikified_course_description = {}
        for method in methods:
            api_wikified_course_description[method] = api.wikify(raw_text, anchor_page_ids, method=method)

        # Compare api lists of results with db list of results
        course_comparative = {}
        for method in methods:
            course_comparative[method] = compare_result_lists(api_wikified_course_description[method], db_wikified_course_descriptions[course_id])

        # Update comparative with the results of the current course
        comparative_summary = {
            'course_id': course_id,
            'raw_text': raw_text,
            'db': db_wikified_course_descriptions[course_id]
        }
        for method in methods:
            comparative_summary[method] = api_wikified_course_description[method]
        comparative['results'].append(comparative_summary)

        # Extract results for pairs (keywords, page_id) and compute pair confusion stats
        db_pairs = [(result.keywords, result.page_id) for result in db_wikified_course_descriptions[course_id]]
        for method in methods:
            api_pairs = [(result.keywords, result.page_id) for result in api_wikified_course_description[method]]

            pair_stats = confusion_stats(api_pairs, db_pairs)
            pair_stats['course_id'] = course_id
            comparative['stats']['pair'][method].append(pair_stats)

        # Extract results for pages (page_id) and compute page confusion stats
        db_page_ids = [result.page_id for result in db_wikified_course_descriptions[course_id]]
        for method in methods:
            api_page_ids = [result.page_id for result in api_wikified_course_description[method]]

            page_stats = confusion_stats(api_page_ids, db_page_ids)
            page_stats['course_id'] = course_id
            comparative['stats']['page'][method].append(page_stats)

        if i == limit:
            break

    return comparative

