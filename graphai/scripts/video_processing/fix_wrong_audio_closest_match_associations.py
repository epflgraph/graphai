from graphai.core.interfaces.caching import AudioDBCachingManager
from graphai.api.celery_tasks.common import (
    fingerprint_lookup_direct,
    fingerprint_lookup_callback
)
from db_cache_manager.db import DB
from graphai.core.interfaces.config import config


def get_all_problematic_pairs():
    db_connector = DB(config['database'])
    cols = [
        'most_similar_id', 'most_similar_fp', 'most_similar_date_added', 'most_similar_duration',
        'id', 'fp', 'date_added', 'duration', 'origin_token'
    ]
    query = """
    SELECT a.id_token AS most_similar_id, a.fingerprint AS most_similar_fp, a.date_added AS most_similar_date_added,
    a.duration AS most_similar_duration,
    b.id_token AS id, b.fingerprint AS fp, b.date_added AS date_added, b.duration AS duration,
    b.origin_token AS origin_token
    FROM cache_graphai.Audio_Main b
    INNER JOIN cache_graphai.Audio_Most_Similar c ON b.id_token=c.id_token
    INNER JOIN cache_graphai.Audio_Main a ON a.id_token=c.most_similar_token
    WHERE a.fingerprint != b.fingerprint OR b.date_added < a.date_added
    ORDER BY date_added;
    """

    results = db_connector.execute_query(query)
    if len(results) == 0:
        return None
    results_dict_list = [{cols[i]: row[i] for i in range(len(cols))} for row in results]
    return results_dict_list


def find_new_closest_match(token, fp):
    db_manager = AudioDBCachingManager()
    input_dict = {
        'fp_token': token, 'result': fp, 'perform_lookup': True
    }
    step_1 = fingerprint_lookup_direct(input_dict, db_manager)
    step_2 = fingerprint_lookup_callback(step_1, db_manager)
    return step_2


def main():
    all_problem_pairs = get_all_problematic_pairs()
    print('Retrieved problem pairs from db')
    if all_problem_pairs is None:
        print('No problems found, exiting...')
        return

    counter = 0
    for row in all_problem_pairs:
        print('Token: %s' % row['id'])
        if row['fp'] != row['most_similar_fp']:
            print('Problem: fingerprint not equal to that of closest match.')
        else:
            print('Problem: closest match added later than token itself.')
        new_match = find_new_closest_match(row['id'], row['fp'])
        print("New closest match: %s" % new_match['closest'])
        counter += 1

        if counter % 10 == 0:
            print("Rows processed: %d" % counter)


if __name__ == '__main__':
    main()
