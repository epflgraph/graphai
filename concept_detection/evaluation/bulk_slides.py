import time

from db_api import DBApi
from new_api import NewApi
from old_api import OldApi

from compare import *

from utils.time.stopwatch import Stopwatch


def summary_slides(slide_ids):
    channel_ids = [slide_id.split('_')[0] for slide_id in slide_ids]
    video_ids = [slide_id.split('_')[1] for slide_id in slide_ids]

    return f'Got {len(slide_ids)} from {len(set(video_ids))} different videos belonging to {len(set(channel_ids))} different channels'


def test_db(limit=10, offset=0, pseudorandom=False, fallback_to_old=True):
    # Instantiate api objects
    db_api = DBApi()
    new_api = NewApi()
    old_api = OldApi()

    # Extract slide texts from db
    slide_texts = db_api.query_slide_texts(limit=limit, offset=offset, pseudorandom=pseudorandom)
    slide_ids = list(slide_texts.keys())
    n_slides = len(slide_ids)
    print(summary_slides(slide_ids))

    # Query wikified slides from db
    db_wikified_slides = db_api.query_wikified_slides(slide_ids)
    print(f'Got {len(db_wikified_slides)} wikified slides from db')

    # Iterate over all slides and compare the results
    t = {
        'n_slides': n_slides,
        'ok_vs_db': 0,
        'ok_already_seen': 0,
        'ok_vs_old': 0,
        'not_ok': 0,
        'already_seen': [],
        'different': [],
        'venns_db': [],
        'venns_old': []
    }
    i = 0
    for slide_id in slide_ids:
        i += 1

        raw_text = slide_texts[slide_id]
        anchor_page_ids = db_api.slide_anchor_page_ids(slide_id)

        # Comparing new vs. db
        new_wikified_slide = new_api.wikify(raw_text, anchor_page_ids)
        comp = compare(new_wikified_slide, db_wikified_slides[slide_id])

        # Append venn diagram information
        t['venns_db'].append(slide_extract_venn(slide_id, comp))

        # If ok, update counters and continue
        if comp['ok']:
            t['ok_vs_db'] += 1
            print('.', end='')
            if i % 100 == 0 or i == n_slides:
                print()
            continue

        # Comparing new vs. db failed.
        # Check which of the page combinations we have already seen
        source_page_ids = [diff['1']['page_id'] for diff in comp['differences']]
        new_combs = new_page_combinations(source_page_ids, anchor_page_ids, t['already_seen'])

        # If no new combinations
        if not new_combs:
            t['ok_already_seen'] += 1
            print(':', end='')
            if i % 100 == 0 or i == n_slides:
                print()
            continue

        # There are new differences
        # Save the new differences to report them
        t['already_seen'].extend(new_combs)

        # Fallback to old api if enabled
        if fallback_to_old:
            # Comparing new vs. db
            old_wikified_slide = old_api.wikify(raw_text, anchor_page_ids)
            comp = compare(new_wikified_slide, old_wikified_slide)

            # Append venn diagram information
            t['venns_old'].append(slide_extract_venn(slide_id, comp))

            # If ok, update counters and continue
            if comp['ok']:
                t['ok_vs_old'] += 1
                print('+', end='')
                if i % 100 == 0 or i == n_slides:
                    print()
                continue

        # Comparisons failed, save the differences to report them
        t['not_ok'] += 1
        t['different'].append({
            'slide_id': slide_id,
            'req': {
                'raw_text': raw_text,
                'anchor_page_ids': anchor_page_ids
            },
            'differences': comp['differences']
        })
        print('*', end='')
        if i % 100 == 0 or i == n_slides:
            print()

    return t


if __name__ == '__main__':
    sw = Stopwatch()

    limit = 1000
    offset = 0
    pseudorandom = True
    t = test_db(limit=limit, offset=offset, pseudorandom=pseudorandom, fallback_to_old=True)

    pprint({key: t[key] for key in t if 'venns' not in key and key != 'already_seen'})

    save_failed(t, f'results/{limit}x{offset}x{pseudorandom}-failed.json')
    plot_venns(t['venns_db'], 'slide_id', f'results/{limit}x{offset}x{pseudorandom}-new-db.png')
    plot_venns(t['venns_old'], 'slide_id', f'results/{limit}x{offset}x{pseudorandom}-new-old.png')

    total_time = sw.delta()
    print(f'Total time: {int(total_time//60)}m {total_time%60:.2f}s')
