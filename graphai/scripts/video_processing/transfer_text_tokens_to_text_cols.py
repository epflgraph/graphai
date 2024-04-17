from graphai.core.interfaces.caching import SlideDBCachingManager, AudioDBCachingManager, VideoConfig
from db_cache_manager.db import escape_everything
from graphai.core.video.video import read_txt_gz_file
from graphai.core.common.common_utils import read_json_file, read_text_file
import json
import argparse
import gc


def read_txt_gz_or_json(fp):
    if fp.endswith('.txt.gz'):
        return read_txt_gz_file(fp)
    elif fp.endswith('.json'):
        return read_json_file(fp)
    else:
        return read_text_file(fp)


def create_new_longtext_columns(db_manager, col_names):
    try:
        db_manager.add_columns(db_manager.cache_table, col_names,
                               ["LONGTEXT"] * len(col_names), ["NULL"] * len(col_names))
    except Exception:
        print('e')


def transfer_results(db_manager, file_manager, input_cols, output_cols, start=0, max_n=-1):
    create_new_longtext_columns(db_manager, output_cols)
    n_rows = db_manager.get_cache_count()
    if start >= n_rows:
        return
    counter = start
    batch_size = 100000
    if max_n == -1:
        final_index = n_rows
    else:
        final_index = min([start + max_n, n_rows])
    while counter < final_index:
        print(f'Retrieving from db: LIMIT {counter},{batch_size}')
        current_rows = db_manager.get_all_details(input_cols, start=counter,
                                                  limit=batch_size, do_date_sort=False)
        print('Processing')
        temp_counter = 0
        for id_token in current_rows:
            values_dict = dict()
            row = current_rows[id_token]
            for i in range(len(input_cols)):
                if row[input_cols[i]] is not None:
                    current_value = read_txt_gz_or_json(
                        file_manager.generate_filepath(row[input_cols[i]])
                    )
                    if not isinstance(current_value, str):
                        current_value = json.dumps(current_value)
                    current_value = escape_everything(current_value)
                    values_dict[output_cols[i]] = current_value
            if len(values_dict) > 0:
                db_manager.insert_or_update_details(id_token, values_dict)
            temp_counter += 1
            if temp_counter % 1000 == 0:
                print((counter + temp_counter))
        counter += batch_size
        gc.collect()


def transfer_ocr_results(start, max_n):
    transfer_results(SlideDBCachingManager(), VideoConfig(),
                     ['ocr_tesseract_token', 'ocr_google_1_token', 'ocr_google_2_token'],
                     ['ocr_tesseract_results', 'ocr_google_1_results', 'ocr_google_2_results'],
                     start, max_n)


def transfer_transcription_results(start, max_n):
    transfer_results(AudioDBCachingManager(), VideoConfig(),
                     ['transcript_token', 'subtitle_token'],
                     ['transcript_results', 'subtitle_results'],
                     start, max_n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slides', action='store_true',
                        help='Transfer the results of OCR on slides.')
    parser.add_argument('--audio', action='store_true',
                        help='Transfer the results of transcription on audio.')
    parser.add_argument('--start', type=int, help='Starting index', default=0)
    parser.add_argument('--max_n', type=int, help='Maximum number of rows to handle', default=-1)
    args = parser.parse_args()
    if args.slides:
        print('Starting copy for slide tables')
        transfer_ocr_results(args.start, args.max_n)
    if args.audio:
        print('Starting copy for audio tables')
        transfer_transcription_results(args.start, args.max_n)


if __name__ == '__main__':
    main()
