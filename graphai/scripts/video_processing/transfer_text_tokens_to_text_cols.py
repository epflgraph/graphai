from graphai.core.common.caching import SlideDBCachingManager, AudioDBCachingManager, VideoConfig
from graphai.core.common.video import read_txt_gz_file
from graphai.core.common.common_utils import read_json_file, read_text_file
import json


def read_txt_gz_or_json(fp):
    if fp.endswith('.txt.gz'):
        return read_txt_gz_file(fp)
    elif fp.endswith('.json'):
        return read_json_file(fp)
    else:
        return read_text_file(fp)


def create_new_longtext_columns(db_manager, col_names):
    try:
        db_manager.add_columns(db_manager.cache_table, col_names, ["LONGTEXT"]*len(col_names), ["NULL"]*len(col_names))
    except Exception as e:
        print('e')


def transfer_results(db_manager, file_manager, input_cols, output_cols):
    create_new_longtext_columns(db_manager, output_cols)
    n_rows = db_manager.get_cache_count()
    counter = 0
    batch_size = 1000
    while counter < n_rows:
        current_rows = db_manager.get_all_details(input_cols, start=counter,
                                                  limit=batch_size)
        for id_token in current_rows:
            values_dict = dict()
            row = current_rows[id_token]
            print(row)
            for i in range(len(input_cols)):
                print(input_cols[i])
                if row[input_cols[i]] is not None:
                    current_value = read_txt_gz_or_json(
                        file_manager.generate_filepath(row[input_cols[i]])
                    )
                    if not isinstance(current_value, str):
                        current_value = json.dumps(current_value)
                    values_dict[output_cols[i]] = current_value
            if len(values_dict) > 0:
                db_manager.insert_or_update_details(id_token, values_dict)
        counter += batch_size
        print(counter)


def transfer_ocr_results():
    transfer_results(SlideDBCachingManager(), VideoConfig(),
                     ['ocr_tesseract_token', 'ocr_google_1_token', 'ocr_google_2_token'],
                     ['ocr_tesseract_results', 'ocr_google_1_results', 'ocr_google_2_results'])


def transfer_transcription_results():
    transfer_results(AudioDBCachingManager(), VideoConfig(),
                     ['transcript_token', 'subtitle_token'],
                     ['transcript_results', 'subtitle_results'])


def main():
    print('Starting copy for slide tables')
    transfer_ocr_results()
    print('Starting copy for audio tables')
    transfer_transcription_results()


if __name__ == '__main__':
    main()
