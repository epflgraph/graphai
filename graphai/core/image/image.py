from graphai.core.common.caching import SlideDBCachingManager
from graphai.core.common.lookup import database_callback_generic
from graphai.core.image.ocr import get_ocr_colnames, GoogleOCRModel
from graphai.core.translation.text_utils import detect_text_language
from graphai.core.video.video import perform_tesseract_ocr


def cache_lookup_extract_slide_text(token, method):
    ocr_colnames = get_ocr_colnames(method)
    db_manager = SlideDBCachingManager()
    existing_list = db_manager.get_details(token, ocr_colnames + ['language'],
                                           using_most_similar=True)
    # Checking whether the token even exists
    if existing_list[0] is None:
        return {
            'results': None,
            'language': None,
            'fresh': False
        }
    for existing in existing_list:
        if existing is None:
            continue

        if all([existing[ocr_colname] is not None for ocr_colname in ocr_colnames]):
            print('Returning cached result')
            results = [
                {
                    'method': ocr_colname,
                    'text': existing[ocr_colname],
                }
                for ocr_colname in ocr_colnames
            ]
            language = existing['language']
            fresh = False

            if language is None:
                language = detect_text_language(results[0]['text'])
                fresh = True

            return {
                'results': results,
                'language': language,
                'fresh': fresh
            }
    return None


def extract_slide_text(token, file_manager, method='google', api_token=None):
    ocr_colnames = get_ocr_colnames(method)

    if method == 'tesseract':
        res = perform_tesseract_ocr(file_manager.generate_filepath(token))
        if res is None:
            results = None
            language = None
        else:
            language = detect_text_language(res)
            results = [
                {
                    'method': ocr_colnames[0],
                    'text': res
                }
            ]
    else:
        if api_token is None:
            results = None
            language = None
        else:
            ocr_model = GoogleOCRModel(api_token)
            ocr_model.establish_connection()
            res1, res2 = ocr_model.perform_ocr(file_manager.generate_filepath(token))

            if res1 is None or res2 is None:
                results = None
                language = None
            else:
                # Since DTD usually performs better, method #1 is our point of reference for langdetect
                language = detect_text_language(res1)
                res_list = [res1, res2]
                results = [
                    {
                        'method': ocr_colnames[i],
                        'text': res_list[i]
                    }
                    for i in range(len(res_list))
                ]

    return {
        'results': results,
        'language': language,
        'fresh': results is not None
    }


def extract_slide_text_callback(results, token, force=False):
    if results['fresh']:
        values_dict = {
            result['method']: result['text']
            for result in results['results']
        }
        values_dict['language'] = results['language']
        database_callback_generic(token, SlideDBCachingManager(), values_dict, force, use_closest_match=True)
    return results
