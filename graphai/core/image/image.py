from graphai.core.common.caching import (
    SlideDBCachingManager,
    write_binary_file_to_token
)
from graphai.core.common.lookup import database_callback_generic
from graphai.core.common.multimedia_utils import (
    get_image_token_status,
    perform_tesseract_ocr,
    detect_text_language
)
from graphai.core.image.ocr import (
    get_ocr_colnames,
    GoogleOCRModel,
    OpenAIOCRModel,
    perform_tesseract_ocr_on_pdf
)
from graphai.core.common.common_utils import (
    retrieve_generic_file_from_generic_url,
    generate_random_token,
    get_file_size,
    get_current_datetime,
    is_token,
    is_url,
    is_effective_url,
    is_pdf
)


def create_image_filename_using_url_format(token, url):
    file_format = url.split('.')[-1].lower()
    if file_format not in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'pdf']:
        return None
    filename = token + '.' + file_format
    return filename


def create_origin_token_using_info(origin, origin_info):
    return f"{origin}://{origin_info['id']}__{origin_info['name']}"


def cache_lookup_retrieve_image_from_url(url, file_manager):
    if not is_effective_url(url):
        return None
    db_manager = SlideDBCachingManager()
    existing = db_manager.get_details_using_origin(url, [])
    if existing is not None:
        token = existing[0]['id_token']
        return {
            'token': token,
            'fresh': False,
            'token_status': get_image_token_status(token),
            'token_size': get_file_size(file_manager.generate_filepath(token))
        }
    return None


def retrieve_image_file_from_url(url, file_manager, force_token=None):
    if not is_url(url):
        return {
            'token': None,
            'fresh': False,
            'token_size': None,
        }
    if force_token is not None:
        token = force_token
    else:
        db_manager = SlideDBCachingManager()
        existing = db_manager.get_details_using_origin(url, [])
        if existing is not None:
            # If the cache row already exists, then we don't create a new token, but instead
            # use the id_token of the existing row (we remove the file extension because it will be re-added soon)
            token = existing[0]['id_token'].split('.')[0]
        else:
            # Otherwise, we generate a random token
            token = generate_random_token()
    filename = create_image_filename_using_url_format(token, url)
    if filename is None:
        return {
            'token': None,
            'fresh': False,
            'token_size': None,
        }
    filename_with_path = file_manager.generate_filepath(filename)
    results = retrieve_generic_file_from_generic_url(url, filename_with_path, filename)
    return {
        'token': results,
        'fresh': results == filename,
        'token_size': get_file_size(filename_with_path),
    }


def retrieve_image_file_from_url_callback(results, url):
    if results['fresh']:
        db_manager = SlideDBCachingManager()
        current_datetime = get_current_datetime()
        values = {
            'origin_token': url
        }
        if db_manager.get_details(results['token'], [], using_most_similar=False)[0] is None:
            # If the row doesn't already exist in the database, we also set its date_added value
            values.update(
                {
                    'date_added': current_datetime
                }
            )
        database_callback_generic(results['token'], db_manager, values, use_closest_match=False)
    return results


def upload_image_from_file(contents, file_extension, file_manager):
    token = generate_random_token()
    filename = token + '.' + file_extension
    try:
        filename_with_path = write_binary_file_to_token(contents, filename, file_manager)
        return {
            'token': filename,
            'fresh': True,
            'token_size': get_file_size(filename_with_path)
        }
    except Exception as e:
        print(e)
        return {
            'token': None,
            'error': str(e),
            'fresh': False,
            'token_size': None
        }


def cache_lookup_extract_slide_text(token, method):
    if not is_token(token):
        return {
            'results': None,
            'language': None,
            'fresh': False
        }
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


def extract_slide_text(token, file_manager, method='google', api_token=None, openai_token=None, pdf_in_pages=True):
    if not is_token(token):
        return {
            'results': None,
            'language': None,
            'fresh': False
        }
    ocr_colnames = get_ocr_colnames(method)

    if method == 'tesseract':
        if is_pdf(token):
            res = perform_tesseract_ocr_on_pdf(file_manager.generate_filepath(token),
                                               language='enfr', in_pages=pdf_in_pages)
        else:
            res = perform_tesseract_ocr(file_manager.generate_filepath(token), language='enfr')
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
        # We do not provide Google Cloud Vision or OpenAI OCR for PDFs.
        if is_pdf(token):
            return {
                'results': [
                    {
                        'method': method,
                        'text': 'Currently, only Tesseract OCR is available for PDFs.',
                        'fail': True
                    }
                ],
                'language': None,
                'fresh': False
            }
        if method == 'google':
            # Google OCR
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
        else:
            # OpenAI OCR
            if openai_token is None:
                results = None
                language = None
            else:
                ocr_model = OpenAIOCRModel(openai_token)
                ocr_model.establish_connection()
                res = ocr_model.perform_ocr(file_manager.generate_filepath(token))

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
