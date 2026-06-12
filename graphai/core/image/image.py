import json
from loguru import logger as sysmsg

from graphai.core.common import common_utils
from graphai.core.common.caching import (
    SlideDBCachingManager,
    VideoConfig,
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
    GeminiOCRModel,
    RCPOCRModel,
)
import pymupdf
from graphai.core.common.common_utils import (
    retrieve_generic_file_from_generic_url,
    generate_random_token,
    get_file_size,
    get_current_datetime,
    is_token,
    is_url,
    is_effective_url,
    file_exists,
    get_most_common_element
)
from itertools import chain


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
        sysmsg.error("[KNbs06RW] Error uploading image from file.")
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
            'fresh': False,
            'file_found': None
        }
    file_manager = VideoConfig()
    file_found = common_utils.get_token_file_found(token, file_manager)
    ocr_colnames = get_ocr_colnames(method)
    db_manager = SlideDBCachingManager()
    existing_list = db_manager.get_details(token, ocr_colnames + ['language'],
                                           using_most_similar=True)
    # Checking whether the token even exists
    if existing_list[0] is None:
        return {
            'results': None,
            'language': None,
            'fresh': False,
            'file_found': file_found
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
                'fresh': fresh,
                'file_found': file_found
            }
    return None


def break_pdf_into_images(token, file_manager):
    file_found = common_utils.get_token_file_found(token, file_manager)
    if file_found is False:
        pdf_path = file_manager.generate_filepath(token)
        print(f'Error: File {pdf_path} does not exist')
        return {
            'pages': None,
            'file_found': False,
        }
    pdf_path = file_manager.generate_filepath(token)
    output_filenames = list()
    with pymupdf.open(pdf_path) as pdf_doc:
        for page in pdf_doc:
            i = page.number
            img_dir = file_manager.generate_filepath(token.replace('.', '__') + f'/page_{i}.png')
            pix = page.get_pixmap()
            pix.save(img_dir)
            output_filenames.append({
                'page': i + 1,
                'filename': img_dir
            })
    return {
        'pages': output_filenames,
        'file_found': True,
    }


def perform_ocr(
    file_path,
    method="google",
    google_api_token=None,
    openai_api_token=None,
    gemini_api_token=None,
    rcp_api_token=None,
    model_type=None,
    enable_tikz=False,
):
    text = None

    if method == 'tesseract':
        text = perform_tesseract_ocr(file_path, language='enfr')

    elif method == 'google' and google_api_token:
        ocr_model = GoogleOCRModel(google_api_token)
        ocr_model.establish_connection()
        text1, text2 = ocr_model.perform_ocr(file_path)

        # Since DTD usually performs better, method #1 is our point of reference for langdetect
        text = text1

    else:
        ocr_model = None
        if method == 'openai' and openai_api_token:
            ocr_model = OpenAIOCRModel(openai_api_token)
        elif method == 'gemini' and gemini_api_token:
            ocr_model = GeminiOCRModel(gemini_api_token)
        elif method == 'rcp' and rcp_api_token:
            ocr_model = RCPOCRModel(rcp_api_token)

        if ocr_model:
            ocr_model.establish_connection()
            text = ocr_model.perform_ocr(file_path, model_type=model_type, enable_tikz=enable_tikz)

    if not text:
        text = ''

    return {
        'results': [{'method': get_ocr_colnames(method)[0], 'text': text}],
        'language': detect_text_language(text),
    }


def extract_slide_text(
    token,
    file_manager,
    method="google",
    google_api_token=None,
    openai_api_token=None,
    gemini_api_token=None,
    rcp_api_token=None,
    model_type=None,
    enable_tikz=False,
):
    # Return no results if not a token
    if not is_token(token):
        return {
            'results': None,
            'language': None,
            'fresh': False,
            'file_found': None
        }

    file_found = common_utils.get_token_file_found(token, file_manager)
    if file_found is False:
        return {
            'results': None,
            'language': None,
            'fresh': False,
            'file_found': False
        }

    # Perform OCR
    file_path = file_manager.generate_filepath(token)
    res = perform_ocr(
        file_path,
        method,
        google_api_token,
        openai_api_token,
        gemini_api_token,
        rcp_api_token,
        model_type,
        enable_tikz,
    )
    res["fresh"] = res["results"] is not None
    res["file_found"] = file_found

    return res


def extract_multi_image_text(
    page_and_filename,
    method="google",
    google_api_token=None,
    openai_api_token=None,
    gemini_api_token=None,
    rcp_api_token=None,
    model_type=None,
    enable_tikz=False,
):
    # Perform OCR on page
    result = perform_ocr(
        page_and_filename["filename"],
        method,
        google_api_token,
        openai_api_token,
        gemini_api_token,
        rcp_api_token,
        model_type,
        enable_tikz,
    )

    print(f"Performed OCR on page {page_and_filename['page']}. Result: {result}")

    # Build result and return it
    return {
        'result': {
            'page': page_and_filename['page'],
            'content': result['results'][0]['text']
        },
        'language': result['language'],
        'method': result['results'][0]['method'],
    }


def collect_multi_image_ocr(results, file_found=True):
    all_results = [result['result'] for result in results]
    language = get_most_common_element([result['language'] for result in results])
    method = get_most_common_element([result['method'] for result in results])
    return {
        'results': [{
            'text': json.dumps(all_results),
            'method': method
        }],
        'language': language,
        'fresh': all(result['content'] is not None for result in all_results),
        'file_found': file_found
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
