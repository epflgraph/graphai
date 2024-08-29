from graphai.core.common.caching import (
    TextDBCachingManager,
    database_callback_generic, fingerprint_based_text_lookup, token_based_text_lookup
)
from graphai.core.translation.text_utils import convert_text_back_to_list, detect_text_language
from graphai.core.common.common_utils import get_current_datetime, convert_text_back_to_list

LONG_TEXT_ERROR = "Unpunctuated text too long (over 512 tokens), " \
                  "try adding punctuation or providing a smaller chunk of text."


def translate_text(text, src, tgt, translation_obj):
    if src == tgt:
        return {
            'result': "'source' and 'target' languages must be different!",
            'text_too_large': False,
            'successful': False,
            'fresh': False,
            'device': None
        }

    how = f"{src}-{tgt}"
    try:
        translated_text, large_warning, all_large_warnings = translation_obj.translate(text, how=how)
        if translated_text is not None and not large_warning:
            success = True
        else:
            success = False
        if large_warning:
            large_warning_indices = [str(i) for i in range(len(all_large_warnings)) if all_large_warnings[i]]
            translated_text = (
                LONG_TEXT_ERROR
                + f"This happened for inputs at indices {', '.join(large_warning_indices)}."
            )
    except NotImplementedError as e:
        print(e)
        translated_text = str(e)
        success = False
        large_warning = False

    return {
        'result': translated_text,
        'text_too_large': large_warning,
        'successful': success,
        'fresh': success,
        'device': translation_obj.get_device()
    }


def translate_text_callback(results, token, text, src, tgt, force=False, return_list=False):
    db_manager = TextDBCachingManager()
    if results['fresh']:
        values_dict = {
            'source': text,
            'target': results['result'],
            'source_lang': src,
            'target_lang': tgt
        }
        existing = db_manager.get_details(token, ['date_added'], using_most_similar=False)[0]
        if existing is None or existing['date_added'] is None:
            values_dict['date_added'] = get_current_datetime()
        database_callback_generic(token, db_manager, values_dict, force, True)
    elif not results['successful']:
        # in case we fingerprinted something and then failed to translate it, we delete its cache row
        db_manager.delete_cache_rows([token])

    # If the computation was successful and return_list is True, we want to convert the text results
    # back to a list (because this flag means that the original input was a list of strings)
    if results['successful']:
        results['result'] = convert_text_back_to_list(results['result'], return_list=return_list)
    return results


def compute_translation_text_fingerprint_callback(results, text, src, tgt):
    # This task does not have the condition of the 'fresh' flag being True because text fingerprinting never fails
    fp = results['result']
    token = results['token']
    db_manager = TextDBCachingManager()
    values_dict = {
        'fingerprint': fp,
        'source': text,
        'source_lang': src,
        'target_lang': tgt,
        'date_added': get_current_datetime()
    }
    db_manager.insert_or_update_details(token, values_dict)
    return results


def cache_lookup_translation_text_using_fingerprint(token, fp, src, tgt, return_list):
    return fingerprint_based_text_lookup(
        token, fp, TextDBCachingManager(), main_col='target', extra_cols=[],
        equality_conditions={'source_lang': src, 'target_lang': tgt},
        modify_result_func=convert_text_back_to_list,
        modify_result_args={'return_list': return_list}
    )


def cache_lookup_translate_text(token, return_list):
    return token_based_text_lookup(token, TextDBCachingManager(), 'target',
                                   modify_result_func=convert_text_back_to_list,
                                   modify_result_args={'return_list': return_list})


def detect_language_translation(text):
    result = detect_text_language(text)
    return {
        'language': result,
        'successful': result is not None
    }
