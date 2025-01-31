import sys

import ffmpeg
import langdetect
import pytesseract
from PIL import Image

from graphai.core.common.caching import (
    VideoConfig,
    VideoDBCachingManager,
    get_token_file_status,
    SlideDBCachingManager,
    AudioDBCachingManager
)
from graphai.core.common.common_utils import file_exists
from graphai.core.common.lookup import is_fingerprinted


def get_video_token_status(token):
    video_config = VideoConfig()
    video_db_manager = VideoDBCachingManager()
    try:
        active = get_token_file_status(token, video_config)
    except Exception:
        return None

    exists, has_fingerprint = is_fingerprinted(token, video_db_manager)
    fully_active = active and exists
    if fully_active:
        try:
            streams = get_available_streams(video_config.generate_filepath(token))
        except Exception as e:
            print(e, file=sys.stderr)
            return {
                'active': False,
                'fingerprinted': False,
                'streams': None
            }
        streams = [x for x in streams if x['codec_type'] == 'audio' or x['codec_type'] == 'video']
    else:
        streams = None
    return {
        'active': fully_active,
        'fingerprinted': has_fingerprint,
        'streams': streams
    }


def get_image_token_status(token):
    video_config = VideoConfig()
    image_db_manager = SlideDBCachingManager()
    try:
        active = get_token_file_status(token, video_config)
    except Exception:
        return None

    exists, has_fingerprint = is_fingerprinted(token, image_db_manager)
    return {
        'active': active and exists,
        'fingerprinted': has_fingerprint
    }


def get_audio_token_status(token):
    video_config = VideoConfig()
    audio_db_manager = AudioDBCachingManager()
    try:
        active = get_token_file_status(token, video_config)
    except Exception:
        return None

    exists, has_fingerprint = is_fingerprinted(token, audio_db_manager)
    return {
        'active': active and exists,
        'fingerprinted': has_fingerprint
    }


def perform_probe(input_filename_with_path):
    """
    Performs a probe using ffprobe
    Args:
        input_filename_with_path: Input file path

    Returns:
        Probe results, see ffprobe documentation
    """
    if not file_exists(input_filename_with_path):
        raise Exception(f'ffmpeg error: File {input_filename_with_path} does not exist')
    try:
        return ffmpeg.probe(input_filename_with_path, cmd='ffprobe')
    except Exception as e:
        raise e


def get_available_streams(input_filename_with_path):
    results = perform_probe(input_filename_with_path)
    return [
        {
            'codec_type': x['codec_type'],
            'codec_name': x.get('codec_name', None),
            'duration': float(x['duration']) if 'duration' in x else None,
            'bit_rate': int(x['bit_rate']) if 'bit_rate' in x else None,
            'sample_rate': int(x['sample_rate']) if 'sample_rate' in x else None,
            'resolution': f"{x.get('width', '')}*{x.get('height', '')}" if x['codec_type'] == 'video' else None
        }
        for x in results['streams']
    ]


def perform_tesseract_ocr(image_path, language=None):
    """
    Performs OCR on an image using Tesseract
    Args:
        image_path: Full path of the image file
        language: Language of the image file

    Returns:
        Extracted text
    """
    if language is None:
        language = 'enfr'
    if not file_exists(image_path):
        print(f'Error: File {image_path} does not exist')
        return None
    return pytesseract.image_to_string(Image.open(image_path),
                                       lang={'en': 'eng', 'fr': 'fra', 'enfr': 'eng+fra'}[language])


def detect_text_language(s):
    """
    Detects the language of the provided string
    Args:
        s: String to detect language for

    Returns:
        Language of the string
    """
    if s is None or s == '':
        return None
    try:
        return langdetect.detect(s)
    except langdetect.lang_detect_exception.LangDetectException:
        return None
