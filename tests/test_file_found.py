import os

from graphai.core.common.caching import VideoConfig
from graphai.core.image.image import break_pdf_into_images, extract_slide_text
from graphai.core.video.video import extract_audio, extract_and_sample_frames
from graphai.core.video.video import compute_video_fingerprint, ignore_slide_fingerprint_results_callback
from graphai.core.voice.transcribe import detect_language_retrieve_from_db_and_split, transcribe_audio_to_text


class FailingModel:
    def transcribe_audio_whisper(self, *_args, **_kwargs):
        raise AssertionError("transcribe model should not be called when file is missing")


def _assert_missing(token):
    file_manager = VideoConfig()
    assert not os.path.exists(file_manager.generate_filepath(token))
    return file_manager


def test_extract_audio_reports_missing_input_file():
    token = 'codex_missing_video_file_for_audio_test.mp4'
    file_manager = _assert_missing(token)

    result = extract_audio(token, file_manager)

    assert result['token'] is None
    assert result['fresh'] is False
    assert result['file_found'] is False


def test_extract_and_sample_frames_reports_missing_input_file():
    token = 'codex_missing_video_file_for_slides_test.mp4'
    file_manager = _assert_missing(token)

    result = extract_and_sample_frames(token, file_manager)

    assert result['result'] is None
    assert result['fresh'] is False
    assert result['file_found'] is False


def test_compute_video_fingerprint_reports_missing_input_file():
    token = 'codex_missing_video_file_for_fingerprint_test.mp4'
    file_manager = _assert_missing(token)

    result = compute_video_fingerprint({'token': token}, file_manager)

    assert result['result'] is None
    assert result['fresh'] is False
    assert result['file_found'] is False


def test_detect_language_split_reports_missing_input_file():
    token = 'codex_missing_audio_file_for_lang_test.ogg'
    file_manager = _assert_missing(token)

    result = detect_language_retrieve_from_db_and_split({'token': token}, file_manager)

    assert result['temp_tokens'] is None
    assert result['fresh'] is False
    assert result['file_found'] is False


def test_transcribe_reports_missing_input_file_without_calling_model():
    token = 'codex_missing_audio_file_for_transcribe_test.ogg'
    file_manager = _assert_missing(token)

    result = transcribe_audio_to_text({'token': token, 'language': 'en'}, FailingModel(), file_manager)

    assert result['transcript_results'] is None
    assert result['fresh'] is False
    assert result['file_found'] is False


def test_extract_slide_text_reports_missing_input_file():
    token = 'codex_missing_image_file_for_ocr_test.png'
    file_manager = _assert_missing(token)

    result = extract_slide_text(token, file_manager, method='tesseract')

    assert result['results'] is None
    assert result['fresh'] is False
    assert result['file_found'] is False


def test_break_pdf_into_images_reports_missing_input_file():
    token = 'codex_missing_pdf_file_for_ocr_test.pdf'
    file_manager = _assert_missing(token)

    result = break_pdf_into_images(token, file_manager)

    assert result['pages'] is None
    assert result['file_found'] is False


def test_ignore_slide_fingerprint_results_preserves_file_found_on_failure():
    result = ignore_slide_fingerprint_results_callback({
        'fp_results': {
            'original_results': {
                'slide_tokens': None,
                'fresh': False,
                'file_found': False,
            }
        }
    })

    assert result['slide_tokens'] is None
    assert result['fresh'] is False
    assert result['file_found'] is False
