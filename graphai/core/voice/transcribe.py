import gc
import sys
from collections import Counter

import ffmpeg
import whisper
import time
import json
from multiprocessing import Lock

from graphai.core.common.caching import (
    AudioDBCachingManager,
    cache_lookup_generic,
    TEMP_SUBFOLDER, database_callback_generic
)
from graphai.core.common.common_utils import file_exists
from graphai.core.common.config import config

# 12 hours
WHISPER_UNLOAD_WAITING_PERIOD = 12 * 3600.0


class WhisperTranscriptionModel:
    def __init__(self):
        try:
            print("Reading whisper model type from config")
            self.model_type = config['whisper']['model_type']
        except Exception:
            print(
                "The whisper model type could not be found in the config file, "
                "using the 'medium' model type as the default. "
                "To use a different one, make sure to add a [whisper] section with the model_type parameter."
            )
            self.model_type = 'medium'

        try:
            print("Reading whisper model path from config")
            self.download_root = config['whisper']['model_path']
            if self.download_root == '':
                self.download_root = None
        except Exception:
            print(
                "The whisper dl path could not be found in the config file, using default (~/.cache/whisper). "
                "To use a different one, make sure to add a [whisper] section with the model_path parameter."
            )
            self.download_root = None

        # The actual Whisper model is lazy loaded in order not to load it twice (celery *and* gunicorn)
        self.model = None
        self.load_lock = Lock()
        self.last_model_use = time.time()

    def load_model_whisper(self):
        """
        Lazy-loads a Whisper model into memory
        """
        with self.load_lock:
            # device=None ensures that the model will use CUDA if available and switch to CPUs otherwise.
            if self.model is None:
                print('Actually loading Whisper model...')
                self.model = whisper.load_model(self.model_type, device=None, in_memory=True,
                                                download_root=self.download_root)
            self.last_model_use = time.time()

    def get_last_usage(self):
        return self.last_model_use

    def unload_model(self, unload_period=WHISPER_UNLOAD_WAITING_PERIOD):
        deleted_models = None
        with self.load_lock:
            if time.time() - self.get_last_usage() > unload_period:
                if self.model is None:
                    deleted_models = list()
                else:
                    self.model = None
                    gc.collect()
                    deleted_models = [self.model_type]
        return deleted_models

    def get_silence_thresholds(self, strict_silence=False):
        if strict_silence:
            if self.model_type == 'base':
                no_speech_threshold = 0.5
                logprob_threshold = -0.5
            else:
                no_speech_threshold = 0.5
                logprob_threshold = -0.45
        else:
            no_speech_threshold = 0.6
            logprob_threshold = -1
        return no_speech_threshold, logprob_threshold

    def detect_audio_segment_lang_whisper(self, input_filename_with_path):
        """
        Detects the language of an audio file using a 30-second sample
        Args:
            input_filename_with_path: Path to input file

        Returns:
            Highest-scoring language code (e.g. 'en')
        """
        self.load_model_whisper()
        audio = whisper.load_audio(input_filename_with_path)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        # detect the spoken language
        _, probs = self.model.detect_language(mel)
        return max(probs, key=probs.get)

    def transcribe_audio_whisper(self, input_filename_with_path, force_lang=None, verbose=False,
                                 strict_silence=False):
        """
        Transcribes an audio file using whisper
        Args:
            input_filename_with_path: Path to input file
            force_lang: Whether to explicitly feed the model the language of the audio.
                        None results in automatic detection.
            verbose: Verbosity of the transcription
            strict_silence: Whether silence detection is strict or lenient.
                            Affects the logprob and no speech thresholds.
        Returns:
            A dictionary with three keys: 'text' contains the full transcript, 'segments' contains a JSON-like dict of
            translated segments which can be used as subtitles, and 'language' which contains the language code.
        """
        self.load_model_whisper()
        if not file_exists(input_filename_with_path):
            print(f'File {input_filename_with_path} does not exist')
            return None
        if force_lang not in [None, 'en', 'fr', 'de', 'it']:
            force_lang = 'en'
        try:
            no_speech_threshold, logprob_threshold = self.get_silence_thresholds(strict_silence)
            # setting fp16 to True makes sure that the model uses GPUs if available (otherwise
            # Whisper automatically switches to fp32)
            if force_lang is None:
                result = self.model.transcribe(input_filename_with_path, verbose=verbose, fp16=True,
                                               no_speech_threshold=no_speech_threshold,
                                               logprob_threshold=logprob_threshold)
            else:
                result = self.model.transcribe(input_filename_with_path, verbose=verbose, language=force_lang,
                                               fp16=True, no_speech_threshold=no_speech_threshold,
                                               logprob_threshold=logprob_threshold)
            transcript_results = result['text']
            subtitle_results = result['segments']
            language_result = result['language']

            if strict_silence:
                subtitle_results = [
                    x for x in subtitle_results
                    if x['avg_logprob'] >= -1.0
                ]
                transcript_results = ''.join([x['text'] for x in subtitle_results])
            return {
                'text': transcript_results,
                'segments': subtitle_results,
                'language': language_result
            }
        except Exception as e:
            print(e, file=sys.stderr)
            return None


def extract_media_segment(input_filename_with_path, output_filename_with_path, output_token, start, length):
    """
    Extracts a segment of a given video or audio file indicated by the starting time and the length
    Args:
        input_filename_with_path: Full path of input file
        output_filename_with_path: Full path of output file
        output_token: Output token
        start: Starting timestamp
        length: Length of segment

    Returns:
        The output token if successful, None otherwise
    """
    if not file_exists(input_filename_with_path):
        print(f'ffmpeg error: File {input_filename_with_path} does not exist')
        return None
    try:
        err = ffmpeg.input(input_filename_with_path). \
            output(output_filename_with_path, c='copy', ss=start, t=length). \
            overwrite_output().run(capture_stdout=True)
    except Exception as e:
        print(e, file=sys.stderr)
        err = str(e)

    if file_exists(output_filename_with_path) and ('ffmpeg error' not in err):
        return output_token
    else:
        return None


def detect_language_retrieve_from_db_and_split(input_dict, file_manager, n_divs=5, segment_length=30):
    token = input_dict['token']
    db_manager = AudioDBCachingManager()
    existing = db_manager.get_details(token, ['duration'],
                                      using_most_similar=True)[0]
    if existing is None or existing['duration'] is None:
        # We need the duration of the file from the cache, so the task fails if the cache row doesn't exist
        return {
            'temp_tokens': None,
            'lang': None,
            'fresh': False
        }

    duration = existing['duration']
    input_filename_with_path = file_manager.generate_filepath(token)
    result_tokens = list()

    # Creating `n_divs` segments (of duration `length` each) of the audio file and saving them to the temp subfolder
    for i in range(n_divs):
        current_output_token = token + '_' + str(i) + '_temp.ogg'
        current_output_token_with_path = file_manager.generate_filepath(current_output_token,
                                                                        force_dir=TEMP_SUBFOLDER)
        current_result = extract_media_segment(
            input_filename_with_path, current_output_token_with_path, current_output_token,
            start=duration * i / n_divs, length=segment_length)
        if current_result is None:
            print('Unspecified error while creating temp files')
            return {
                'lang': None,
                'temp_tokens': None,
                'fresh': False
            }

        result_tokens.append(current_result)

    return {
        'lang': None,
        'temp_tokens': result_tokens,
        'fresh': True
    }


def detect_language_parallel(tokens_dict, i, model, file_manager):
    if not tokens_dict['fresh']:
        return {
            'lang': None,
            'fresh': False
        }

    current_token = tokens_dict['temp_tokens'][i]
    try:
        language = model.detect_audio_segment_lang_whisper(
            file_manager.generate_filepath(current_token, force_dir=TEMP_SUBFOLDER)
        )
    except Exception:
        return {
            'lang': None,
            'fresh': False
        }

    return {
        'lang': language,
        'fresh': True
    }


def detect_language_callback(token, results_list, force):
    if all([x['lang'] is not None for x in results_list]):
        # This indicates success (regardless of freshness)
        languages = [x['lang'] for x in results_list]
        most_common_lang = Counter(languages).most_common(1)[0][0]
        values_dict = {'language': most_common_lang}
        database_callback_generic(token, AudioDBCachingManager(), values_dict, force, use_closest_match=True)
        return {
            'token': token,
            'language': most_common_lang,
            'fresh': True
        }
    return {
        'token': None,
        'language': None,
        'fresh': False
    }


def transcribe_audio_to_text(input_dict, model, file_manager, strict_silence=False):
    token = input_dict['token']
    lang = input_dict['language']

    # If the token is null, it means that some error happened in the previous step (e.g. the file didn't exist
    # in language detection)
    if token is None:
        return {
            'transcript_results': None,
            'subtitle_results': None,
            'language': None,
            'fresh': False
        }

    result_dict = model.transcribe_audio_whisper(file_manager.generate_filepath(token),
                                                 force_lang=lang, verbose=True,
                                                 strict_silence=strict_silence)

    if result_dict is None:
        return {
            'transcript_results': None,
            'subtitle_results': None,
            'language': None,
            'fresh': False
        }

    transcript_results = result_dict['text']
    subtitle_results = result_dict['segments']
    subtitle_results = json.dumps(subtitle_results, ensure_ascii=False)
    language_result = result_dict['language']

    return {
        'transcript_results': transcript_results,
        'subtitle_results': subtitle_results,
        'language': language_result,
        'fresh': True
    }


def transcribe_callback(token, results, force):
    if results['fresh']:
        values_dict = {
            'transcript_results': results['transcript_results'],
            'subtitle_results': results['subtitle_results'],
            'language': results['language']
        }
        # use_closest_match is True because we need to insert the same values for closest token
        # if different from original token
        database_callback_generic(token, AudioDBCachingManager(), values_dict, force, use_closest_match=True)
    return results
