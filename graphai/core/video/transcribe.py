import sys
import whisper
from multiprocessing import Lock

from graphai.core.common.common_utils import file_exists
from graphai.core.interfaces.config import config


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

    def load_model_whisper(self):
        """
        Lazy-loads a Whisper model into memory
        Args:
            model_type: Type of model, see Whisper docs for details
        Returns:
            Model object
        """
        with self.load_lock:
            # device=None ensures that the model will use CUDA if available and switch to CPUs otherwise.
            if self.model is None:
                print('Actually loading Whisper model...')
                self.model = whisper.load_model(self.model_type, device=None, in_memory=True,
                                                download_root=self.download_root)

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
