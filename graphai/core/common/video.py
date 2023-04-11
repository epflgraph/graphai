import json
import os
import random
import re
import sys
import time
from datetime import datetime
import glob
import math

import acoustid
import chromaprint
import ffmpeg
from PIL import Image
import pytesseract
import gzip
import numpy as np
import wget
import whisper
from fuzzywuzzy import fuzz
from google.cloud import storage, speech
from .caching import make_sure_path_exists


FRAME_FORMAT = 'frame-%06d.png'
OCR_FORMAT = 'ocr-%06d.txt.gz'


def file_exists(file_path):
    return os.path.exists(file_path)


def get_dir_files(root_dir):
    return [sub_path for sub_path in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, sub_path))]


def count_files(root_dir):
    count = 0
    for sub_path in os.listdir(root_dir):
        if os.path.isfile(os.path.join(root_dir, sub_path)):
            count += 1
    return count


def write_text_file(filename_with_path, contents):
    with open(filename_with_path, 'w') as f:
        f.write(contents)


def write_json_file(filename_with_path, d):
    with open(filename_with_path, 'w') as f:
        json.dump(d, f)


def read_text_file(filename_with_path):
    f=open(filename_with_path, 'r')
    result=f.read()
    f.close()
    return result


def read_json_file(filename_with_path):
    f=open(filename_with_path, 'r')
    result=json.load(f)
    f.close()
    return result


def generate_random_token():
    """
    Generates a random string using the current time and a random number to be used as a token.
    Returns:
        Random token
    """
    return ('%.06f' % time.time()).replace('.','') + '%08d'%random.randint(0,int(1e7))


def retrieve_file_from_url(url, output_filename_with_path, output_token):
    """
    Retrieves a file from a given URL using WGET and stores it locally.
    Args:
        url: the URL
        output_filename_with_path: Path of output file
        output_token: Token of output file

    Returns:
        Output token if successful, None otherwise
    """
    try:
        response = wget.download(url, output_filename_with_path)
    except Exception as e:
        print(e, file=sys.stderr)
        return None
    if file_exists(output_filename_with_path):
        return output_token
    else:
        return None


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
    return ffmpeg.probe(input_filename_with_path, cmd='ffprobe')


def md5_video_or_audio(input_filename_with_path, video=True):
    if not file_exists(input_filename_with_path):
        print(f'ffmpeg error: File {input_filename_with_path} does not exist')
        return None
    in_stream = ffmpeg.input(input_filename_with_path)
    if video:
        # video
        try:
            in_stream = in_stream.video
        except:
            print("No video found. If you're trying to hash an audio file, provide video=False.")
            return None
    else:
        # audio
        try:
            in_stream = in_stream.audio
        except:
            print("No audio found. If you're trying to has the audio track of a video file, "
                  "make sure your video has audio.")
            return None
    result, _ = ffmpeg.output(in_stream, 'pipe:', format='md5').run(capture_stdout=True)
    # The result looks like 'MD5=9735151f36a3e628b0816b1bba3b9640\n' so we clean it up
    return (result.decode('utf8').strip())[4:]


def detect_audio_format_and_duration(input_filename_with_path, input_token):
    """
    Detects the duration of the audio track of the provided video file and returns its name in ogg format
    Args:
        input_filename_with_path: Path of input file
        input_token: Token of input file

    Returns:
        Token of output file consisting of input token + audio format, plus audio duration.
    """
    try:
        probe_results = perform_probe(input_filename_with_path)
    except Exception as e:
        print(e, file=sys.stderr)
        return None
    audio_type = probe_results['streams'][1]['codec_name']
    output_suffix='_audio.ogg'
    output_token = input_token + output_suffix
    return output_token, float(probe_results['format']['duration'])


def extract_audio_from_video(input_filename_with_path, output_filename_with_path, output_token):
    """
    Extracts the audio track from a video.
    Args:
        input_filename_with_path: Path of input file
        output_filename_with_path: Path of output file
        output_token: Token of output file

    Returns:
        Output token if successful, None if not.
    """
    if not file_exists(input_filename_with_path):
        print(f'ffmpeg error: File {input_filename_with_path} does not exist')
        return None
    try:
        err = ffmpeg.input(input_filename_with_path).audio. \
            output(output_filename_with_path, acodec='libopus', ar=48000).\
            overwrite_output().run(capture_stdout=True)
    except Exception as e:
        print(e, file=sys.stderr)
        err = str(e)

    if file_exists(output_filename_with_path) and ('ffmpeg error' not in err):
        return output_token
    else:
        return None


def extract_audio_segment(input_filename_with_path, output_filename_with_path, output_token, start, length):
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


def find_beginning_and_ending_silences(input_filename_with_path, distance_from_end_tol=0.01, noise_thresh=0.0001):
    """
    Detects silence at the beginning and the end of an audio file
    Args:
        input_filename_with_path: Path of input file
        distance_from_end_tol: Tolerance value for distance of the silence from each end of the file in seconds
        noise_thresh: Noise threshold

    Returns:
        A dictionary with the beginning and ending timestamps of the file with the two silences removed.
    """
    if not file_exists(input_filename_with_path):
        raise Exception(f'ffmpeg error: File {input_filename_with_path} does not exist')
    _, results = ffmpeg.input(input_filename_with_path).filter('silencedetect', n=noise_thresh).\
                output('pipe:', format='null').run(capture_stderr=True)
    results = results.decode('utf8')
    # the audio length will be accurate since the filter forces a decoding of the audio file
    audio_length = re.findall(r'time=\d{2}:\d{2}:\d{2}.\d+', results)
    audio_length = [datetime.strptime(x.strip('time=').strip(), '%H:%M:%S.%f') for x in audio_length]
    audio_length = max([t.hour*3600+t.minute*60+t.second+t.microsecond/1e6 for t in audio_length])
    results = re.split(r'[\n\r]', results)
    results = [x for x in results if '[silencedetect' in x]
    results = [x.split(']')[1].strip() for x in results]
    if len(results) > 0:
        silence_beginnings_and_ends = [float(result.split('|')[0].split(':')[1].strip()) for result in results]
        if len(results) == 2:
            silence_start = silence_beginnings_and_ends[0]
            silence_end = silence_beginnings_and_ends[1]
            if silence_start <= distance_from_end_tol:
                return_dict = {'ss': silence_end, 'to': audio_length}
            elif silence_end >= audio_length - distance_from_end_tol:
                return_dict = {'ss': 0, 'to': silence_start}
            else:
                return_dict = {'ss': 0, 'to': audio_length}
        else:
            first_silence_start = silence_beginnings_and_ends[0]
            first_silence_end = silence_beginnings_and_ends[1]
            last_silence_start = silence_beginnings_and_ends[-2]
            last_silence_end = silence_beginnings_and_ends[-1]
            return_dict = dict()
            if first_silence_start <= distance_from_end_tol:
                return_dict['ss'] = first_silence_end
            else:
                return_dict['ss'] = 0
            if last_silence_end >= audio_length - distance_from_end_tol:
                return_dict['to'] = last_silence_start
            else:
                return_dict['to'] = audio_length
    else:
        return_dict = {'ss': 0, 'to': audio_length}
    return return_dict


def remove_silence_doublesided(input_filename_with_path, output_filename_with_path, output_token,
                               threshold=0.0001):
    """
    Removes silence from the beginning and the end of the provided file.
    Args:
        input_filename_with_path: Path of the input file
        output_filename_with_path: Path of the output file
        output_token: Token of the output file (filename without path)
        threshold: Noise threshold for silence detection

    Returns:
        Token of the resulting audio file and its duration if successful, None and 0 if unsuccessful
    """
    try:
        from_and_to = find_beginning_and_ending_silences(input_filename_with_path, noise_thresh=threshold)
        err = ffmpeg.input(input_filename_with_path).audio. \
            output(output_filename_with_path, c='copy', ss=from_and_to['ss'], to=from_and_to['to']).\
            overwrite_output().run(capture_stdout=True)
    except Exception as e:
        print(e, file=sys.stderr)
        from_and_to = {'ss': 0, 'to': 0}
        err = str(e)

    if file_exists(output_filename_with_path) and ('ffmpeg error' not in err):
        return output_token, from_and_to['to'] - from_and_to['ss']
    else:
        return None, 0.0


def perceptual_hash_audio(input_filename_with_path, max_length=7200):
    """
    Computes the perceptual hash of an audio file
    Args:
        input_filename_with_path: Path of the input file
        max_length: Maximum length of the file in seconds

    Returns:
        String representation of the computed fingerprint and its decoded representation. Both are None if the
        file doesn't exist.
    """
    if not file_exists(input_filename_with_path):
        print(f'File {input_filename_with_path} does not exist')
        return None, None
    results = acoustid.fingerprint_file(input_filename_with_path, maxlength=max_length)
    fingerprint = results[1]
    decoded = chromaprint.decode_fingerprint(fingerprint)
    return fingerprint.decode('utf8'), decoded


def compare_audio_fingerprints(decoded_1, decoded_2):
    """
    Compares two decoded fingerprints
    Args:
        decoded_1: Fingerprint 1
        decoded_2: Fingerprint 2

    Returns:
        Fuzzy matching ratio between 0 and 1
    """
    return fuzz.ratio(decoded_1, decoded_2) / 100


def compare_encoded_audio_fingerprints(f1, f2=None):
    """
    Compares two string-encoded audio fingerprints
    and returns the ratio of the fuzzy match between them
    (value between 0 and 1, with 1 indicating an exact match).
    Args:
        f1: The target fingerprint
        f2: The second fingerprint, can be None (similarity is 0 if so)

    Returns:
        Ratio of fuzzy match between the two fingerprints
    """
    # when fuzzywuzzy is used in combination with python-Levenshtein (fuzzywuzzy[speedup],
    # there's a 10-fold speedup here.
    if f2 is None:
        return 0
    return compare_audio_fingerprints(chromaprint.decode_fingerprint(f1.encode('utf8')),
                                        chromaprint.decode_fingerprint(f2.encode('utf8')))


def find_closest_audio_fingerprint(target_fp, fp_list, token_list, min_similarity=0.8):
    """
    Given a target fingerprint and a list of candidate fingerprints, finds the one with the highest similarity
    to the target whose similarity is above a minimum value.
    Args:
        target_fp: Target fingerprint
        fp_list: List of candidate fingerprints
        token_list: List of tokens corresponding to those fingerprints
        min_similarity: Minimum similarity value. If the similarity of the most similar candidate to the target
                        is lower than this value, None will be returned as the result.

    Returns:
        Closest fingerprint, its token, and the highest score. All three are None if the closest one does not
        satisfy the minimum similarity criterion.
    """
    # If the list of fingerprints is empty, there's no "closest" fingerprint to the target and the result is null.
    if len(fp_list) == 0:
        return None, None, None
    fp_similarities = np.array([compare_encoded_audio_fingerprints(target_fp, fp2) for fp2 in fp_list])
    max_index = np.argmax(fp_similarities)
    # If the similarity of the most similar fingerprint is also greater than the minimum similarity value,
    # then it's a match. Otherwise, the result is null.
    if fp_similarities[max_index] >= min_similarity:
        return token_list[max_index], fp_list[max_index], fp_similarities[max_index]
    else:
        return None, None, None


def generate_gcp_uri(bucket_name, blob_name):
    return f'gs://{bucket_name}/{blob_name}'


def upload_file_to_google_cloud(input_filename_with_path, input_token, bucket_name='epflgraph_bucket'):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(input_token)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = 0

    blob.upload_from_filename(input_filename_with_path, if_generation_match=generation_match_precondition)

    print(
        f"File {input_filename_with_path} uploaded to {generate_gcp_uri(bucket_name, input_token)}."
    )
    return True


def transcribe_gcs(bucket_name, input_token, sample_rate=48000, timeout=600, lang="en-US"):
    """Asynchronously transcribes the audio file specified by the gcs_uri."""

    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=generate_gcp_uri(bucket_name, input_token))
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
        sample_rate_hertz=sample_rate,
        language_code=lang
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
    response = operation.result(timeout=timeout)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    all_transcripts = list()
    all_confidences = list()
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        all_transcripts.append(result.alternatives[0].transcript)
        all_confidences.append(result.alternatives[0].confidence)
    return all_transcripts, all_confidences, lang


def extract_frames(input_filename_with_path, output_folder_with_path, output_folder, frame_format=FRAME_FORMAT):
    if not file_exists(input_filename_with_path):
        print(f'ffmpeg error: File {input_filename_with_path} does not exist')
        return None
    try:
        make_sure_path_exists(output_folder_with_path)
        err = ffmpeg.input(input_filename_with_path).video. \
            output(os.path.join(output_folder_with_path, frame_format), r=1). \
            overwrite_output().run(capture_stdout=True)
    except Exception as e:
        print(e, file=sys.stderr)
        err = str(e)

    if file_exists(os.path.join(output_folder_with_path, (frame_format) % (1))) and ('ffmpeg error' not in err):
        return output_folder
    else:
        return None


def generate_frame_sample_indices(input_folder_with_path, step=16):
    # Get number of frames
    n_frames = len(glob.glob(os.path.join(input_folder_with_path+'/*.png')))
    # Generate list of frame sample indices
    frame_sample_indices = list(np.arange(1, n_frames-step+1, step))+[n_frames-1]
    # Return frame sample indices
    return frame_sample_indices


class WhisperTranscriptionModel():
    def __init__(self, model_type='medium'):
        # The actual Whisper model is lazy loaded in order not to load it twice (celery *and* gunicorn)
        self.model_type = model_type
        self.model = None


    def load_model_whisper(self):
        """
        Loads a Whisper model into memory
        Args:
            model_type: Type of model, see Whisper docs for details

        Returns:
            Model object
        """
        # device=None ensures that the model will use CUDA if available and switch to CPUs otherwise.
        if self.model is None:
            self.model = whisper.load_model(self.model_type, device=None, in_memory=True)


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


    def transcribe_audio_whisper(self, input_filename_with_path, force_lang=None, verbose=False):
        """
        Transcribes an audio file using whisper
        Args:
            input_filename_with_path: Path to input file
            force_lang: Whether to explicitly feed the model the language of the audio.
                        None results in automatic detection.
            verbose: Verbosity of the transcription

        Returns:
            A dictionary with three keys: 'text' contains the full transcript, 'segments' contains a JSON-like dict of
            translated segments which can be used as subtitles, and 'language' which contains the language code.
        """
        self.load_model_whisper()
        if not file_exists(input_filename_with_path):
            print(f'File {input_filename_with_path} does not exist')
            return None
        assert force_lang in [None, 'en', 'fr', 'de', 'it']
        try:
            # setting fp16 to True makes sure that the model uses GPUs if available (otherwise
            # Whisper automatically switches to fp32)
            if force_lang is None:
                result = self.model.transcribe(input_filename_with_path, verbose=verbose, fp16=True)
            else:
                result = self.model.transcribe(input_filename_with_path, verbose=verbose,
                                          language=force_lang, fp16=True)
        except Exception as e:
            print(e, file=sys.stderr)
            return None
        return result
