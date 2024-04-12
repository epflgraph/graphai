import os
import sys
import io
import math
import glob
import re
import random
import json
import time
from datetime import datetime, timedelta
import gzip

from sacremoses.tokenize import MosesTokenizer
import wget
import subprocess

import numpy as np

import acoustid
import ffmpeg
import imagehash
from PIL import Image
from fuzzywuzzy import fuzz

import pytesseract
from google.cloud import vision
import whisper

import fasttext
from fasttext_reducer.reduce_fasttext_models import generate_target_path

from graphai.core.common.config import config
from graphai.core.common.common_utils import make_sure_path_exists, file_exists
from graphai.core.common.text_utils import perceptual_hash_text

FRAME_FORMAT_PNG = 'frame-%06d.png'
FRAME_FORMAT_JPG = 'frame-%06d.jpg'
TESSERACT_OCR_FORMAT = 'ocr-%06d.txt.gz'

PUNKT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
STOPWORDS = {
    'fr': ['au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'elle', 'en', 'et', 'eux', 'il', 'ils', 'je',
           'la', 'le', 'les', 'leur', 'lui', 'ma', 'mais', 'me', 'même', 'mes', 'moi', 'mon', 'ne', 'nos', 'notre',
           'nous', 'on', 'ou', 'par', 'pas', 'pour', 'qu', 'que', 'qui', 'sa', 'se', 'ses', 'son', 'sur', 'ta', 'te',
           'tes', 'toi', 'ton', 'tu', 'un', 'une', 'vos', 'votre', 'vous', 'c', 'd', 'j', 'l', 'à', 'm', 'n', 's', 't',
           'y', 'été', 'étée', 'étées', 'étés', 'étant', 'étante', 'étants', 'étantes', 'suis', 'es', 'est', 'sommes',
           'êtes', 'sont', 'serai', 'seras', 'sera', 'serons', 'serez', 'seront', 'serais', 'serait', 'serions',
           'seriez', 'seraient', 'étais', 'était', 'étions', 'étiez', 'étaient', 'fus', 'fut', 'fûmes', 'fûtes',
           'furent', 'sois', 'soit', 'soyons', 'soyez', 'soient', 'fusse', 'fusses', 'fût', 'fussions', 'fussiez',
           'fussent', 'ayant', 'ayante', 'ayantes', 'ayants', 'eu', 'eue', 'eues', 'eus', 'ai', 'as', 'avons', 'avez',
           'ont', 'aurai', 'auras', 'aura', 'aurons', 'aurez', 'auront', 'aurais', 'aurait', 'aurions', 'auriez',
           'auraient', 'avais', 'avait', 'avions', 'aviez', 'avaient', 'eut', 'eûmes', 'eûtes', 'eurent', 'aie',
           'aies', 'ait', 'ayons', 'ayez', 'aient', 'eusse', 'eusses', 'eût', 'eussions', 'eussiez', 'eussent'],
    'en': ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
           'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
           'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
           'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
           'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
           'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
           'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
           'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
           'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
           'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
           "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
           'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
           "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
           "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
           "wouldn't"]
}


def generate_random_token():
    """
    Generates a random string using the current time and a random number to be used as a token.
    Returns:
        Random token
    """
    return ('%.06f' % time.time()).replace('.', '') + '%08d' % random.randint(0, int(1e7))


def retrieve_file_from_generic_url(url, output_filename_with_path, output_token):
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
        wget.download(url, output_filename_with_path)
    except Exception as e:
        print(e, file=sys.stderr)
        return None
    if file_exists(output_filename_with_path):
        return output_token
    else:
        return None


def retrieve_file_from_kaltura(url, output_filename_with_path, output_token):
    """
    Retrieves a file in m3u8 format from Kaltura and stores it locally
    Args:
        url: URL of the m3u8 playlist
        output_filename_with_path: Full path of output file
        output_token: Token of output file

    Returns:
        Output token if retrieval is successful, None otherwise
    """
    # Downloading using ffmpeg
    try:
        err = ffmpeg.input(url, protocol_whitelist="file,http,https,tcp,tls,crypto"). \
            output(output_filename_with_path, c="copy"). \
            overwrite_output().run(capture_stdout=True)
        err = str(err[0])
    except Exception as e:
        print(e, file=sys.stderr)
        err = str(e)
    # If the file exists and there were no errors, the download has been successful
    if file_exists(output_filename_with_path) and ('ffmpeg error' not in err.lower()):
        return output_token
    else:
        return None


def retrieve_file_from_youtube(url, output_filename_with_path, output_token):
    """
    Downloads a video from YouTube
    Args:
        url: Youtube URL
        output_filename_with_path: Full path of output file
        output_token: Token of output file

    Returns:
        Token of output file if successful, None otherwise
    """
    cmd_str = f"yt-dlp -o '{output_filename_with_path}' -f '[ext=mp4]' {url}"
    result_code = subprocess.run(cmd_str, shell=True)
    if file_exists(output_filename_with_path) and result_code.returncode == 0:
        return output_token
    else:
        return None


def retrieve_file_from_url(url, output_filename_with_path, output_token):
    if 'youtube.com/' in url or 'youtu.be/' in url:
        return retrieve_file_from_youtube(url, output_filename_with_path, output_token)
    else:
        return retrieve_file_from_generic_url(url, output_filename_with_path, output_token)


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


def perform_slow_audio_probe(input_filename_with_path):
    """
        Performs a slower probe using ffmpeg by decoding the audio stream
        Args:
            input_filename_with_path: Input file path

        Returns:
            Probe results
        """
    if not file_exists(input_filename_with_path):
        raise Exception(f'ffmpeg error: File {input_filename_with_path} does not exist')
    results = ffmpeg.input(input_filename_with_path).audio.output('pipe:', format='null').run(capture_stderr=True)
    all_matches = re.findall(r"time=\d{2}:\d{2}:\d{2}\.\d{2}", str(results[1]))
    final_time_str = all_matches[-1][5:]
    final_time_parsed = time.strptime(final_time_str, '%H:%M:%S.%f')
    return {
        'format': {
            'duration': timedelta(
                hours=final_time_parsed.tm_hour,
                minutes=final_time_parsed.tm_min,
                seconds=final_time_parsed.tm_sec
            ).total_seconds()
        }
    }


def generate_symbolic_token(origin, token):
    """
    Generates a new symbolic token based on the origin token and the token itself
    Args:
        origin: Origin token
        token: Target token

    Returns:
        Symbolic token
    """
    return origin + '_' + token


def md5_video_or_audio(input_filename_with_path, video=True):
    """
    Computes the md5 hash of the video or audio stream of a video file
    Args:
        input_filename_with_path: Full path of the input file
        video: Whether to compute the md5 for the video stream or the audio stream

    Returns:
        MD5 hash
    """
    if not file_exists(input_filename_with_path):
        print(f'ffmpeg error: File {input_filename_with_path} does not exist')
        return None
    in_stream = ffmpeg.input(input_filename_with_path)
    if video:
        # video
        try:
            in_stream = in_stream.video
        except Exception:
            print("No video found. If you're trying to hash an audio file, provide video=False.")
            return None
    else:
        # audio
        try:
            in_stream = in_stream.audio
        except Exception:
            print("No audio found. If you're trying to has the audio track of a video file, "
                  "make sure your video has audio.")
            return None
    try:
        result, _ = ffmpeg.output(
            in_stream, 'pipe:', c='copy', format='md5'
        ).run(capture_stdout=True)
    except Exception:
        print("An error occurred while fingerprinting")
        return None
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
        return None, None
    if probe_results.get('format', None) is None or probe_results['format'].get('duration', None) is None:
        try:
            probe_results = perform_slow_audio_probe(input_filename_with_path)
        except Exception as e:
            print(e, file=sys.stderr)
            return None, None
    output_suffix = '_audio.ogg'
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
            output(output_filename_with_path, acodec='libopus', ar=48000). \
            overwrite_output().run(capture_stdout=True)
    except Exception as e:
        print(e, file=sys.stderr)
        err = str(e)

    if file_exists(output_filename_with_path) and ('ffmpeg error' not in err):
        return output_token
    else:
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
    _, results = ffmpeg.input(input_filename_with_path)\
                       .filter('silencedetect', n=noise_thresh)\
                       .output('pipe:', format='null')\
                       .run(capture_stderr=True)
    results = results.decode('utf8')
    # the audio length will be accurate since the filter forces a decoding of the audio file
    audio_length = re.findall(r'time=\d{2}:\d{2}:\d{2}.\d+', results)
    audio_length = [datetime.strptime(x.strip('time=').strip(), '%H:%M:%S.%f') for x in audio_length]
    audio_length = max([t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6 for t in audio_length])
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
        audio_length = from_and_to['to'] - from_and_to['ss']
        output = extract_media_segment(input_filename_with_path, output_filename_with_path, output_token,
                                       start=from_and_to['ss'], length=audio_length)
    except Exception:
        output = None
        audio_length = 0.0

    if output is not None and file_exists(output_filename_with_path):
        return output, audio_length
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
        return None
    results = acoustid.fingerprint_file(input_filename_with_path, maxlength=max_length)
    fingerprint_value = results[1]
    return perceptual_hash_text(fingerprint_value.decode('utf8'))


def perceptual_hash_image(input_filename_with_path, hash_size=16):
    """
    Computes the perceptual hash of an image file
    Args:
        input_filename_with_path: Path of the input file
        hash_size: Size of hash
    Returns:
        String representation of the computed fingerprint. None if file does not exist
    """
    if not file_exists(input_filename_with_path):
        print(f'File {input_filename_with_path} does not exist')
        return None
    results = imagehash.dhash(Image.open(input_filename_with_path), hash_size=hash_size)
    return str(results)


def compare_decoded_fingerprints(decoded_1, decoded_2):
    """
    Compares two decoded fingerprints
    Args:
        decoded_1: Fingerprint 1
        decoded_2: Fingerprint 2

    Returns:
        Fuzzy matching ratio between 0 and 1
    """
    return fuzz.ratio(decoded_1, decoded_2) / 100


def compare_encoded_fingerprints(f1, f2=None, decoder_func=imagehash.hex_to_hash):
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
    return compare_decoded_fingerprints(decoder_func(f1.encode('utf8')),
                                        decoder_func(f2.encode('utf8')))


def find_closest_fingerprint_from_list(target_fp, fp_list, token_list, date_list, min_similarity=0.8,
                                       decoder_func=imagehash.hex_to_hash, strip_underscores=True):
    """
    Given a target fingerprint and a list of candidate fingerprints, finds the one with the highest similarity
    to the target whose similarity is above a minimum value.
    Args:
        target_fp: Target fingerprint
        fp_list: List of candidate fingerprints
        token_list: List of tokens corresponding to those fingerprints
        min_similarity: Minimum similarity value. If the similarity of the most similar candidate to the target
                        is lower than this value, None will be returned as the result.
        decoder_func: The function that decodes the string hash, different for audio vs image hashes

    Returns:
        Closest fingerprint, its token, and the highest score. All three are None if the closest one does not
        satisfy the minimum similarity criterion.
    """
    # If the list of fingerprints is empty, there's no "closest" fingerprint to the target and the result is null.
    if len(fp_list) == 0:
        return None, None, None, None
    if min_similarity < 1:
        if strip_underscores:
            trailing_underscores = '_'.join(target_fp.split('_')[1:])
            target_fp = target_fp.split('_')[0]
            fp_list = [x.split('_')[0] for x in fp_list if x.endswith(trailing_underscores)]
            if len(fp_list) == 0:
                return None, None, None, None
        fp_similarities = np.array([compare_encoded_fingerprints(target_fp, fp2, decoder_func) for fp2 in fp_list])
    else:
        # if an exact match is required, we switch to a much faster equality comparison
        fp_similarities = [1 if target_fp == fp2 else 0 for fp2 in fp_list]
    # The index returned by argmax is the first occurrence of the maximum
    max_index = np.argmax(fp_similarities)
    # If the similarity of the most similar fingerprint is also greater than the minimum similarity value,
    # then it's a match. Otherwise, the result is null.
    if fp_similarities[max_index] >= min_similarity:
        return token_list[max_index], fp_list[max_index], date_list[max_index], fp_similarities[max_index]
    else:
        return None, None, None, None


def find_closest_audio_fingerprint_from_list(target_fp, fp_list, token_list, date_list, min_similarity=0.8):
    """
    Finds closest audio fingerprint from list
    """
    return find_closest_fingerprint_from_list(target_fp, fp_list, token_list, date_list, min_similarity,
                                              decoder_func=imagehash.hex_to_hash)


def find_closest_image_fingerprint_from_list(target_fp, fp_list, token_list, date_list, min_similarity=0.8):
    """
    Finds closest image fingerprint from list
    """
    return find_closest_fingerprint_from_list(target_fp, fp_list, token_list, date_list, min_similarity,
                                              decoder_func=imagehash.hex_to_hash)


def extract_frames(input_filename_with_path, output_folder_with_path, output_folder):
    """
    Extracts frames from an image file
    Args:
        input_filename_with_path: Path of input video file
        output_folder_with_path: Path of output image folder
        output_folder: The output folder only (as return token)

    Returns:
        The return token if successful, None otherwise
    """
    if not file_exists(input_filename_with_path):
        print(f'ffmpeg error: File {input_filename_with_path} does not exist')
        return None
    try:
        print('Creating the path...')
        make_sure_path_exists(output_folder_with_path)
        # DO NOT CHANGE r=1 HERE
        # This parameter ensures that one frame is extracted per second, and the whole logic of the algorithm
        # relies on timestamp being identical to frame number.
        print('Starting ffmpeg slide extraction...')
        err = ffmpeg.input(input_filename_with_path).video. \
            filter("fps", 1).output(os.path.join(output_folder_with_path, FRAME_FORMAT_PNG)). \
            overwrite_output().run(capture_stdout=True)
    except Exception as e:
        print(e, file=sys.stderr)
        err = str(e)

    if file_exists(os.path.join(output_folder_with_path, (FRAME_FORMAT_PNG) % (1))) and ('ffmpeg error' not in err):
        return output_folder
    else:
        return None


def generate_frame_sample_indices(input_folder_with_path, step=12):
    """
    Generates indices for extracted frames (so we don't use every single frame for our calculations)
    Args:
        input_folder_with_path: Full path of the input image folder
        step: Step size for indices

    Returns:
        List of indices
    """
    # Get number of frames
    n_frames = len(glob.glob(os.path.join(input_folder_with_path + '/*.png')))
    # Generate list of frame sample indices
    frame_sample_indices = list(np.arange(1, n_frames - step + 1, step)) + [n_frames]
    # Return frame sample indices
    return frame_sample_indices


def read_txt_gz_file(fp):
    """
    Reads the contents of a txt.gz file
    Args:
        fp: File path

    Returns:
        Resulting text
    """
    with gzip.open(fp, 'r') as fid:
        extracted_text = fid.read().decode('utf-8')
    return extracted_text


def write_txt_gz_file(text, fp):
    """
    Writes text to a txt.gz file
    Args:
        text: String to write
        fp: File path

    Returns:
        None
    """
    with gzip.open(fp, 'w') as fid:
        fid.write(text.encode('utf-8'))


def read_json_gz_file(fp):
    """
    Reads contents of a json.gz file
    Args:
        fp: File path

    Returns:
        Contents of JSON file as dict
    """
    return json.loads(read_txt_gz_file(fp))


def perform_tesseract_ocr(image_path, language=None):
    """
    Performs OCR on an image using Tesseract
    Args:
        image_path: Full path of the image file
        language: Language of the slide

    Returns:
        Extracted text
    """
    if language is None:
        language = 'en'
    if not file_exists(image_path):
        print(f'Error: File {image_path} does not exist')
        return None
    return pytesseract.image_to_string(Image.open(image_path), lang={'en': 'eng', 'fr': 'fra'}[language])


def tesseract_ocr_or_get_cached(ocr_path, image_path, language):
    """
    Performs OCR using tesseract or uses cached results
    Args:
        ocr_path: Root path of OCR files
        image_path: Root path of image files
        language: Langauge of the slides

    Returns:
        Extracted text
    """
    if file_exists(ocr_path):
        extracted_text = read_txt_gz_file(ocr_path)
    else:
        extracted_text = perform_tesseract_ocr(image_path, language)
        if extracted_text is None:
            extracted_text = ''
        write_txt_gz_file(extracted_text, ocr_path)
    return extracted_text


def generate_img_and_ocr_paths_and_perform_tesseract_ocr(input_folder_with_path, k, language=None):
    if language is None:
        language = 'en'
    image_path = os.path.join(input_folder_with_path, (FRAME_FORMAT_PNG) % (k))
    ocr_path = os.path.join(input_folder_with_path, (TESSERACT_OCR_FORMAT) % (k))
    extracted_text = tesseract_ocr_or_get_cached(ocr_path, image_path, language)
    return extracted_text


class NLPModels:
    def __init__(self):
        n_dims = config['fasttext']['dim']
        base_dir = config['fasttext']['path']
        self.model_paths = {
            lang: generate_target_path(base_dir, lang, n_dims)
            for lang in ['en', 'fr']
        }
        self.nlp_models = None
        self.tokenizers = None
        self.stopwords = None

    def load_nlp_models(self):
        """
        Lazy-loads and returns the NLP models used for local OCR in slide detection
        Returns:
            The NLP model dict
        """
        if self.nlp_models is None:
            self.nlp_models = {
                lang: fasttext.load_model(self.model_paths[lang])
                for lang in self.model_paths
            }
            self.tokenizers = {
                lang: MosesTokenizer(lang)
                for lang in self.nlp_models
            }
            self.stopwords = STOPWORDS

    def get_words(self, text, lang='en', valid_only=False):
        self.load_nlp_models()
        current_tokenizer = self.tokenizers[lang]
        current_stopwords = self.stopwords[lang]
        all_words = current_tokenizer.tokenize(text, return_str=False, escape=False)
        if valid_only:
            all_words = [w for w in all_words
                         if str(w.lower()) not in current_stopwords
                         and str(w) not in PUNKT]
            if lang == 'fr':
                all_words = [w.strip("'") for w in all_words]
        return all_words

    def get_text_word_vector(self, text, lang='en', valid_only=True):
        self.load_nlp_models()
        current_model = self.nlp_models[lang]
        all_valid_words = self.get_words(text, lang, valid_only=valid_only)

        result_vector = sum(current_model.get_word_vector(w) for w in all_valid_words)
        return result_vector

    def get_text_word_vector_using_words(self, words, lang='en'):
        self.load_nlp_models()
        current_model = self.nlp_models[lang]

        result_vector = sum(current_model.get_word_vector(w) for w in words)
        return result_vector


def get_cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def frame_ocr_distance(input_folder_with_path, k1, k2, nlp_models: NLPModels, language=None):
    """
    Computes OCR distance between two frames
    Args:
        input_folder_with_path: Full path of input image folder
        k1: Index of frame 1
        k2: Index of frame 2
        nlp_models: NLP models used to compute distance between OCR results
        language: Language of the text in the input images

    Returns:
        Distance between the two frames
    """
    if language is None:
        language = 'en'

    # Cache the results since they will be used multiple times during slide detection
    extracted_text1 = generate_img_and_ocr_paths_and_perform_tesseract_ocr(input_folder_with_path, k1, language)
    extracted_text2 = generate_img_and_ocr_paths_and_perform_tesseract_ocr(input_folder_with_path, k2, language)

    # Calculate NLP objects
    nlp_1 = nlp_models.get_words(extracted_text1, language)
    nlp_2 = nlp_models.get_words(extracted_text2, language)

    # Calculate distance score
    if np.max([len(nlp_1), len(nlp_2)]) < 16:
        text_dif = 0
    elif np.min([len(nlp_1), len(nlp_2)]) < 4 and np.max([len(nlp_1), len(nlp_2)]) >= 16:
        text_dif = 1
    else:
        text_sim = get_cosine_sim(nlp_models.get_text_word_vector(extracted_text1, language),
                                  nlp_models.get_text_word_vector(extracted_text2, language))
        text_dif = 1 - text_sim
        text_dif = text_dif * (1 - np.exp(-np.mean([len(nlp_1), len(nlp_2)]) / 16))

    # Return distance score
    return text_dif


def frame_hash_similarity(input_folder_with_path, k1, k2):
    """
    Computes the hash-based similarity between two frames
    Args:
        input_folder_with_path: Full path of the input image folder
        k1: Index of frame 1
        k2: Index of frame 2

    Returns:
        Similarity between the two frames (between 0 and 1)
    """
    image_1_path = os.path.join(input_folder_with_path, (FRAME_FORMAT_PNG) % (k1))
    image_2_path = os.path.join(input_folder_with_path, (FRAME_FORMAT_PNG) % (k2))

    image_1_hash = perceptual_hash_image(image_1_path)
    image_2_hash = perceptual_hash_image(image_2_path)

    return compare_encoded_fingerprints(image_1_hash, image_2_hash, decoder_func=imagehash.hex_to_hash)


def compute_ocr_noise_level(input_folder_with_path, frame_sample_indices, nlp_models, language=None):
    """
    Computes noise values for a sequence of frames
    Args:
        input_folder_with_path: Full path of the input image folder
        frame_sample_indices: Indices of the sampled frames
        nlp_models: The NLP models used for the OCR distance
        language: Language of the slides

    Returns:
        List of distances identified as noise (i.e. below the default noise threshold)
    """
    print('Estimating transition threshold ...')
    distance_list = list()
    for k in range(1, len(frame_sample_indices)):
        d = frame_ocr_distance(input_folder_with_path, frame_sample_indices[k - 1], frame_sample_indices[k],
                               nlp_models, language)
        if 0.0001 < d < 0.01:
            distance_list.append(d)
    return distance_list


def compute_ocr_threshold(distance_list, default_threshold=0.1):
    """
    Computes the OCR noise threshold using a list of subsequent frame distances
    Args:
        distance_list: List of OCR distances
        default_threshold: Default value to use if the list is empty

    Returns:
        The noise threshold
    """
    threshold = float(5 * np.median(distance_list))
    if math.isnan(threshold):
        return default_threshold
    return threshold


def check_ocr_and_hash_thresholds(input_folder_with_path, k_l, k_r,
                                  ocr_dist_threshold, hash_similarity_threshold, nlp_models,
                                  language=None):
    d = frame_ocr_distance(input_folder_with_path, k_l, k_r, nlp_models, language)
    s_hash = frame_hash_similarity(input_folder_with_path, k_l, k_r)
    return (d > ocr_dist_threshold and s_hash < hash_similarity_threshold), d, s_hash


def frame_ocr_transition(input_folder_with_path, k_l, k_r, ocr_dist_threshold, hash_similarity_threshold, nlp_models,
                         language=None):
    """
    Recursive function that finds slide transitions through binary tree search
    Args:
        input_folder_with_path: Full path of input image folder, where they all follow FRAME_FORMAT
        k_l: Leftmost index of the binary search
        k_r: Rightmost index of the binary search
        ocr_dist_threshold: Minimum OCR-based distance for two frames to be considered distinct
        hash_similarity_threshold: Maximum hash-based similarity for two frames to be considered distinct
        nlp_models: NLP models for the OCR results
        language: Language of the document

    Returns:
        [transition frame index, distance] if a transition is found, [None, None] otherwise
    """
    k_m = int(np.mean([k_l, k_r]))
    threshold_check, d, s_hash = check_ocr_and_hash_thresholds(input_folder_with_path, k_l, k_r,
                                                               ocr_dist_threshold, hash_similarity_threshold,
                                                               nlp_models, language)
    if k_m == k_l or k_m == k_r:
        return [k_r, d]
    else:
        if threshold_check:
            [k_sep_l, d_l] = frame_ocr_transition(input_folder_with_path, k_l, k_m, ocr_dist_threshold,
                                                  hash_similarity_threshold,
                                                  nlp_models, language)
            [k_sep_r, d_r] = frame_ocr_transition(input_folder_with_path, k_m, k_r, ocr_dist_threshold,
                                                  hash_similarity_threshold,
                                                  nlp_models, language)
            if k_sep_l is None and k_sep_r is None:
                return [None, None]
            elif k_sep_l is not None and k_sep_r is None:
                return [k_sep_l, d_l]
            elif k_sep_l is None and k_sep_r is not None:
                return [k_sep_r, d_r]
            else:
                if d_r >= d_l:
                    return [k_sep_r, d_r]
                else:
                    return [k_sep_l, d_l]
        else:
            return [None, None]


def compute_video_ocr_transitions(input_folder_with_path, frame_sample_indices, ocr_dist_threshold, hash_dist_threshold,
                                  nlp_models, language=None, keep_first=True):
    """
    Computes all the slide transitions for slides extracted from a video file
    Args:
        input_folder_with_path: Path of the slide folder
        frame_sample_indices: Indices of sampled frames
        ocr_dist_threshold: Threshold for OCR distance (below which slides are considered to be the same)
        hash_dist_threshold: Threshold for perceptual hash similarity (above which they are considered to be the same)
        nlp_models: NLP models for parsing the OCR results
        language: Language of the slides
        keep_first: Whether to return the first frame index as a slide, True by default

    Returns:
        List of transitory slides
    """
    if len(frame_sample_indices) == 0:
        return list()
    generate_img_and_ocr_paths_and_perform_tesseract_ocr(input_folder_with_path,
                                                         frame_sample_indices[0], language)
    if keep_first:
        transition_list = [frame_sample_indices[0]]
    else:
        transition_list = list()
    for k in range(1, len(frame_sample_indices)):
        [t, d] = frame_ocr_transition(
            input_folder_with_path,
            frame_sample_indices[k - 1],
            frame_sample_indices[k],
            ocr_dist_threshold,
            hash_dist_threshold,
            nlp_models,
            language
        )
        if t is not None and t < frame_sample_indices[-1]:
            transition_list.append(t)
    # Making sure the first and second elements are not the same
    if len(transition_list) >= 2:
        t_check, d, s_hash = check_ocr_and_hash_thresholds(input_folder_with_path,
                                                           transition_list[0], transition_list[1],
                                                           ocr_dist_threshold, hash_dist_threshold, nlp_models,
                                                           language)
        if not t_check:
            transition_list = transition_list[1:]
    return transition_list


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

    def load_model_whisper(self):
        """
        Lazy-loads a Whisper model into memory
        Args:
            model_type: Type of model, see Whisper docs for details

        Returns:
            Model object
        """
        # device=None ensures that the model will use CUDA if available and switch to CPUs otherwise.
        if self.model is None:
            print('Actually loading Whisper model...')
            self.model = whisper.load_model(self.model_type, device=None, in_memory=True,
                                            download_root=self.download_root)

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
                                 no_speech_threshold=0.6, logprob_threshold=-1):
        """
        Transcribes an audio file using whisper
        Args:
            input_filename_with_path: Path to input file
            force_lang: Whether to explicitly feed the model the language of the audio.
                        None results in automatic detection.
            verbose: Verbosity of the transcription
            no_speech_threshold: If the probability of a segment containing no speech is above this threshold
            (and the model has low confidence in the text it has predicted), it is treated as silent.
            logprob_threshold: Log probability threshold
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
        except Exception as e:
            print(e, file=sys.stderr)
            return None
        return result


class GoogleOCRModel:
    def __init__(self):
        try:
            print("Reading Google API key from config")
            self.api_key = config['google']['api_key']
        except Exception:
            self.api_key = None

        if self.api_key is None:
            print(
                "The Google API key could not be found in the config file. "
                "Make sure to add a [google] section with the api_key parameter. "
                "Google API endpoints cannot be used as there is no default API key."
            )

        # The actual Google model is lazy loaded in order not to load it twice (celery *and* gunicorn)
        self.model = None

    def establish_connection(self):
        """
        Lazily connects to the Google API
        Returns:
            True if a connection already exists or if a new connection is successfully established, False otherwise
        """
        if self.model is None:
            if self.api_key is not None:
                print('Establishing Google API connection...')
                try:
                    self.model = vision.ImageAnnotatorClient(client_options={"api_key": self.api_key})
                    return True
                except Exception:
                    print('Failed to connect to Google API!')
                    return False
            else:
                print('No API key provided!')
                return False
        else:
            return True

    def perform_ocr(self, input_filename_with_path):
        """
        Performs OCR with two methods (text_detection and document_text_detection)
        Args:
            input_filename_with_path: Full path of the input image file

        Returns:
            Text results of the two OCR methods
        """
        model_loaded = self.establish_connection()
        if not model_loaded:
            return None, None
        # Loading the image
        if not file_exists(input_filename_with_path):
            print(f'Error: File {input_filename_with_path} does not exist')
            return None, None
        with io.open(input_filename_with_path, 'rb') as image_file:
            image_content = image_file.read()
        g_image_obj = vision.Image(content=image_content)
        # Waiting for results (accounting for possible failures)
        results_1 = self.wait_for_ocr_results(image_object=g_image_obj, method='dtd')
        results_2 = self.wait_for_ocr_results(image_object=g_image_obj, method='td')
        return results_1, results_2

    def wait_for_ocr_results(self, image_object, method='dtd', retries=6):
        """
        Makes call to Google OCR API and waits for the results
        Args:
            image_object: Image object for the Google API
            method: Method to use, 'td' for text detection and 'dtd' for document text detection
            retries: Number of retries to perform in case of failure

        Returns:
            OCR results
        """
        assert method in ['dtd', 'td']
        results = None
        for i in range(retries):
            try:
                if method == 'dtd':
                    results = self.model.document_text_detection(image=image_object)
                else:
                    results = self.model.text_detection(image=image_object)
                break
            except Exception:
                print('Failed to call OCR engine. Trying again in 60 seconds ...')
                time.sleep(5)
        if results is not None:
            results = results.full_text_annotation.text
        return results
