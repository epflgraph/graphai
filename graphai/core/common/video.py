import os
import sys
import io
import configparser
import math
import glob
import re
import random
import json
import time
from datetime import datetime
import gzip
import wget

import numpy as np

import hashlib
import acoustid
import chromaprint
import ffmpeg
import imagehash
from PIL import Image
import langdetect
import pysbd
import fingerprint
from fuzzywuzzy import fuzz

import pytesseract
from google.cloud import vision
import spacy
import whisper
import openai
import tiktoken
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from graphai.definitions import CONFIG_DIR
from graphai.core.common.caching import make_sure_path_exists, file_exists

FRAME_FORMAT_PNG = 'frame-%06d.png'
FRAME_FORMAT_JPG = 'frame-%06d.jpg'
TESSERACT_OCR_FORMAT = 'ocr-%06d.txt.gz'


def write_text_file(filename_with_path, contents):
    """
    Writes contents to text file
    Args:
        filename_with_path: Full path of the file
        contents: Textual contents

    Returns:
        None
    """
    with open(filename_with_path, 'w') as f:
        f.write(contents)


def write_json_file(filename_with_path, d):
    """
    Writes dictionary to JSON file
    Args:
        filename_with_path: Full path of the file
        d: Dictionary to write

    Returns:
        None
    """
    with open(filename_with_path, 'w') as f:
        json.dump(d, f)


def read_text_file(filename_with_path):
    """
    Opens and reads the contents of a text file
    Args:
        filename_with_path: Full path of the file

    Returns:
        Contents of the file
    """
    with open(filename_with_path, 'r') as f:
        return f.read()


def read_json_file(filename_with_path):
    """
    Reads the contents of a JSON file
    Args:
        filename_with_path: Full path of the file

    Returns:
        Dictionary containing contents of the JSON file
    """
    with open(filename_with_path, 'r') as f:
        return json.load(f)


def md5_text(s):
    """
    Computes the md5 hash of a string
    Args:
        s: The string

    Returns:
        MD5 hash
    """
    return hashlib.md5(s.encode('utf8')).hexdigest()


def generate_random_token():
    """
    Generates a random string using the current time and a random number to be used as a token.
    Returns:
        Random token
    """
    return ('%.06f' % time.time()).replace('.', '') + '%08d' % random.randint(0, int(1e7))


def format_datetime_for_mysql(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def get_current_datetime():
    """
    Returns current datetime formatted for MySQL
    Returns:
        Datetime string
    """
    current_datetime = format_datetime_for_mysql(datetime.now())
    return current_datetime


def generate_src_tgt_dict(src, tgt):
    """
    Creates a source language and target language dictionary for translation
    Args:
        src: Source lang
        tgt: Target lang

    Returns:
        dict
    """
    return {'source_lang': src, 'target_lang': tgt}


def generate_translation_text_token(s, src, tgt):
    """
    Generates an md5-based token for a string
    Args:
        s: The string
        src: Source lang
        tgt: Target lang

    Returns:
        Token
    """
    return md5_text(s) + '_' + src + '_' + tgt


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
    result, _ = ffmpeg.output(in_stream, 'pipe:', c='copy', format='md5').run(capture_stdout=True)
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
        return None, None
    results = acoustid.fingerprint_file(input_filename_with_path, maxlength=max_length)
    fingerprint_value = results[1]
    decoded = chromaprint.decode_fingerprint(fingerprint_value)
    return fingerprint_value.decode('utf8'), decoded


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


def perceptual_hash_text(s, min_window_length=5, max_window_length=50, hash_len=32):
    """
    Computes the perceptual hash of a strong
    Args:
        s: String to hash
        min_window_length: Minimum window length for k-grams
        max_window_length: Maximum window length for k-grams
        hash_len: Length of the hash

    Returns:
        Perceptual hash of string
    """
    string_length = len(s)
    window_length = max([min_window_length, string_length // hash_len])
    window_length = min([max_window_length, window_length])
    if string_length < window_length:
        s = s + ''.join([' '] * (window_length - string_length + 1))
    kgram_length = max([10, int(window_length / 2)])

    fprinter = fingerprint.Fingerprint(kgram_len=kgram_length, window_len=window_length, base=10, modulo=256)
    hash_numbers = fprinter.generate(str=s)
    if len(hash_numbers) > hash_len:
        sample_indices = np.linspace(start=0, stop=len(hash_numbers) - 1, num=hash_len - 1, endpoint=False).tolist()
        sample_indices.append(len(hash_numbers) - 1)
        sample_indices = [int(x) for x in sample_indices]
        hash_numbers = [hash_numbers[i] for i in sample_indices]
    elif len(hash_numbers) < hash_len:
        hash_numbers = hash_numbers + [(0, 0)] * (32 - len(hash_numbers))
    fp_result = ''.join([f"{n[0]:02x}" for n in hash_numbers])
    return fp_result


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


def compare_encoded_fingerprints(f1, f2=None, decoder_func=chromaprint.decode_fingerprint):
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
                                       decoder_func=chromaprint.decode_fingerprint):
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
                                              decoder_func=chromaprint.decode_fingerprint)


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
    frame_sample_indices = list(np.arange(1, n_frames - step + 1, step)) + [n_frames - 1]
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
        if extracted_text is not None:
            write_txt_gz_file(extracted_text, ocr_path)
    return extracted_text


def frame_ocr_distance(input_folder_with_path, k1, k2, nlp_models, language=None):
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

    # Generate frame file paths
    image_1_path = os.path.join(input_folder_with_path, (FRAME_FORMAT_PNG) % (k1))
    image_2_path = os.path.join(input_folder_with_path, (FRAME_FORMAT_PNG) % (k2))

    # Generate OCR file paths
    ocr_1_path = os.path.join(input_folder_with_path, (TESSERACT_OCR_FORMAT) % (k1))
    ocr_2_path = os.path.join(input_folder_with_path, (TESSERACT_OCR_FORMAT) % (k2))

    # Cache the results since they will be used multiple times during slide detection
    extracted_text1 = tesseract_ocr_or_get_cached(ocr_1_path, image_1_path, language)
    extracted_text2 = tesseract_ocr_or_get_cached(ocr_2_path, image_2_path, language)

    # Calculate NLP objects
    nlp_1 = nlp_models[language](extracted_text1)
    nlp_2 = nlp_models[language](extracted_text2)

    # Calculate distance score
    if np.max([len(nlp_1), len(nlp_2)]) < 32:
        text_dif = 0
    elif np.min([len(nlp_1), len(nlp_2)]) < 4 and np.max([len(nlp_1), len(nlp_2)]) >= 32:
        text_dif = 1
    else:
        text_sim = nlp_1.similarity(nlp_2)
        text_dif = 1 - text_sim
        text_dif = text_dif * (1 - np.exp(-np.mean([len(nlp_1), len(nlp_2)]) / 32))

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
    d = frame_ocr_distance(input_folder_with_path, k_l, k_r, nlp_models, language)
    s_hash = frame_hash_similarity(input_folder_with_path, k_l, k_r)
    if k_m == k_l or k_m == k_r:
        return [k_r, d]
    else:
        if d > ocr_dist_threshold and s_hash < hash_similarity_threshold:
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
                                  nlp_models, language=None):
    """
    Computes all the slide transitions for slides extracted from a video file
    Args:
        input_folder_with_path: Path of the slide folder
        frame_sample_indices: Indices of sampled frames
        ocr_dist_threshold: Threshold for OCR distance (below which slides are considered to be the same)
        hash_dist_threshold: Threshold for perceptual hash similarity (above which they are considered to be the same)
        nlp_models: NLP models for parsing the OCR results
        language: Language of the slides

    Returns:
        List of transitory slides
    """
    if len(frame_sample_indices) == 0:
        return list()
    transition_list = [frame_sample_indices[0]]
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
    if len(transition_list) >= 2 and transition_list[0] == transition_list[1]:
        transition_list = transition_list[1:]
    return transition_list


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


def count_tokens_for_openai(text, model="cl100k_base"):
    """
    Counts the number of tokens in a given text for a given OpenAI model
    Args:
        text: The text to tokenize
        model: The OpenAI model to use

    Returns:
        Number of tokens
    """
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))


def force_dict_to_text(t):
    if isinstance(t, dict):
        return json.dumps(t)
    return t


def generate_summary_text_token(text, text_type='text', summary_type='summary', len_class='normal', tone='info'):
    assert text_type in ['person', 'unit', 'concept', 'course', 'lecture', 'MOOC', 'publication', 'text']
    assert summary_type in ['summary', 'title']
    assert len_class in ['vshort', 'short', 'normal']
    assert tone in ['info', 'promo']

    text = force_dict_to_text(text)
    token = md5_text(text) + '_' + text_type + '_' + summary_type + '_' + len_class + '_' + tone
    return token


def generate_summary_type_dict(text_type, summary_type, len_class, tone):
    return {
        'input_type': text_type,
        'summary_type': summary_type,
        'summary_len_class': len_class,
        'summary_tone': tone
    }


class WhisperTranscriptionModel():
    def __init__(self):
        # The actual Whisper model is lazy loaded in order not to load it twice (celery *and* gunicorn)
        config_contents = configparser.ConfigParser()
        try:
            print('Reading model configuration from file')
            config_contents.read(f'{CONFIG_DIR}/models.ini')
            self.model_type = config_contents['WHISPER'].get('model_type', fallback='medium')
        except Exception:
            print(f'Could not read file {CONFIG_DIR}/models.ini or '
                  f'file does not have section [WHISPER], falling back to defaults.')
            self.model_type = 'medium'
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
                result = self.model.transcribe(input_filename_with_path, verbose=verbose, language=force_lang, fp16=True)
        except Exception as e:
            print(e, file=sys.stderr)
            return None
        return result


class NLPModels:
    def __init__(self):
        spacy.prefer_gpu()
        self.nlp_models = None

    def get_nlp_models(self):
        """
        Lazy-loads and returns the NLP models used for local OCR in slide detection
        Returns:
            The NLP model dict
        """
        if self.nlp_models is None:
            self.nlp_models = {
                'en': spacy.load('en_core_web_lg'),
                'fr': spacy.load('fr_core_news_md')
            }
        return self.nlp_models


class GoogleOCRModel():
    def __init__(self):
        self.model = None
        config_contents = configparser.ConfigParser()
        try:
            print('Reading API key from file')
            config_contents.read(f'{CONFIG_DIR}/models.ini')
            self.api_key = config_contents['GOOGLE'].get('api_key', fallback=None)
        except Exception:
            self.api_key = None
        if self.api_key is None:
            print(f'Could not read file {CONFIG_DIR}/models.ini or '
                  f'file does not have section [GOOGLE], Google API '
                  f'endpoints cannot be used as there is no '
                  f'default API key.')

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


class TranslationModels:
    def __init__(self):
        self.models = None

    def load_models(self):
        """
        Loads Huggingface translation and tokenization models plus a pysbd segmenter
        Returns:
            None
        """
        if self.models is None:
            self.models = dict()
            print('Loading EN-FR')
            self.models['en-fr'] = dict()
            self.models['en-fr']['tokenizer'] = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
            self.models['en-fr']['model'] = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
            self.models['en-fr']['segmenter'] = pysbd.Segmenter(language='en', clean=False)
            print('Loading FR-EN')
            self.models['fr-en'] = dict()
            self.models['fr-en']['tokenizer'] = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
            self.models['fr-en']['model'] = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
            self.models['fr-en']['segmenter'] = pysbd.Segmenter(language='fr', clean=False)

    def _tokenize_and_get_model_output(self, sentence, tokenizer, model):
        """
        Internal method. Translates one single sentence.
        Args:
            sentence: Sentence to translate
            tokenizer: Tokenizer model
            model: Translation model

        Returns:
            Translation result or None if translation fails
        """
        try:
            input_ids = tokenizer.encode(sentence, return_tensors="pt")
            outputs = model.generate(input_ids, max_length=512)
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return decoded
        except IndexError as e:
            print(e)
            return None

    def _translate(self, text, tokenizer, model, segmenter):
        """
        Internal method. Translates entire text, sentence by sentence.
        Args:
            text: Text to translate
            tokenizer: Tokenizer model
            model: Translation model
            segmenter: Sentence segmenter

        Returns:
            Translated text, plus a flag denoting whether any of the sentences was too long and unpunctuated
        """
        sentences = segmenter.segment(text.replace('\n', '. '))
        sentences = [sentence.strip() for sentence in sentences]
        full_result = ''
        for sentence in sentences:
            print(sentence)
            if len(sentence) == 0:
                continue
            if len(sentence) < 4:
                full_result += ' ' + sentence
                continue
            decoded = self._tokenize_and_get_model_output(sentence, tokenizer, model)
            print(decoded)
            if decoded is None:
                return None, True
            full_result += ' ' + decoded

        return full_result, False

    def translate(self, text, how='en-fr'):
        """
        Translates provided text
        Args:
            text: Text to translate
            how: source-target language

        Returns:
            Translated text and 'unpunctuated text too long' flag
        """
        self.load_models()
        if how not in self.models.keys():
            raise NotImplementedError("Source or target language not implemented")
        if text is None or text == '':
            return None, False
        tokenizer = self.models[how]['tokenizer']
        model = self.models[how]['model']
        segmenter = self.models[how]['segmenter']
        return self._translate(text, tokenizer, model, segmenter)


class FingerprintParameters:
    def __init__(self):
        self.min_similarity = dict()
        self.load_values()

    def load_values(self):
        """
        Loads the values of fingerprinting similarity thresholds from file, or failing that, sets them to defaults
        Returns:
            None
        """
        defaults = {
            'text': '1.0',
            'image': '1.0',
            'audio': '0.8',
            'video': '1.0'
        }
        config_contents = configparser.ConfigParser()
        try:
            print('Reading fingerprint min similarity values from file')
            config_contents.read(f'{CONFIG_DIR}/fingerprint.ini')
            self.min_similarity['text'] = float(config_contents['FP'].get('text', fallback=defaults['text']))
            self.min_similarity['image'] = float(config_contents['FP'].get('image', fallback=defaults['image']))
            self.min_similarity['audio'] = float(config_contents['FP'].get('audio', fallback=defaults['audio']))
            self.min_similarity['video'] = float(config_contents['FP'].get('video', fallback=defaults['video']))
        except Exception:
            self.min_similarity = {k: float(v) for k, v in defaults.items()}

    def get_min_sim_text(self):
        return self.min_similarity['text']

    def get_min_sim_image(self):
        return self.min_similarity['image']

    def get_min_sim_audio(self):
        return self.min_similarity['audio']

    def get_min_sim_video(self):
        return self.min_similarity['video']


class ChatGPTSummarizer:
    def __init__(self):
        config_contents = configparser.ConfigParser()
        try:
            print('Reading ChatGPT API key from file')
            config_contents.read(f'{CONFIG_DIR}/models.ini')
            self.api_key = config_contents['CHATGPT'].get('api_key', fallback=None)
        except Exception:
            self.api_key = None
        if self.api_key is None:
            print(f'Could not read file {CONFIG_DIR}/models.ini or '
                  f'file does not have section [CHATGPT], ChatGPT API '
                  f'endpoints cannot be used as there is no '
                  f'default API key.')

    def establish_connection(self):
        """
        Ensures that an API key exists and sets it as the OpenAI key
        Returns:
            Boolean indicating whether an API key was found
        """
        if self.api_key is not None:
            openai.api_key = self.api_key
            return True
        else:
            return False

    def _generate_completion(self, text, system_message, max_len):
        """
        Internal method, generates a chat completion, which is the OpenAI API endpoint for ChatGPT interactions
        Args:
            text: The text to be provided in the "user" role, i.e. the text that is to be processed by ChatGPT
            system_message: The text to be provided in the "system" role, which provides directives to ChatGPT
            max_len: Approximate maximum length of the response in words

        Returns:
            Results returned by ChatGPT, a flag that is True if there were too many tokens, and the total # of tokens
            if (and only if) the request is successful (0 otherwise).
            A (None, True, 0) result means that the completion failed because the message had too many tokens,
            while a (None, False, 0) result indicates a different error (e.g. failed connection).
        """
        has_api_key = self.establish_connection()
        if not has_api_key:
            return None, False, 0
        # We count the approximate number of tokens in order to choose the right model (i.e. context size)
        approx_token_count = count_tokens_for_openai(text) + count_tokens_for_openai(system_message) + int(2 * max_len)
        if approx_token_count < 4096:
            model_type = 'gpt-3.5-turbo'
        elif 4096 < approx_token_count < 16384:
            model_type = 'gpt-3.5-turbo-16k'
        else:
            # If the token count is above 16384, the text is too large and we can't summarize it
            return None, True, 0
        try:
            # Generate the completion
            completion = openai.ChatCompletion.create(
                model=model_type,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": text}
                ],
                max_tokens=int(2 * max_len)
            )
            print(completion)
        except openai.error.InvalidRequestError as e:
            # We check to see if the exception was caused by too many tokens in the input
            print(e)
            if "This model's maximum context length is" in e:
                return None, True, 0
            else:
                return None, False, 0
        except Exception as e:
            # Any error other than "too many tokens" is dealt with here
            print(e)
            return None, False, 0
        return completion.choices[0].message.content, False, completion.usage.total_tokens

    def generate_summary(self, text_or_dict, text_type='lecture', summary_type='summary',
                         len_class='normal', tone='info', max_normal_len=100, max_short_len=40):
        """
        Generates a summary or a title for the provided text
        Args:
            text_or_dict: String or dictionary containing all the text to summarize and synthesize into one summary
            text_type: Type of text, e.g. "lecture", "course". Useful for summaries.
            summary_type: The type of the summary to be produced, either 'title' or 'summary' (default)
            len_class: Whether there's a constraint on the number of sentences ('vshort' for 1, 'short' for 2)
                       or not ('normal', default).
            tone: Whether to use a marketing tone ('promo') or an informative tone ('info', default)
            max_normal_len: Approximate maximum length of result (in words)

        Returns:
            Result of summarization, plus a flag indicating whether there were too many tokens
        """
        if isinstance(text_or_dict, dict):
            text_dict_fields = [x.lower() for x in text_or_dict.keys()]
        else:
            if text_type != "text":
                text_dict_fields = ["text"]
            else:
                text_dict_fields = list()

        # Telling the API what information is being provided on the entity
        if len(text_dict_fields) > 0:
            system_message = f"You will be given the {', '.join(text_dict_fields)} of a {text_type}."
        else:
            system_message = f"You will be given a {text_type}."

        # We have certain constraints on the length of the response.
        n_sentences = None
        max_len = max_normal_len
        if len_class == 'vshort':
            n_sentences = 1
            max_len = max_short_len / 2
        elif len_class == 'short':
            n_sentences = 2
            max_len = max_short_len
        if n_sentences is not None:
            sentences = f" {n_sentences}-sentence"
        else:
            sentences = ""
        max_len_str = f" with under {max_len} words."

        # Based on the text_type, we may have additional constraints.
        # This section should be expanded based on feedback
        exclude_name = (summary_type == 'summary' and n_sentences == 1) or \
                       (summary_type == 'title' and n_sentences is not None)
        additional_constraints = ""
        if text_type == "person":
            additional_constraints = " INCLUDE their job title and place of work in the response (if available)."
            if exclude_name:
                additional_constraints += " EXCLUDE the name of the person from the response."
        elif text_type == "unit":
            additional_constraints = " INCLUDE the institution that it is part of in the response (if available)."
            if exclude_name:
                additional_constraints += " EXCLUDE the name of the unit from the response."
        elif text_type == 'concept':
            if exclude_name:
                additional_constraints += " EXCLUDE the name of the concept from the response."
        elif text_type == 'course':
            additional_constraints = " INCLUDE the name of the professor teaching it (if available)."
            if exclude_name:
                additional_constraints += " EXCLUDE the name of the course from the response."
        elif text_type == 'MOOC':
            additional_constraints = " INCLUDE the name of the professor teaching it (if available)."
            if exclude_name:
                additional_constraints += " EXCLUDE the name of the MOOC from the response."
        elif text_type == 'lecture':
            if exclude_name:
                additional_constraints += " EXCLUDE the name of the lecture from the response."
        elif text_type == 'publication':
            additional_constraints += " EXCLUDE the paper's name from the response."
            if exclude_name:
                additional_constraints += " EXCLUDE the names of the authors from the response."
            else:
                additional_constraints = " INCLUDE the names of the first few authors in the response."

        # This is the main part that determines whether we get a title or a summary
        if summary_type == 'title':
            system_message += f" Generate a title for the {text_type}{max_len_str}."
        else:
            system_message += f" Generate a{sentences} summary for the " \
                              f"{text_type}{max_len_str}."

        # Adding the additional constraints
        system_message += additional_constraints

        if tone == 'promo':
            system_message += " Write in a promotional tone."
        else:
            system_message += " Write in a neutral, informative tone."

        # Now we compile the response format
        response_format = f"\"{summary_type}: "
        sample_response = ""
        # This section should also be expanded based on feedback
        if text_type == 'person':
            if n_sentences == 1:
                response_format += "[BRIEF DESCRIPTION OF CURRENT JOB]\""
                sample_response = \
                    f"\"{summary_type}: Associate Professor at EPFL working on social network analysis"
            elif n_sentences == 2:
                response_format += "[BRIEF DESCRIPTION OF CURRENT JOB], [BRIEF DESCRIPTION OF INTERESTS].\""
                sample_response = \
                    f"\"{summary_type}: Associate Professor at EPFL working on social network analysis, " \
                    f"with contributions to graph theory and graph neural networks\""
            else:
                response_format += "[RESPONSE]\""
        elif text_type == 'unit':
            if n_sentences is not None:
                response_format += "[BRIEF DESCRIPTION OF RESEARCH OR DEVELOPMENT AREAS]\""
                sample_response = \
                    f"\"{summary_type}: Laboratory at EPFL working on social network analysis and graph neural networks"
            else:
                response_format += "[RESPONSE]\""
        elif text_type == 'concept':
            if n_sentences is not None:
                response_format += "[BRIEF EXPLANATION OF THE CONCEPT]\""
                sample_response = \
                    f"\"{summary_type}: Algorithm in graph theory that finds a minimum cut in a given graph"
            else:
                response_format += "[RESPONSE]\""
        elif text_type == 'course':
            if n_sentences is not None:
                response_format += "[BRIEF DESCRIPTION OF COURSE CONTENTS]\""
                sample_response = \
                    f"\"{summary_type}: Course on graph theory and graph algorithms for BSc Computer Science students"
            else:
                response_format += "[RESPONSE]\""
        elif text_type == 'MOOC':
            if n_sentences is not None:
                response_format += "[BRIEF DESCRIPTION OF MOOC CONTENTS]\""
                sample_response = \
                    f"\"{summary_type}: Introductory-level MOOC on graph theory and graph algorithms"
            else:
                response_format += "[RESPONSE]\""
        elif text_type == 'lecture':
            if n_sentences is not None:
                response_format += "[BRIEF DESCRIPTION OF LECTURE CONTENTS]\""
                sample_response = \
                    f"\"{summary_type}: Lecture introducing directed acyclic graphs and presenting a number of theorems"
            else:
                response_format += "[RESPONSE]\""
        elif text_type == 'publication':
            if n_sentences is not None:
                response_format += "[BRIEF DESCRIPTION OF PUBLICATION CONTENTS]\""
                sample_response = \
                    f"\"{summary_type}: Paper by Dillenbourg et al. introducing the concept of collaborative learning"
            else:
                response_format += "[RESPONSE]\""
        else:
            response_format += "[RESPONSE]\""
        system_message += f"Give your response in the form: {response_format}"
        if sample_response != "":
            system_message += f"\nHere's an example of an acceptable response: {sample_response}"

        if isinstance(text_or_dict, dict):
            text = "\n\n".join([f"{k}: {v}" for k, v in text_or_dict.items()])
        else:
            text = f"Text: {text_or_dict}"

        results, too_many_tokens, n_total_tokens = self._generate_completion(text, system_message, max_normal_len)
        # Now we remove the "Title:" or "Summary:" at the beginning
        results = ':'.join(results.split(':')[1:]).strip().strip('"')
        return results, too_many_tokens, n_total_tokens
