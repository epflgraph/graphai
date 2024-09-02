import os
import shutil
import sys
import math
import glob
import re
import random
import json
import time
from datetime import timedelta
import gzip
from itertools import chain
from multiprocessing import Lock

from sacremoses.tokenize import MosesTokenizer
import wget
import subprocess

import numpy as np

import ffmpeg
import imagehash
from PIL import Image

import pytesseract

import fasttext
from fasttext_reducer.reduce_fasttext_models import generate_target_path

from graphai.core.common.caching import (
    VideoConfig,
    VideoDBCachingManager,
    get_token_file_status,
    SlideDBCachingManager,
    AudioDBCachingManager
)
from graphai.core.common.config import config
from graphai.core.common.common_utils import (
    make_sure_path_exists,
    file_exists,
    get_current_datetime,
    copy_file_within_folder
)
from graphai.core.common.fingerprinting import (
    perceptual_hash_image,
    md5_video_or_audio,
    compare_encoded_fingerprints,
    perceptual_hash_audio
)
from graphai.core.common.lookup import (
    retrieve_fingerprint_callback,
    ignore_fingerprint_results_callback, is_fingerprinted, database_callback_generic
)

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


def get_file_size(file_path):
    if file_path is None:
        return None
    try:
        return os.path.getsize(file_path)
    except OSError:
        return None


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
        return str(result_code)


def retrieve_file_from_any_source(url, output_filename_with_path, output_token, is_kaltura=False):
    if is_kaltura:
        return retrieve_file_from_kaltura(url, output_filename_with_path, output_token)
    else:
        if 'youtube.com/' in url or 'youtu.be/' in url:
            return retrieve_file_from_youtube(url, output_filename_with_path, output_token)
        else:
            return retrieve_file_from_generic_url(url, output_filename_with_path, output_token)


def create_filename_using_url_format(token, url):
    file_format = url.split('.')[-1].lower()
    if file_format not in ['mp4', 'mkv', 'flv', 'avi', 'mov']:
        file_format = 'mp4'
    filename = token + '.' + file_format
    return filename


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


def get_available_streams(input_filename_with_path):
    results = perform_probe(input_filename_with_path)
    return [(x['codec_type'], x.get('codec_name', None)) for x in results['streams']]


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


def detect_audio_duration(input_filename_with_path):
    """
    Detects the duration of the audio track of the provided video file and returns its name in ogg format
    Args:
        input_filename_with_path: Path of input file
        input_token: Token of input file

    Returns:
        Audio duration.
    """
    try:
        probe_results = perform_probe(input_filename_with_path)
    except Exception as e:
        print(e, file=sys.stderr)
        return None
    if probe_results.get('format', None) is None or probe_results['format'].get('duration', None) is None:
        try:
            probe_results = perform_slow_audio_probe(input_filename_with_path)
        except Exception as e:
            print(e, file=sys.stderr)
            return None
    return float(probe_results['format']['duration'])


def generate_audio_token(token):
    return token + '_audio.ogg'


def extract_audio_from_video(input_filename_with_path, output_filename_with_path, output_token):
    """
    Extracts the audio track from a video.
    Args:
        input_filename_with_path: Path of input file
        output_filename_with_path: Path of output file
        output_token: Token of output file

    Returns:
        Output token and duration of audio if successful, None if not.
    """
    if not file_exists(input_filename_with_path):
        print(f'ffmpeg error: File {input_filename_with_path} does not exist')
        return None, None
    duration = detect_audio_duration(input_filename_with_path)
    if duration is None:
        return None, None
    try:
        err = ffmpeg.input(input_filename_with_path).audio. \
            output(output_filename_with_path, acodec='libopus', ar=48000). \
            overwrite_output().run(capture_stdout=True)
    except Exception as e:
        print(e, file=sys.stderr)
        err = str(e)

    if file_exists(output_filename_with_path) and ('ffmpeg error' not in err):
        return output_token, duration
    else:
        return None, None


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
        streams = get_available_streams(input_filename_with_path)
        video_stream_name = [x for x in streams if x[0] == 'video'][0][1]
        print('Starting ffmpeg slide extraction...')
        if video_stream_name != 'png':
            # If the video stream is NOT a single picture, we have to apply the "fps" filter to get one frame/second.
            # One frame per second is a fundamental assumption of the pipeline, so this value (1) should never
            # be changed (the algorithm assumes that timestamp == frame number).
            err = ffmpeg.input(input_filename_with_path).video. \
                filter("fps", 1).output(os.path.join(output_folder_with_path, FRAME_FORMAT_PNG)). \
                overwrite_output().run(capture_stdout=True)
        else:
            # If the video stream is just an image, the fps filter would fail (and it'd be unneeded), so it's skipped.
            err = ffmpeg.input(input_filename_with_path).video. \
                output(os.path.join(output_folder_with_path, FRAME_FORMAT_PNG)). \
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
        self.n_dims = int(config['fasttext']['dim'])
        self.base_dir = config['fasttext']['path']
        self.model_paths = {
            lang: generate_target_path(self.base_dir, lang, self.n_dims)
            for lang in ['en', 'fr']
        }
        self.nlp_models = None
        self.tokenizers = None
        self.stopwords = None
        self.load_lock = Lock()

    def load_nlp_models(self):
        """
        Lazy-loads and returns the NLP models used for local OCR in slide detection
        Returns:
            The NLP model dict
        """
        with self.load_lock:
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
        if len(text) == 0:
            return []
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
        all_valid_words = self.get_words(text, lang, valid_only=valid_only)
        return self._word_list_to_word_vector(all_valid_words, lang)

    def get_text_word_vector_using_words(self, words, lang='en'):
        self.load_nlp_models()
        return self._word_list_to_word_vector(words, lang)

    def _word_list_to_word_vector(self, words, lang='en'):
        if len(words) == 0:
            return np.zeros((self.n_dims, ))

        current_model = self.nlp_models[lang]
        result_vector = sum(current_model.get_word_vector(w) for w in words)
        return result_vector


def get_cosine_sim(v1, v2):
    return np.dot(np.array(v1).flatten(), np.array(v2).flatten()) / (np.linalg.norm(v1) * np.linalg.norm(v2))


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
        text_dif = 0.0
    elif np.min([len(nlp_1), len(nlp_2)]) < 4 and np.max([len(nlp_1), len(nlp_2)]) >= 16:
        text_dif = 1.0
    else:
        text_sim = get_cosine_sim(nlp_models.get_text_word_vector(extracted_text1, language),
                                  nlp_models.get_text_word_vector(extracted_text2, language))
        text_dif = 1 - text_sim
        assert isinstance(text_dif, float)
        text_dif = text_dif * (1 - np.exp(-np.mean([len(nlp_1), len(nlp_2)]) / 16))
        assert isinstance(text_dif, float)

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


def compute_ocr_threshold(distance_list, default_threshold=0.05):
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
    d = float(frame_ocr_distance(input_folder_with_path, k_l, k_r, nlp_models, language))
    s_hash = float(frame_hash_similarity(input_folder_with_path, k_l, k_r))
    ocr_dist_pass = d > ocr_dist_threshold
    hash_sim_pass = s_hash < hash_similarity_threshold
    return (ocr_dist_pass and hash_sim_pass), d, s_hash


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
        streams = get_available_streams(video_config.generate_filepath(token))
        # Here we only care about codec types (audio/video), not codec names (e.g. h264, aac, png)
        streams = [x[0] for x in streams]
        streams = [x for x in streams if x == 'audio' or x == 'video']
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


def retrieve_file_from_url(url, file_manager, is_kaltura=True, force_token=None):
    if force_token is not None:
        token = force_token
    else:
        db_manager = VideoDBCachingManager()
        existing = db_manager.get_details_using_origin(url, [])
        if existing is not None:
            # If the cache row already exists, then we don't create a new token, but instead
            # use the id_token of the existing row (we remove the file extension because it will be re-added soon)
            token = existing[0]['id_token'].split('.')[0]
        else:
            # Otherwise, we generate a random token
            token = generate_random_token()
    filename = create_filename_using_url_format(token, url)
    filename_with_path = file_manager.generate_filepath(filename)
    results = retrieve_file_from_any_source(url, filename_with_path, filename, is_kaltura)
    return {
        'token': results,
        'fresh': results == filename,
        'token_size': get_file_size(filename_with_path)
    }


def retrieve_file_from_url_callback(results, url):
    if results['fresh']:
        db_manager = VideoDBCachingManager()
        current_datetime = get_current_datetime()
        values = {
            'date_modified': current_datetime,
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


def compute_video_fingerprint(results, file_manager, force=False):
    token = results['token']
    db_manager = VideoDBCachingManager()
    if token is None or not results.get('fresh', True):
        fp = None
        fresh = False
        perform_lookup = False
        fp_token = None
    else:
        existing = db_manager.get_details(token, ['fingerprint'])[0]
        if not force and existing is not None and existing['fingerprint'] is not None:
            fp = existing['fingerprint']
            fresh = False
            perform_lookup = False
            fp_token = None
        else:
            fp = md5_video_or_audio(file_manager.generate_filepath(token), video=True)
            fresh = fp is not None
            perform_lookup = fp is not None
            fp_token = token if fp is not None else None
    return {
        'result': fp,
        'fp_token': fp_token,
        'perform_lookup': perform_lookup,
        'fresh': fresh,
        'original_results': results
    }


def compute_video_fingerprint_callback(results):
    if results['fresh']:
        token = results['fp_token']
        values_dict = {
            'fingerprint': results['result']
        }
        db_manager = VideoDBCachingManager()
        # The video might already have a row in the cache, or may be nonexistent there because it was not
        # retrieved from a URI. If the latter is the case, we add the current datetime to the cache row.
        if db_manager.get_details(token, [])[0] is None:
            values_dict['date_added'] = get_current_datetime()
        database_callback_generic(token, db_manager, values_dict, force=False, use_closest_match=False)
    return results


def cache_lookup_retrieve_file_from_url(url, file_manager):
    db_manager = VideoDBCachingManager()
    existing = db_manager.get_details_using_origin(url, [])
    if existing is not None:
        token = existing[0]['id_token']
        return {
            'token': token,
            'fresh': False,
            'token_status': get_video_token_status(token),
            'token_size': get_file_size(file_manager.generate_filepath(token))
        }
    return None


def cache_lookup_extract_audio(token):
    # Here, the caching logic is a bit complicated. The results of audio extraction are cached in the
    # audio tables, whereas the closest-matching video is cached in the video tables. As a result, we
    # need to look for the cached extracted audio of two videos: the provided token and its closest
    # token.
    video_db_manager = VideoDBCachingManager()
    audio_db_manager = AudioDBCachingManager()
    # Retrieving the closest match of the current video
    closest_token = video_db_manager.get_closest_match(token)
    # Looking up the cached audio result of the current video
    existing_own = audio_db_manager.get_details_using_origin(token, cols=['duration'])
    # Looking up the cached audio result of the closest match video (if it's not the same as the current video)
    if closest_token is not None and closest_token != token:
        existing_closest = audio_db_manager.get_details_using_origin(closest_token, cols=['duration'])
    else:
        existing_closest = None
    # We first look at the video's own existing audio, then at that of the closest match because the video's
    # own precomputed audio (if any) takes precedence.
    all_existing = [existing_own, existing_closest]
    for existing in all_existing:
        if existing is not None:
            print('Returning cached result')
            return {
                'token': existing[0]['id_token'],
                'fresh': False,
                'duration': existing[0]['duration'],
                'token_status': get_audio_token_status(existing[0]['id_token'])
            }

    return None


def extract_audio(token, file_manager):
    output_token = generate_audio_token(token)
    results, input_duration = extract_audio_from_video(file_manager.generate_filepath(token),
                                                       file_manager.generate_filepath(output_token),
                                                       output_token)
    if results is None:
        return {
            'token': None,
            'fresh': False,
            'duration': 0.0
        }
    return {
        'token': results,
        'fresh': True,
        'duration': input_duration
    }


def extract_audio_callback(results, origin_token, file_manager, force=False):
    if results['fresh']:
        current_datetime = get_current_datetime()
        db_manager = AudioDBCachingManager()
        db_manager.insert_or_update_details(
            results['token'],
            {
                'duration': results['duration'],
                'origin_token': origin_token,
                'date_added': current_datetime
            }
        )
        if not force:
            # If the force flag is False, we may need to propagate the results of this computation to its closest match.
            # The propagation happens if:
            # 1. The token has a closest match (that isn't itself)
            # 2. The closest match does NOT have cached slide results
            video_db_manager = VideoDBCachingManager()
            closest_video_match = video_db_manager.get_closest_match(origin_token)
            if (closest_video_match is not None and closest_video_match != origin_token
                    and db_manager.get_details_using_origin(closest_video_match, []) is None):
                symbolic_token = generate_symbolic_token(closest_video_match, results['token'])
                file_manager.create_symlink(file_manager.generate_filepath(results['token']), symbolic_token)
                # Everything is the same aside from the id_token, which is the symbolic token, and the origin_token,
                # which is the closest video match.
                db_manager.insert_or_update_details(
                    symbolic_token,
                    {
                        'duration': results['duration'],
                        'origin_token': closest_video_match,
                        'date_added': current_datetime
                    }
                )
                # We make the symlink file the closest match of the main file (to make sure closest match refs flow in
                # the same direction).
                db_manager.insert_or_update_closest_match(results['token'], {
                    'most_similar_token': symbolic_token
                })
    return results


def reextract_cached_audio(token, file_manager):
    video_db_manager = VideoDBCachingManager()
    closest_token = video_db_manager.get_closest_match(token)
    audio_db_manager = AudioDBCachingManager()
    existing_audio_own = audio_db_manager.get_details_using_origin(token, cols=['duration'])
    if closest_token is not None and closest_token != token:
        existing_audio_closest = audio_db_manager.get_details_using_origin(closest_token,
                                                                           cols=['duration'])
    else:
        existing_audio_closest = None
    if existing_audio_own is None and existing_audio_closest is None:
        return {
            'token': None,
            'fresh': False,
            'duration': 0.0
        }
    existing_audio = existing_audio_own if existing_audio_own is not None else existing_audio_closest
    token_to_use_as_name = existing_audio[0]['id_token']
    output_filename_with_path = file_manager.generate_filepath(token_to_use_as_name)
    input_filename_with_path = file_manager.generate_filepath(token)
    output_filename, _ = extract_audio_from_video(
        input_filename_with_path, output_filename_with_path, token_to_use_as_name
    )
    # If there was an error of any kind (e.g. non-existing video file), the returned token will be None
    if output_filename is None:
        return {
            'token': None,
            'fresh': False,
            'duration': 0.0
        }

    return {
        'token': output_filename,
        'fresh': True,
        'duration': existing_audio[0]['duration']
    }


def compute_audio_fingerprint(results, file_manager, force=False):
    token = results['token']
    # Making sure that the cache row for the audio file already exists.
    # This cache row is created when the audio is extracted from its corresponding video, so it must exist!
    # We also need this cache row later in order to be able to return the duration of the audio file.
    db_manager = AudioDBCachingManager()
    existing = db_manager.get_details(token, cols=['fingerprint', 'duration'],
                                      using_most_similar=False)[0]
    if existing is None:
        return {
            'result': None,
            'fp_token': None,
            'perform_lookup': False,
            'fresh': False,
            'duration': 0.0,
            'original_results': results
        }
    if not force and existing['fingerprint'] is not None:
        fp = existing['fingerprint']
        fresh = False
        perform_lookup = False
        fp_token = None
    else:
        fp = perceptual_hash_audio(file_manager.generate_filepath(token))
        fresh = fp is not None
        perform_lookup = fp is not None
        fp_token = token if fp is not None else None
    return {
        'result': fp,
        'fp_token': fp_token,
        'perform_lookup': perform_lookup,
        'fresh': fresh,
        'duration': existing['duration'],
        'original_results': results
    }


def compute_audio_fingerprint_callback(results, force=False):
    if results['fresh']:
        token = results['fp_token']
        values = {
            'fingerprint': results['result']
        }
        closest_fp_token = database_callback_generic(token, AudioDBCachingManager(), values,
                                                     force, True)
        if closest_fp_token != token:
            results['fp_token'] = closest_fp_token
    return results


def cache_lookup_detect_slides(token):
    video_db_manager = VideoDBCachingManager()
    # Retrieving the closest match of the current video
    closest_token = video_db_manager.get_closest_match(token)
    slide_db_manager = SlideDBCachingManager()
    existing_slides_own = slide_db_manager.get_details_using_origin(token, cols=['slide_number', 'timestamp'])
    if closest_token is not None and closest_token != token:
        existing_slides_closest = slide_db_manager.get_details_using_origin(closest_token,
                                                                            cols=['slide_number', 'timestamp'])
    else:
        existing_slides_closest = None
    # We first look at the video's own existing slides, then at those of the closest match because the video's
    # own precomputed slides (if any) take precedence.
    all_existing = [existing_slides_own, existing_slides_closest]
    for existing_slides in all_existing:
        if existing_slides is not None:
            print('Returning cached result')
            return {
                'fresh': False,
                'slide_tokens': {
                    x['slide_number']: {
                        'token': x['id_token'],
                        'timestamp': int(x['timestamp']),
                        'token_status': get_image_token_status(x['id_token'])
                    }
                    for x in existing_slides
                }
            }

    return None


def extract_and_sample_frames(token, file_manager):
    # Extracting frames
    print('Extracting frames...')
    output_folder = token + '_all_frames'
    output_folder = extract_frames(file_manager.generate_filepath(token),
                                   file_manager.generate_filepath(output_folder),
                                   output_folder)
    # If there was an error of any kind (e.g. non-existing video file), the returned token will be None
    if output_folder is None:
        return {
            'result': None,
            'sample_indices': None,
            'fresh': False
        }
    # Generating frame sample indices
    frame_indices = generate_frame_sample_indices(file_manager.generate_filepath(output_folder))
    return {
        'result': output_folder,
        'sample_indices': frame_indices,
        'fresh': True
    }


def compute_noise_level_parallel(results, i, n, language, file_manager, nlp_model):
    if not results['fresh']:
        return {
            'result': None,
            'sample_indices': None,
            'noise_level': None,
            'fresh': False
        }
    all_sample_indices = results['sample_indices']
    start_index = int(i * len(all_sample_indices) / n)
    end_index = int((i + 1) * len(all_sample_indices) / n)
    current_sample_indices = all_sample_indices[start_index:end_index]
    noise_level_list = compute_ocr_noise_level(
        file_manager.generate_filepath(results['result']),
        current_sample_indices,
        nlp_model,
        language=language
    )
    return {
        'result': results['result'],
        'sample_indices': results['sample_indices'],
        'noise_level': noise_level_list,
        'fresh': True
    }


def compute_noise_threshold_callback(results, hash_thresh):
    if not results[0]['fresh']:
        return {
            'result': None,
            'sample_indices': None,
            'threshold': None,
            'fresh': False
        }
    list_of_noise_value_lists = [x['noise_level'] for x in results]
    all_noise_values = list(chain.from_iterable(list_of_noise_value_lists))
    threshold = compute_ocr_threshold(all_noise_values)
    return {
        'result': results[0]['result'],
        'sample_indices': results[0]['sample_indices'],
        'threshold': threshold,
        'hash_threshold': hash_thresh,
        'fresh': True
    }


def compute_slide_transitions_parallel(results, i, n, language, file_manager, nlp_model):
    if not results['fresh']:
        return {
            'result': None,
            'transitions': None,
            'threshold': None,
            'hash_threshold': None,
            'fresh': False
        }
    all_sample_indices = results['sample_indices']
    start_index = int(i * len(all_sample_indices) / n)
    end_index = int((i + 1) * len(all_sample_indices) / n)
    current_sample_indices = all_sample_indices[start_index:end_index]
    slide_transition_list = compute_video_ocr_transitions(
        file_manager.generate_filepath(results['result']),
        current_sample_indices,
        results['threshold'],
        results['hash_threshold'],
        nlp_model,
        language=language,
        keep_first=True
    )
    return {
        'result': results['result'],
        'transitions': slide_transition_list,
        'threshold': results['threshold'],
        'hash_threshold': results['hash_threshold'],
        'fresh': True
    }


def compute_slide_transitions_callback(results, language, file_manager, nlp_model):
    if not results[0]['fresh']:
        return {
            'result': None,
            'slides': None,
            'fresh': False
        }
    # Cleaning up the slides in-between slices
    original_list_of_slide_transition_lists = [x['transitions'] for x in results]
    original_list_of_slide_transition_lists = [x for x in original_list_of_slide_transition_lists if len(x) > 0]
    list_of_slide_transition_lists = list()
    for i in range(len(original_list_of_slide_transition_lists) - 1):
        l1 = original_list_of_slide_transition_lists[i]
        l2 = original_list_of_slide_transition_lists[i + 1]
        t_check, d, s_hash = check_ocr_and_hash_thresholds(file_manager.generate_filepath(results[0]['result']),
                                                           l1[-1], l2[0],
                                                           results[0]['threshold'],
                                                           results[0]['hash_threshold'],
                                                           nlp_model,
                                                           language)
        if not t_check:
            l1 = l1[:-1]
        if len(l1) > 0:
            list_of_slide_transition_lists.append(l1)
    list_of_slide_transition_lists.append(original_list_of_slide_transition_lists[-1])
    all_transitions = list(chain.from_iterable(list_of_slide_transition_lists))
    # Making doubly sure there are no duplicates
    all_transitions = sorted(list(set(all_transitions)))
    return {
        'result': results[0]['result'],
        'slides': all_transitions,
        'fresh': True
    }


def detect_slides_callback(results, token, file_manager, force=False, attempt=0):
    slide_tokens = None
    if results['fresh']:
        db_manager = SlideDBCachingManager()
        #####################################
        # Deleting pre-existing cached slides
        #####################################
        if force:
            # If force=True, then there's the possibility that the cache contains previously-extracted slides.
            # Since the new slides and the old slides may not be 100% identical,
            # the old cache rows need to be deleted first.
            existing_slides_own = db_manager.get_details_using_origin(token, cols=[])
            if existing_slides_own is not None:
                db_manager.delete_cache_rows([x['id_token'] for x in existing_slides_own])

        ###############################################
        # Removing non-slide frames and leftover slides
        ###############################################
        # Delete non-slide frames from the frames directory
        list_of_slides = [(FRAME_FORMAT_PNG) % (x) for x in results['slides']]
        list_of_ocr_results = [(TESSERACT_OCR_FORMAT) % (x) for x in results['slides']]
        base_folder = results['result']
        base_folder_with_path = file_manager.generate_filepath(base_folder)
        if attempt <= 1:
            # We only do this on retry #1, because otherwise, the _all_frames folder no longer exists
            for f in os.listdir(base_folder_with_path):
                if f not in list_of_slides and f not in list_of_ocr_results:
                    os.remove(os.path.join(base_folder_with_path, f))
        # Renaming the `all_frames` directory to `slides`
        slides_folder = base_folder.replace('_all_frames', '_slides')
        slides_folder_with_path = file_manager.generate_filepath(slides_folder)
        # Make sure the slides and all_frames folders don't both exist. If that is the case, it means that
        # the slides folder is left over from before (because force==True), so we have to delete it (recursively)
        # before we rename _all_frames to _slides.
        if os.path.exists(slides_folder_with_path) and os.path.exists(base_folder_with_path):
            shutil.rmtree(slides_folder_with_path)
        # Now rename _all_frames to _slides
        if os.path.exists(base_folder_with_path):
            os.rename(base_folder_with_path, slides_folder_with_path)
        else:
            # If the _all_frames folder doesn't exist, assert that the slides folder does!
            assert os.path.exists(slides_folder_with_path)

        ####################################
        # Result formatting and DB insertion
        ####################################
        slide_tokens = [os.path.join(slides_folder, s) for s in list_of_slides]
        ocr_tokens = [os.path.join(slides_folder, s) for s in list_of_ocr_results]
        slide_tokens = {i + 1: {'token': slide_tokens[i], 'timestamp': results['slides'][i]}
                        for i in range(len(slide_tokens))}
        ocr_tokens = {i + 1: ocr_tokens[i] for i in range(len(ocr_tokens))}
        current_datetime = get_current_datetime()
        # Inserting fresh results into the database
        for slide_number in slide_tokens:
            db_manager.insert_or_update_details(
                slide_tokens[slide_number]['token'],
                {
                    'origin_token': token,
                    'timestamp': slide_tokens[slide_number]['timestamp'],
                    'slide_number': slide_number,
                    'ocr_tesseract_results': read_txt_gz_file(
                        file_manager.generate_filepath(ocr_tokens[slide_number])),
                    'date_added': current_datetime
                }
            )
        if not force:
            # If the force flag is False, we may need to propagate the results of this computation to its closest match.
            # The propagation happens if:
            # 1. The token has a closest match (that isn't itself)
            # 2. The closest match does NOT have cached slide results
            video_db_manager = VideoDBCachingManager()
            closest_video_match = video_db_manager.get_closest_match(token)
            if (closest_video_match is not None and closest_video_match != token
                    and db_manager.get_details_using_origin(closest_video_match, []) is None):
                for slide_number in slide_tokens:
                    # For each slide, we get its token (which is the name of its file) and create a new file with a new
                    # token that has a symlink to the actual slide file.
                    current_token = slide_tokens[slide_number]['token']
                    symbolic_token = generate_symbolic_token(closest_video_match, current_token)
                    file_manager.create_symlink(file_manager.generate_filepath(current_token), symbolic_token)
                    # Everything is the same aside from the id_token, which is the symbolic token, and the origin_token,
                    # which is the closest video match.
                    db_manager.insert_or_update_details(
                        symbolic_token,
                        {
                            'origin_token': closest_video_match,
                            'timestamp': slide_tokens[slide_number]['timestamp'],
                            'slide_number': slide_number,
                            'ocr_tesseract_results': read_txt_gz_file(
                                file_manager.generate_filepath(ocr_tokens[slide_number])),
                            'date_added': current_datetime
                        }
                    )
                    # We make the symlink file the closest match of the main file (to make sure
                    # closest match refs flow in the same direction).
                    db_manager.insert_or_update_closest_match(current_token, {
                        'most_similar_token': symbolic_token
                    })
    return {
        'slide_tokens': slide_tokens,
        'fresh': results['fresh']
    }


def reextract_cached_slides(token, file_manager):
    video_db_manager = VideoDBCachingManager()
    closest_token = video_db_manager.get_closest_match(token)
    slide_db_manager = SlideDBCachingManager()
    existing_slides_own = slide_db_manager.get_details_using_origin(token, cols=['slide_number', 'timestamp'])
    if closest_token is not None and closest_token != token:
        existing_slides_closest = slide_db_manager.get_details_using_origin(closest_token,
                                                                            cols=['slide_number', 'timestamp'])
    else:
        existing_slides_closest = None
    if existing_slides_own is None and existing_slides_closest is None:
        return {
            'slide_tokens': None,
            'fresh': False
        }
    existing_slides = existing_slides_own if existing_slides_own is not None else existing_slides_closest
    output_folder = existing_slides[0]['id_token'].split('/')[0]
    timestamps_to_keep = sorted([x['timestamp'] for x in existing_slides])
    output_folder_with_path = file_manager.generate_filepath(output_folder)
    # If the slides folder already exists, it needs to be deleted recursively
    if os.path.exists(output_folder_with_path):
        shutil.rmtree(output_folder_with_path)
    output_folder = extract_frames(file_manager.generate_filepath(token),
                                   output_folder_with_path,
                                   output_folder)
    # If there was an error of any kind (e.g. non-existing video file), the returned token will be None
    if output_folder is None:
        return {
            'slide_tokens': None,
            'fresh': False
        }
    slides_to_timestamps = {FRAME_FORMAT_PNG % x: x for x in timestamps_to_keep}
    # Fix rounding errors in frame extraction
    list_of_extracted_frames = os.listdir(output_folder_with_path)
    slides_not_in_frames = set(slides_to_timestamps.keys()) - set(list_of_extracted_frames)
    timestamps_not_in_frames = [slides_to_timestamps[x] for x in slides_not_in_frames]
    for timestamp in timestamps_not_in_frames:
        if timestamp == 0:
            copy_file_within_folder(output_folder_with_path,
                                    FRAME_FORMAT_PNG % (timestamp + 1),
                                    FRAME_FORMAT_PNG % timestamp)
        if timestamp == max(timestamps_to_keep):
            copy_file_within_folder(output_folder_with_path,
                                    max(list_of_extracted_frames),
                                    FRAME_FORMAT_PNG % timestamp)
    # Remove unused frames
    for f in list_of_extracted_frames:
        if f not in slides_to_timestamps:
            os.remove(os.path.join(output_folder_with_path, f))
    slide_tokens = [os.path.join(output_folder, s) for s in slides_to_timestamps]
    slide_tokens = {i + 1: {'token': slide_tokens[i], 'timestamp': timestamps_to_keep[i]}
                    for i in range(len(slide_tokens))}
    return {
        'slide_tokens': slide_tokens,
        'fresh': True
    }


def compute_slide_fingerprint(token, file_manager):
    # Making sure the slide's cache row exists, because otherwise, the operation should be cancelled!
    db_manager = SlideDBCachingManager()
    existing_slide_list = db_manager.get_details(token, cols=[], using_most_similar=False)
    if existing_slide_list[0] is None:
        return {
            'result': None,
            'fp_token': None,
            'perform_lookup': False,
            'fresh': False
        }
    fingerprint = perceptual_hash_image(file_manager.generate_filepath(token))
    if fingerprint is None:
        return {
            'result': None,
            'fp_token': None,
            'perform_lookup': False,
            'fresh': False
        }
    return {
        'result': fingerprint,
        'fp_token': token,
        'perform_lookup': True,
        'fresh': True
    }


def compute_slide_set_fingerprint(results, origin_token, file_manager):
    # Making sure the cache rows exist, because otherwise, the operation should be cancelled!
    db_manager = SlideDBCachingManager()
    existing_slide_list = db_manager.get_details_using_origin(origin_token, cols=['fingerprint'])
    if existing_slide_list is None:
        return {
            'result': None,
            'fp_token': None,
            'perform_lookup': False,
            'fresh': False,
            'original_results': results
        }
    if all(existing_slide['fingerprint'] is not None for existing_slide in existing_slide_list):
        return {
            'result': None,
            'fp_token': None,
            'perform_lookup': False,
            'fresh': False,
            'original_results': results
        }
    tokens = [existing_slide['id_token'] for existing_slide in existing_slide_list
              if existing_slide['fingerprint'] is None]
    fingerprints = [perceptual_hash_image(file_manager.generate_filepath(token)) for token in tokens]
    if any(fp is None for fp in fingerprints):
        return {
            'result': None,
            'fp_token': None,
            'perform_lookup': False,
            'fresh': False,
            'original_results': results
        }
    return {
        'result': fingerprints,
        'fp_token': tokens,
        'perform_lookup': True,
        'fresh': True,
        'original_results': results
    }


def compute_slide_fingerprint_callback(results, force=False):
    if results['fresh']:
        tokens = results['fp_token']
        fp_results = results['result']
        if not isinstance(tokens, list):
            tokens = [tokens]
            fp_results = [fp_results]
        fp_tokens_to_pass_on = list()
        for i in range(len(tokens)):
            token = tokens[i]
            current_fp_result = fp_results[i]
            db_manager = SlideDBCachingManager()
            values = {
                'fingerprint': current_fp_result,
            }
            closest_token = database_callback_generic(token, db_manager, values, force, True)
            fp_tokens_to_pass_on.append(closest_token)
        # Now we add the correct fp tokens to pass to the fingerprint closest match lookups
        if isinstance(results['fp_token'], list):
            results['fp_token'] = fp_tokens_to_pass_on
        else:
            results['fp_token'] = fp_tokens_to_pass_on[0]
    return results


def retrieve_slide_fingerprint_callback(results):
    return retrieve_fingerprint_callback(results, SlideDBCachingManager(), True)


def ignore_slide_fingerprint_results_callback(results):
    # Ignoring the fingerprinting results and returning the results relevant to the task chain.
    # Used in tasks like transcription and OCR, where fingerprinting is performed before the task itself, but where
    # the results of the fingerprinting are not returned.
    results_to_return = results['fp_results']['original_results']
    slide_tokens = results_to_return['slide_tokens']
    fresh = results_to_return['fresh']
    if slide_tokens is not None:
        for slide_number in slide_tokens:
            slide_tokens[slide_number]['token_status'] = get_image_token_status(slide_tokens[slide_number]['token'])
    return {
        'slide_tokens': slide_tokens,
        'fresh': fresh
    }


def ignore_audio_fingerprint_results_callback(results):
    return ignore_fingerprint_results_callback(results, get_audio_token_status)


def retrieve_audio_fingerprint_callback(results):
    return retrieve_fingerprint_callback(results, AudioDBCachingManager(), True)


def retrieve_video_fingerprint_callback(results):
    return retrieve_fingerprint_callback(results, VideoDBCachingManager(), False)


def ignore_video_fingerprint_results_callback(results):
    return ignore_fingerprint_results_callback(results, get_video_token_status)
