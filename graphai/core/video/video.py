import os
import sys
import math
import glob
import re
import random
import json
import time
from datetime import timedelta
import gzip

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

from graphai.core.interfaces.caching import VideoConfig, VideoDBCachingManager, get_token_file_status, is_fingerprinted, \
    SlideDBCachingManager, AudioDBCachingManager
from graphai.core.interfaces.config import config
from graphai.core.common.common_utils import (
    make_sure_path_exists,
    file_exists
)
from graphai.core.common.fingerprinting import (
    perceptual_hash_image,
    compare_encoded_fingerprints
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
        return None


def retrieve_file_from_url(url, output_filename_with_path, output_token, is_kaltura=False):
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
    return [(x['codec_type'], x['codec_name']) for x in results['streams']]


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
        # DO NOT CHANGE r=1 HERE
        # This parameter ensures that one frame is extracted per second, and the whole logic of the algorithm
        # relies on timestamp being identical to frame number.
        print('Starting ffmpeg slide extraction...')
        if video_stream_name != 'png':
            # If the video stream is NOT a single picture, we have to apply the "fps" filter to get one frame/second.
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


def get_ocr_colnames(method):
    if method == 'tesseract':
        return ['ocr_tesseract_results']
    else:
        return ['ocr_google_1_results', 'ocr_google_2_results']


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
