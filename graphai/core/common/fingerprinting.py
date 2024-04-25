from bisect import bisect

import acoustid
import ffmpeg
import fingerprint
import imagehash
import numpy as np
from PIL import Image
from fuzzywuzzy import fuzz

from graphai.core.common.common_utils import file_exists


def perceptual_hash_text(s):
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
    hash_len = 32
    string_length = len(s)
    window_lengths = [1, 4, 10, 20, 50]
    kgram_lengths = [int(np.ceil(x / 2)) for x in window_lengths]
    length_index = bisect(window_lengths, int(np.ceil(string_length / hash_len))) - 1
    window_length = window_lengths[length_index]
    kgram_length = kgram_lengths[length_index]

    fprinter = fingerprint.Fingerprint(kgram_len=kgram_length, window_len=window_length, base=10, modulo=256)
    try:
        hash_numbers = fprinter.generate(str=s)
        if len(hash_numbers) > hash_len:
            sample_indices = np.linspace(start=0, stop=len(hash_numbers) - 1, num=hash_len - 1, endpoint=False).tolist()
            sample_indices.append(len(hash_numbers) - 1)
            sample_indices = [int(x) for x in sample_indices]
            hash_numbers = [hash_numbers[i] for i in sample_indices]
        elif len(hash_numbers) < hash_len:
            hash_numbers = hash_numbers + [(0, 0)] * (32 - len(hash_numbers))
        fp_result = ''.join([f"{n[0]:02x}" for n in hash_numbers])
    except fingerprint.FingerprintException:
        fp_result = ''.join(['0'] * 64)
    return "%s_%02d_%02d" % (fp_result, window_length, kgram_length)


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
        strip_underscores: For text fingerprints, removes the trailing underscores

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


def find_closest_fingerprint_for_list_from_list(target_fp, fp_list, token_list, date_list, min_similarity=0.8,
                                                decoder_func=imagehash.hex_to_hash, strip_underscores=True):
    if isinstance(target_fp, str):
        return find_closest_fingerprint_from_list(target_fp, fp_list, token_list, date_list, min_similarity,
                                                  decoder_func, strip_underscores)
    else:
        closest_token_list = list()
        closest_fp_list = list()
        best_date_list = list()
        score_list = list()
        for current_fp in target_fp:
            closest_token, closest_fingerprint, best_date, score = find_closest_fingerprint_from_list(
                current_fp, fp_list, token_list, date_list, min_similarity,
                decoder_func, strip_underscores
            )
            closest_token_list.append(closest_token)
            closest_fp_list.append(closest_fingerprint)
            best_date_list.append(best_date)
            score_list.append(score)
        return closest_token_list, closest_fp_list, best_date_list, score_list


def find_closest_audio_fingerprint_from_list(target_fp, fp_list, token_list, date_list, min_similarity=0.8):
    """
    Finds closest audio fingerprint from list
    """
    return find_closest_fingerprint_for_list_from_list(target_fp, fp_list, token_list, date_list, min_similarity,
                                                       decoder_func=imagehash.hex_to_hash)


def find_closest_image_fingerprint_from_list(target_fp, fp_list, token_list, date_list, min_similarity=0.8):
    """
    Finds closest image fingerprint from list
    """

    return find_closest_fingerprint_for_list_from_list(target_fp, fp_list, token_list, date_list, min_similarity,
                                                       decoder_func=imagehash.hex_to_hash)


def find_closest_text_fingerprint_from_list(target_fp, fp_list, token_list, date_list, min_similarity=0.8):
    """
    Finds closest image fingerprint from list
    """
    return find_closest_fingerprint_for_list_from_list(target_fp, fp_list, token_list, date_list, min_similarity,
                                                       decoder_func=imagehash.hex_to_hash, strip_underscores=True)
