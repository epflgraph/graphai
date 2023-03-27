import random
import re
import sys
import time
from datetime import datetime
import numpy as np

import ffmpeg

import chromaprint
import wget
import acoustid
from fuzzywuzzy import fuzz

from graphai.core.common.video import file_exists, \
    STANDARD_FPS, TEMP_SUBFOLDER, VideoConfig, count_files, get_dir_files, SLIDE_OUTPUT_FORMAT, DBCachingManager


video_config = VideoConfig()
video_db_manager = DBCachingManager()


def generate_random_token():
    return ('%.06f' % time.time()).replace('.','') + '%08d'%random.randint(0,int(1e7))


def retrieve_file_from_url(url, output_filename_with_path, output_token):
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
    try:
        probe_results = perform_probe(input_filename_with_path)
    except Exception as e:
        print(e, file=sys.stderr)
        return None
    audio_type = probe_results['streams'][1]['codec_name']
    # note to self: maybe we should skip the probing and just put this in an aac file in any case
    output_suffix='_audio.' + audio_type
    output_token = input_token + output_suffix
    return output_token, float(probe_results['format']['duration'])


def extract_audio_from_video(input_filename_with_path, output_filename_with_path, output_token):
    # TODO Take all the `force` flags out of here and into celery_tasks/video.py or voice.py
    # TODO Also remove all the existence checks
    # TODO consolidate filename generation into one function that can be called from elsewhere and
    #  make all these functions receive both the base name and the name with the path, the former being
    #  the return value in case of success.
    if not file_exists(input_filename_with_path):
        print(f'ffmpeg error: File {input_filename_with_path} does not exist')
        return None
    try:
        err = ffmpeg.input(input_filename_with_path).audio. \
            output(output_filename_with_path, c='copy').\
            overwrite_output().run(capture_stdout=True)
    except Exception as e:
        print(e, file=sys.stderr)
        err = str(e)

    if file_exists(output_filename_with_path) and ('ffmpeg error' not in err):
        return output_token
    else:
        return None


def find_beginning_and_ending_silences(input_filename_with_path, distance_from_end_tol=0.01, noise_thresh=0.0001):
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


def perceptual_hash_audio(input_filename_with_path, max_length=3600):
    if not file_exists(input_filename_with_path):
        print(f'File {input_filename_with_path} does not exist')
        return None, None
    results = acoustid.fingerprint_file(input_filename_with_path, maxlength=max_length)
    fingerprint = results[1]
    decoded = chromaprint.decode_fingerprint(fingerprint)
    return fingerprint.decode('utf8'), decoded


def compare_audio_fingerprints(decoded_1, decoded_2):
    return fuzz.ratio(decoded_1, decoded_2) / 100


def compare_encoded_audio_fingerprints(f1, f2=None):
    # when fuzzywuzzy is used in combination with python-Levenshtein (fuzzywuzzy[speedup],
    # there's a 10-fold speedup here.
    if f2 is None:
        return 0
    return compare_audio_fingerprints(chromaprint.decode_fingerprint(f1.encode('utf8')),
                                        chromaprint.decode_fingerprint(f2.encode('utf8')))


def find_closest_audio_fingerprint(fp, fp_list, token_list, min_similarity=0.8):
    if len(fp_list) == 0:
        return None, None, None
    fp_similarities = np.array([compare_encoded_audio_fingerprints(fp, fp2) for fp2 in fp_list])
    max_index = np.argmax(fp_similarities)
    if fp_similarities[max_index] >= min_similarity:
        return token_list[max_index], fp_list[max_index], fp_similarities[max_index]
    else:
        return None, None, None


def compute_mpeg7_signature(input_filename, output_suffix='_sig.xml', force=False):
    input_filename_with_path = video_config.generate_filename(input_filename)
    output_filename = input_filename + output_suffix
    output_filename_with_path = video_config.generate_filename(output_filename)

    # If force=False, check whether the file has already been computed, return existing result if so
    if not force and file_exists(output_filename_with_path):
        print('Result already exists, returning cached result')
        return output_filename, False

    ffmpeg.input(input_filename_with_path).video.filter('fps', STANDARD_FPS).\
        filter('signature', format='xml', filename=output_filename_with_path). \
        output('pipe:', format='null').run(capture_stdout=False)
    if file_exists(output_filename_with_path):
        return output_filename, True
    else:
        return None, False


def compute_video_slides(input_filename, force=False):
    probe_results = perform_probe(video_config.generate_filename(input_filename))
    video_stream = [x for x in probe_results['streams'] if x['codec_type'] == 'video'][0]
    width = video_stream['width']
    height = video_stream['height']

    input_filename_with_path = video_config.generate_filename(input_filename)
    output_template = input_filename + SLIDE_OUTPUT_FORMAT
    output_template_with_path = video_config.generate_filename(output_template)
    output_template_dirs_only = '/'.join(output_template_with_path.split('/')[:-1])

    first_slide = output_template_with_path % 1
    if not force and file_exists(first_slide):
        print('Result already exists, returning cached result')
        return output_template_dirs_only.split('/')[-1], False, count_files(output_template_dirs_only), \
               get_dir_files(output_template_dirs_only)

    try:
        # extract all the keyframes (i-frames) and store them as image files
        ffmpeg.input(input_filename_with_path).video. \
            filter('select', "eq(pict_type,PICT_TYPE_I)").\
            output(output_template_with_path, format='image2', fps_mode='vfr', s=f'{width}x{height}').\
            overwrite_output().run(capture_stdout=False)
    except Exception as e:
        print(e, file=sys.stderr)

    if file_exists(first_slide):
        return output_template_dirs_only.split('/')[-1], True, count_files(output_template_dirs_only), \
               get_dir_files(output_template_dirs_only)
    else:
        return None, False, 0, []



def compare_mpeg7_signatures(input_filename_1, input_filename_2, output_template='comparison%d_sig.xml'):
    input_filename_with_path_1 = video_config.generate_filename(input_filename_1)
    input_filename_with_path_2 = video_config.generate_filename(input_filename_2)
    output_template_with_path = video_config.generate_filename(
        input_filename_1 + '_' + input_filename_2 + '_' + output_template, force_dir=TEMP_SUBFOLDER)

    in1 = ffmpeg.input(input_filename_with_path_1).video.filter('fps', STANDARD_FPS)
    in2 = ffmpeg.input(input_filename_with_path_2).video.filter('fps', STANDARD_FPS)
    combined_input = ffmpeg.filter((in1, in2), 'signature', nb_inputs=2,
                            format='xml', filename=output_template_with_path,
                            detectmode='fast')
    _, results = ffmpeg.output(combined_input, 'pipe:', format='null').run(capture_stderr=True)
    results = re.split(r'[\n\r]', results.decode('utf8'))
    results = [x for x in results if 'matching' in x and 'no matching of' not in x]
    if len(results) > 0:
        matches = [x for x in results if 'frames matching' in x]
        matches = [re.findall(r"[-+]?(?:\d*\.*\d+)", match.split(']')[1]) for match in matches]
        return_dict = {'match': True}
        if any(['whole video matching' in x for x in results]):
            return_dict['whole'] = True
        else:
            return_dict['whole'] = False
        return_dict['matches'] = [
            {
                'start_1': float(match_details[1]),
                'start_2': float(match_details[3]),
                'n_frames': int(match_details[4])
            }
            for match_details in matches
        ]
    else:
        return_dict = {'match': False, 'whole': False, 'matches': []}
    return return_dict