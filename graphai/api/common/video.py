import random
import re
import sys
import time
from datetime import datetime

import ffmpeg

import chromaprint
import wget
import acoustid
from fuzzywuzzy import fuzz

from graphai.core.common.video import file_exists, \
    STANDARD_FPS, TEMP_SUBFOLDER, VideoConfig, count_files, get_dir_files, SLIDE_OUTPUT_FORMAT


video_config = VideoConfig()


def generate_random_token():
    return ('%.06f' % time.time()).replace('.','') + '%08d'%random.randint(0,int(1e7))


def retrieve_file_from_url(url, out_filename=None):
    if out_filename is None:
        out_filename = url.split('/')[-1]
    out_filename_with_path = video_config.generate_filename(out_filename)
    try:
        response = wget.download(url, out_filename_with_path)
    except Exception as e:
        print(e, file=sys.stderr)
        return None
    return out_filename


def perform_probe(filename):
    return ffmpeg.probe(video_config.generate_filename(filename), cmd='ffprobe')


def md5_video_or_audio(input_filename, video=True):
    input_filename_with_path = video_config.generate_filename(input_filename)
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


def extract_audio_from_video(input_filename, force=False):
    try:
        probe_results = perform_probe(input_filename)
    except Exception as e:
        print(e, file=sys.stderr)
        return None, False
    audio_type = probe_results['streams'][1]['codec_name']
    # note to self: maybe we should skip the probing and just put this in an aac file in any case
    output_suffix='_audio.' + audio_type
    input_filename_with_path = video_config.generate_filename(input_filename)
    output_filename = input_filename + output_suffix
    output_filename_with_path = video_config.generate_filename(output_filename)

    # If force=False, check whether the file has already been computed, return existing result if so
    if not force and file_exists(output_filename_with_path):
        print('Result already exists, returning cached result')
        return output_filename, False, float(probe_results['format']['duration'])
    try:
        err = ffmpeg.input(input_filename_with_path).audio. \
            output(output_filename_with_path, c='copy').\
            overwrite_output().run(capture_stdout=True)
    except Exception as e:
        print(e, file=sys.stderr)
        err = str(e)

    if file_exists(output_filename_with_path) and ('ffmpeg error' not in err):
        return output_filename, True, float(probe_results['format']['duration'])
    else:
        return None, False, 0.0


def find_beginning_and_ending_silences(input_filename_with_path, distance_from_end_tol=0.01, noise_thresh=0.0001):
    if not file_exists(input_filename_with_path):
        raise Exception(f'File {input_filename_with_path} does not exist')
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


def remove_silence_doublesided(input_filename, force=False, threshold=0.0001):
    audio_type = input_filename.split('.')[-1]
    output_suffix = '_nosilence.' + audio_type
    input_filename_with_path = video_config.generate_filename(input_filename)
    output_filename = input_filename + output_suffix
    output_filename_with_path = video_config.generate_filename(output_filename)

    # If force=False, check whether the file has already been computed, return existing result if so
    if not force and file_exists(output_filename_with_path):
        print('Result already exists, returning cached result')
        output_probe = perform_probe(output_filename)
        # the audio length is approximate since the bitrate is used to estimate it
        return output_filename, False, float(output_probe['format']['duration'])
    try:
        from_and_to = find_beginning_and_ending_silences(input_filename_with_path, noise_thresh=threshold)
        err = ffmpeg.input(input_filename_with_path).audio. \
            output(output_filename_with_path, c='copy', ss=from_and_to['ss'], to=from_and_to['to']).\
            overwrite_output().run(capture_stdout=True)
    except Exception as e:
        print(e, file=sys.stderr)
        err = str(e)

    if file_exists(output_filename_with_path) and ('ffmpeg error' not in err):
        output_probe = perform_probe(output_filename)
        # the audio length is approximate since the bitrate is used to estimate it
        return output_filename, True, float(output_probe['format']['duration'])
    else:
        return None, False, 0.0


def compare_audio_fingerprints(decoded_1, decoded_2):
    return fuzz.ratio(decoded_1, decoded_2)


def perceptual_hash_audio(input_filename, max_length=3600):
    input_filename_with_path = video_config.generate_filename(input_filename)
    results = acoustid.fingerprint_file(input_filename_with_path, maxlength=max_length)
    length = results[0]
    fingerprint = results[1]
    decoded = chromaprint.decode_fingerprint(fingerprint)
    return fingerprint.decode('utf8'), decoded, length


def compare_string_audio_fingerprints(s1, s2):
    return fuzz.ratio(chromaprint.decode_fingerprint(s1.encode('utf8')),
                      chromaprint.decode_fingerprint(s2.encode('utf8')))


def compute_signature(input_filename, output_suffix='_sig.xml', force=False):
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
    probe_results = perform_probe(input_filename)
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



def compare_signatures(input_filename_1, input_filename_2, output_template='comparison%d_sig.xml'):
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