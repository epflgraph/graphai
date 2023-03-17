import os
import sys
import time
from urllib.error import HTTPError
import random
import xml.etree.cElementTree as ET
import errno
import wget
import re
import ffmpeg

ROOT_VIDEO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../Storage/'))
VIDEO_SUBFOLDER = 'Video'
VIDEO_FORMATS = ['.mkv', '.mp4', '.avi', '.mov', '.flv']
AUDIO_SUBFOLDER = 'Audio'
AUDIO_FORMATS = ['.mp3', '.flac', '.wav', '.aac']
OTHER_SUBFOLDER = 'Other'
SIGNATURE_SUBFOLDER = 'Signatures'
SIGNATURE_FORMATS = ['_sig.xml']
TEMP_SUBFOLDER = 'Temp'

STANDARD_FPS = 30


def make_sure_path_exists(path, file_at_the_end=False):
    try:
        if not file_at_the_end:
            os.makedirs(path)
        else:
            os.makedirs('/'.join(path.split('/')[:-1]))
        return
    except OSError as exception:
        if exception.errno != errno.EEXIST and exception.errno != errno.EPERM:
            raise


def file_exists(file_path):
    return os.path.exists(file_path)


def concat_file_path(filename, subfolder):
    return os.path.join(ROOT_VIDEO_DIR, subfolder, filename)


def generate_random_token():
    return ('%.06f' % time.time()).replace('.','') + '%08d'%random.randint(0,int(1e7))


def generate_filename(filename, force_dir=None):
    if force_dir is not None:
        filename_with_path = concat_file_path(filename, force_dir)
    elif any([filename.endswith(x) for x in VIDEO_FORMATS]):
        filename_with_path = concat_file_path(filename, VIDEO_SUBFOLDER)
    elif any([filename.endswith(x) for x in AUDIO_FORMATS]):
        filename_with_path = concat_file_path(filename, AUDIO_SUBFOLDER)
    elif any([filename.endswith(x) for x in SIGNATURE_FORMATS]):
        filename_with_path = concat_file_path(filename, SIGNATURE_SUBFOLDER)
    else:
        filename_with_path = concat_file_path(filename, OTHER_SUBFOLDER)
    return filename_with_path


def retrieve_file_from_url(url, out_filename=None):
    if out_filename is None:
        out_filename = url.split('/')[-1]
    out_filename_with_path = generate_filename(out_filename)
    try:
        response = wget.download(url, out_filename_with_path)
    except HTTPError as e:
        print(e)
        return None
    return out_filename


def perform_probe(filename):
    return ffmpeg.probe(generate_filename(filename), cmd='ffprobe')


def hash_video_or_audio(input_filename, video=True):
    input_filename_with_path = generate_filename(input_filename)
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


def extract_audio_from_video(input_filename):
    # TODO Make sure we haven't already computed it, and return the old result if we have
    try:
        probe_results = perform_probe(input_filename)
    except Exception as e:
        print(e, file=sys.stderr)
        return None
    audio_type = probe_results['streams'][1]['codec_name']
    # TODO maybe we should skip the probing and just put this in an aac file in any case
    output_suffix='_audio.' + audio_type
    input_filename_with_path = generate_filename(input_filename)
    output_filename = input_filename + output_suffix
    output_filename_with_path = generate_filename(output_filename)
    try:
        ffmpeg.input(input_filename_with_path).audio. \
            output(output_filename_with_path, c='copy', ss='0').overwrite_output().run(capture_stdout=False)
    except Exception as e:
        print(e, file=sys.stderr)
        return None
    if file_exists(output_filename_with_path):
        return output_filename
    else:
        return None


def compute_signature(input_filename, output_suffix='_sig.xml'):
    input_filename_with_path = generate_filename(input_filename)
    output_filename = input_filename + output_suffix
    output_filename_with_path = generate_filename(output_filename)
    ffmpeg.input(input_filename_with_path).video.filter('fps', STANDARD_FPS).\
        filter('signature', format='xml', filename=output_filename_with_path). \
        output('pipe:', format='null').run(capture_stdout=False)
    if file_exists(output_filename_with_path):
        return output_filename
    else:
        return None


def compare_signatures(input_filename_1, input_filename_2, output_template='comparison%d_sig.xml'):
    input_filename_with_path_1 = generate_filename(input_filename_1)
    input_filename_with_path_2 = generate_filename(input_filename_2)
    output_template_with_path = generate_filename(
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