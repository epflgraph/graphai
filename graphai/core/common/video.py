import os
import time
import xml.etree.cElementTree as ET
import wget
import re
import ffmpeg

ROOT_VIDEO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../Storage/'))
STANDARD_FPS = 30


def retrieve_file_from_url(url, out_filename=None):
    if out_filename is None:
        out_filename = url.split('/')[-1]
    response = wget.download(url, out_filename)
    return response


def hash_video_or_audio(input_filename, video=True):
    in_stream = ffmpeg.input(input_filename)
    if video:
        # video
        in_stream = in_stream.video
    else:
        # audio
        in_stream = in_stream.audio
    result, _ = ffmpeg.output(in_stream, 'pipe:', format='md5').run(capture_stdout=True)
    # The result looks like 'MD5=9735151f36a3e628b0816b1bba3b9640\n' so we clean it up
    return (result.decode('utf8').strip())[4:]


def compute_signature(input_filename, output_filename='sig.xml'):
    ffmpeg.input(input_filename).video.filter('fps', STANDARD_FPS).\
        filter('signature', format='xml', filename=output_filename). \
        output('pipe:', format='null').run(capture_stdout=False)


def compare_signatures(input_filename_1, input_filename_2, output_template='sig%d.xml'):
    in1 = ffmpeg.input(input_filename_1).video.filter('fps', STANDARD_FPS)
    in2 = ffmpeg.input(input_filename_2).video.filter('fps', STANDARD_FPS)
    combined_input = ffmpeg.filter((in1, in2), 'signature', nb_inputs=2,
                            format='xml', filename=output_template,
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