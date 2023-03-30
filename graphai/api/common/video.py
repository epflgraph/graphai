import re
import sys
import ffmpeg

from graphai.core.common.video import file_exists, \
    STANDARD_FPS, TEMP_SUBFOLDER, VideoConfig, count_files, get_dir_files, SLIDE_OUTPUT_FORMAT, DBCachingManager, \
    TranscriptionModel, perform_probe

video_config = VideoConfig()
video_db_manager = DBCachingManager()
transcription_model = TranscriptionModel()


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