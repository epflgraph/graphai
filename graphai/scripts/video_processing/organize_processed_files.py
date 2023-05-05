from graphai.core.common.caching import SlideDBCachingManager, VideoConfig
from graphai.core.common.video import generate_random_token, read_json_gz_file, FRAME_FORMAT_JPG, \
    TESSERACT_OCR_FORMAT, detect_text_language, write_txt_gz_file, perceptual_hash_image
import os
import argparse

cache_path_manager = VideoConfig()
slide_db_manager = SlideDBCachingManager()


def extract_video_key(file_path):
    path_parts = file_path.split('/')
    return path_parts[-3] + '/' + path_parts[-2]


def extract_origin_folder(file_path):
    return file_path.split('/')[-4]


def extract_slide_frame_number(file_path):
    return int(file_path.split('/')[-1].split('_')[2])


def extract_google_ocr_frame_number(file_path):
    return int(file_path.split('/')[-1].split('_')[5])


def extract_tesseract_ocr_frame_number(file_path):
    return int(file_path.split('/')[-1].split('.')[0])


def extract_file_extension(file_path):
    return '.' + '.'.join(file_path.split('/')[-1].split('.')[1:])


def extract_file_name(file_path):
    return file_path.split('/')[-1]


def generate_video_token(file_extension):
    return generate_random_token() + file_extension


def generate_slide_token(video_token, frame_index):
    return video_token + '_slides/' + ((FRAME_FORMAT_JPG) % frame_index)


def create_symlink(old_path, new_filename):
    new_path = cache_path_manager.generate_filepath(new_filename)
    os.symlink(old_path, new_path)
    return new_path


def organize_processed_file(src_path, index_in_folder, video_tokens):
    """
    Creates symlinks/files for an already processed file and updates the caching tables
    Args:
        src_path: Full path of the file
        index_in_folder: The file's index in its parent folder, used for slide numbers
        video_tokens: Dictionary mapping channel/video values to randomly generated tokens that are the basis
            of all the other tokens (slide and ocr)

    Returns:
        None
    """
    file_name = extract_file_name(src_path)
    file_extension = extract_file_extension(src_path)
    video_key = extract_video_key(src_path)
    origin_folder = extract_origin_folder(src_path)

    # Creating new symlinks or files
    if origin_folder == 'video_lectures':
        new_file_name = generate_video_token(file_extension)
        video_tokens[video_key] = new_file_name
        # Creating a symbolic link to the video file in cache file structure
        create_symlink(src_path, new_file_name)
    else:
        # If it's not a video, it's slide-related
        video_token = video_tokens[video_key]
        if origin_folder == 'final_slide_files':
            # Slide file
            frame_index = extract_slide_frame_number(src_path)
            slide_token = generate_slide_token(video_token, frame_index)
            new_file_name = slide_token
            # Creating a symbolic link to the image file in cache file structure
            new_file_path = create_symlink(src_path, new_file_name)
            # Computing the file's fingerprint
            fingerprint = perceptual_hash_image(new_file_path)
            # Inserting the token and its details into the cache table
            slide_db_manager.insert_or_update_details(
                slide_token,
                {
                    'origin_token': video_token,
                    'timestamp': frame_index,
                    'slide_number': index_in_folder,
                    'fingerprint': fingerprint
                }
            )
        elif origin_folder == 'slides_ocr_google':
            # Google OCR results file
            frame_index = extract_google_ocr_frame_number(src_path)
            slide_token = generate_slide_token(video_token, frame_index)
            if '_dtd.' in file_name:
                # Method 1 is document text detection
                new_file_name = slide_token + '_ocr_google_1_token.txt.gz'
                col_name = 'ocr_google_1_token'
            else:
                # Method 1 is text detection
                new_file_name = slide_token + '_ocr_google_2_token.txt.gz'
                col_name = 'ocr_google_2_token'
            # Reading the contents of the json.gz file in order to write them to a txt.gz file
            new_contents = read_json_gz_file(src_path)['fullTextAnnotation']['text']
            write_txt_gz_file(new_contents, cache_path_manager.generate_filepath(new_file_name))
            if '_dtd.' in file_name:
                # We detect the language of the slide using the results of method 1
                language = detect_text_language(new_contents)
                # Update the respective row with the ocr token and language
                slide_db_manager.insert_or_update_details(
                    slide_token,
                    {
                        col_name: new_file_name,
                        'language': language
                    }
                )
            else:
                # Update the respective row with the ocr token
                slide_db_manager.insert_or_update_details(
                    slide_token,
                    {
                        col_name: new_file_name
                    }
                )
        elif origin_folder == 'frames_ocr_tessaract':
            # Tesseract OCR results
            frame_index = extract_tesseract_ocr_frame_number(src_path)
            slide_token = generate_slide_token(video_token, frame_index)
            new_file_name = video_token + '_slides/' + (TESSERACT_OCR_FORMAT) % frame_index
            # Creating a symlink for the txt.gz file
            create_symlink(src_path, new_file_name)
            # Updating the cache with the ocr token
            slide_db_manager.insert_or_update_details(
                slide_token,
                {
                    'ocr_tesseract_token': new_file_name
                }
            )


def handle_already_processed_files(root_dir):
    folders_to_process = ['mined/video_lectures',
                          'modelled/img/final_slide_files',
                          'modelled/json/slides_ocr_google',
                          'modelled/raw/frames_ocr_tessaract']
    video_tokens = dict()
    for base_folder in folders_to_process:
        base_path = os.path.join(root_dir, base_folder)
        for channel_folder in os.listdir(base_path):
            channel_path = os.path.join(base_path, channel_folder)
            if not os.path.isdir(channel_path):
                continue
            for video_folder in os.listdir(channel_path):
                video_folder_path = os.path.join(channel_path, video_folder)
                if not os.path.isdir(video_folder_path):
                    continue
                folder_index_counter = 1
                for filename in os.listdir(video_folder_path):
                    file_path = os.path.join(video_folder_path, filename)
                    if base_folder == 'modelled/img/final_slide_files':
                        if '_R_1280x720_Q_80' not in filename:
                            continue
                    elif base_folder == 'mined/video_lectures':
                        if '_video_file_' not in filename:
                            continue
                    organize_processed_file(file_path, folder_index_counter, video_tokens)
                    print(f"{file_path} processed!")
                    folder_index_counter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='Root directory that contains the folders "mined" and "modelled"')
    args = parser.parse_args()
    handle_already_processed_files(args.root_dir)