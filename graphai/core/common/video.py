import os
import errno

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


class VideoConfig():
    def __init__(self, root_dir=ROOT_VIDEO_DIR):
        self.root_dir = root_dir


    def concat_file_path(self, filename, subfolder):
        make_sure_path_exists(os.path.join(self.root_dir, subfolder))
        return os.path.join(self.root_dir, subfolder, filename)


    def set_root_dir(self, new_root_dir):
        self.root_dir = new_root_dir
        make_sure_path_exists(new_root_dir)

    def generate_filename(self, filename, force_dir=None):
        if force_dir is not None:
            filename_with_path = self.concat_file_path(filename, force_dir)
        elif any([filename.endswith(x) for x in VIDEO_FORMATS]):
            filename_with_path = self.concat_file_path(filename, VIDEO_SUBFOLDER)
        elif any([filename.endswith(x) for x in AUDIO_FORMATS]):
            filename_with_path = self.concat_file_path(filename, AUDIO_SUBFOLDER)
        elif any([filename.endswith(x) for x in SIGNATURE_FORMATS]):
            filename_with_path = self.concat_file_path(filename, SIGNATURE_SUBFOLDER)
        else:
            filename_with_path = self.concat_file_path(filename, OTHER_SUBFOLDER)
        return filename_with_path


