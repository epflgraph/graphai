import os
import errno

import fasttext
from fasttext.util import download_model, reduce_model
from argparse import ArgumentParser
from contextlib import chdir


def make_sure_path_exists(path, full_perm=True):
    """
    Recursively creates the folders in a path.
    Args:
        path: The path that needs to exist (and will thus be created if it doesn't)
        full_perm: If set, the function will assign full permission (chmod 777) to each newly created folder
    Returns:
        None
    """
    if path == '/' or path == '':
        return
    if os.path.isdir(path):
        return
    try:
        parent_path = '/'.join(path.split('/')[:-1])
        make_sure_path_exists(parent_path)
        os.mkdir(path)
        if full_perm:
            os.chmod(path, 0o777)
    except OSError as exception:
        if exception.errno != errno.EEXIST and exception.errno != errno.EPERM:
            raise


def generate_model_filename(lang='en', target_dim=30):
    return f'cc.{lang}.{target_dim}.bin'


def download_model_to_dir(target_dir, lang='en'):
    with chdir(target_dir):
        filename = download_model(lang)
        os.remove(os.path.join(target_dir, filename + '.gz'))
        return os.path.join(target_dir, filename)


def generate_target_path(target_dir, lang, target_dim=30):
    return os.path.join(target_dir, generate_model_filename(lang, target_dim))


def reduce_and_save(source_filename, full_target_path, target_dim=30):
    model = fasttext.load_model(source_filename)
    reduce_model(model, target_dim)
    model.save_model(full_target_path)


def reduce_fasttext_models(root_dir, lang, target_dim):
    make_sure_path_exists(root_dir)
    final_file_name = generate_model_filename(lang, target_dim)
    original_file_name = generate_model_filename(lang, 300)
    if os.path.exists(os.path.join(root_dir, final_file_name)):
        print('Final model file already exists, exiting...')
        return

    if os.path.exists(os.path.join(root_dir, original_file_name)):
        print('Skipping download as base (300-dim) model already exists.')
        source_filename = os.path.join(root_dir, original_file_name)
    else:
        source_filename = download_model_to_dir(root_dir, lang)
    target_filename = generate_target_path(root_dir, lang, target_dim)
    reduce_and_save(source_filename, target_filename, target_dim)
    print(f'ft_{lang}={target_filename}')


def reduce_fasttext_models_main():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--dim', type=int, default=30)
    args = parser.parse_args()
    reduce_fasttext_models(args.root_dir, args.lang, args.dim)


if __name__ == '__main__':
    reduce_fasttext_models_main()
