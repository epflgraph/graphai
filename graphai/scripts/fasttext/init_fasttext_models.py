import os
import errno

import fasttext
from fasttext.util import download_model, reduce_model
from argparse import ArgumentParser
from contextlib import chdir
from graphai.core.common.common_utils import make_sure_path_exists


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


def init_fasttext_models(root_dir, lang, target_dim):
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


def init_fasttext_models_main():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--lang', type=str, choices=['en', 'fr'], required=True)
    parser.add_argument('--dim', type=int, default=30)
    args = parser.parse_args()
    init_fasttext_models(args.root_dir, args.lang, args.dim)


if __name__ == '__main__':
    init_fasttext_models_main()
