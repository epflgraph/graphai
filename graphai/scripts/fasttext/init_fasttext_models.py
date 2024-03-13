import os

import fasttext
from fasttext.util import download_model, reduce_model
from argparse import ArgumentParser
from contextlib import chdir
from graphai.core.common.common_utils import make_sure_path_exists


def download_model_to_dir(target_dir, lang='en'):
    with chdir(target_dir):
        filename = download_model(lang)
        os.remove(os.path.join(target_dir, filename + '.gz'))
        return os.path.join(target_dir, filename)


def download_models_to_dir(target_dir):
    return download_model_to_dir(target_dir, 'en'), download_model_to_dir(target_dir, 'fr')


def generate_target_path(target_dir, lang):
    return os.path.join(target_dir, f'cc.{lang}.30.bin')


def reduce_and_save(source_filename, full_target_path, target_dim=30):
    model = fasttext.load_model(source_filename)
    reduce_model(model, target_dim)
    model.save_model(full_target_path)


def reduce_and_save_all(filenames, target_paths, target_dim=30):
    for i in range(len(filenames)):
        reduce_and_save(filenames[i], target_paths[i], target_dim)


def main():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--lang', type=str, choices=['en', 'fr'], required=True)
    args = parser.parse_args()
    root_dir = args.root_dir
    lang = args.lang
    make_sure_path_exists(root_dir)
    source_filename = download_model_to_dir(root_dir, lang)
    target_filename = generate_target_path(root_dir, lang)
    reduce_and_save(source_filename, target_filename, 30)
    print(f'ft_{lang}={target_filename}')


if __name__ == '__main__':
    main()
