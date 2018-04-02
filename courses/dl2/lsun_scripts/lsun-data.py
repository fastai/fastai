from __future__ import print_function
from tqdm import tqdm
import argparse, cv2, lmdb, numpy, os
from os.path import exists, join

__author__ = 'Fisher Yu'
__email__ = 'fy@cs.princeton.edu'
__license__ = 'MIT'
# (Minor edits by Jeremy Howard)


def export_images(db_path, out_dir, flat=False):
    print('Exporting', db_path, 'to', out_dir)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in tqdm(cursor):
            key = key.decode()
            if not flat: image_out_dir = join(out_dir, '/'.join(key[:3]))
            else: image_out_dir = out_dir
            if not exists(image_out_dir): os.makedirs(image_out_dir)
            image_out_path = join(image_out_dir, key + '.jpg')
            with open(image_out_path, 'wb') as fp: fp.write(val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lmdb_path', nargs='+', type=str,
                        help='The path to the lmdb database folder. '
                             'Support multiple database paths.')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--flat', action='store_true',
                        help='If enabled, the images are imported into output '
                             'directory directly instead of hierarchical '
                             'directories.')
    args = parser.parse_args()
    lmdb_paths = args.lmdb_path
    for lmdb_path in lmdb_paths: export_images(lmdb_path, args.out_dir, args.flat)


if __name__ == '__main__': main()
