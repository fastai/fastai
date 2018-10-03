"""
Script to merge
"""

import argparse
from pathlib import Path
import json
import csv


def get_texts(root):
    for dir_ in root.iterdir():
        for wiki_file in dir_.iterdir():
            with open(wiki_file, encoding='utf-8') as f_in:
                for line in f_in:
                    article = json.loads(line)
                    text = article['text']
                    yield text


def write_file(file_path, text_iter, num_tokens):
    total_num_tokens = 0
    print(f'Writing to {file_path}...')
    j = 0
    with open(file_path, 'w', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        for i, text in enumerate(text_iter):
            j += 1
            writer.writerow([text])
            # f_out.write(text)

            # calculate approximate length based on tokens
            total_num_tokens += len(text.split())
            if total_num_tokens > num_tokens:
                break
            if i % 10000 == 0:
                print('Processed {:,} documents. Total # tokens: {:,}.'.format(i, total_num_tokens))
    print('{}. # documents: {:,}. # tokens: {:,}.'.format(
        file_path, j, total_num_tokens))


def main(args):

    input_path = Path(args.input)
    output = Path(args.output)
    assert input_path.exists(), f'Error: {input_path} does not exist.'
    output.mkdir(exist_ok=True)

    train_path = output.joinpath('train.csv')
    val_path = output.joinpath('val.csv')
    text_iter = get_texts(input_path)
    write_file(train_path, text_iter, args.num_tokens)
    write_file(val_path, text_iter, args.num_tokens / 10)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='the directory where the Wikipedia data extracted '
                             'with WikiExtractor.py is located. Consists of '
                             'directories AA, AB, AC, etc.')
    parser.add_argument('-o', '--output', required=True,
                        help='the output directory where the merged Wikipedia '
                             'documents should be saved')
    parser.add_argument('-n', '--num-tokens', type=int, default=100000000,
                        help='the # of tokens that the merged document should '
                             'contain (default: 100M)')
    args = parser.parse_args()
    main(args)
