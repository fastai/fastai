"""
Script to create small and large WikiText datasets from Wikipedia articles in
any language that were downloaded with `prepare_wiki.sh`.
Articles are tokenized using the Moses tokenizer. Articles with least than
100 tokens are removed.
"""
import argparse
from pathlib import Path
import json

from shutil import copyfile

from sacremoses import MosesTokenizer


def get_texts(root):
    for dir_ in root.iterdir():
        for wiki_file in dir_.iterdir():
            with open(wiki_file, encoding='utf-8') as f_in:
                for line in f_in:
                    article = json.loads(line)
                    text = article['text']
                    title = article['title']
                    if text.strip() == title:
                        # print('No content continuing...')
                        continue
                    yield text


def write_wikitext(file_path, text_iter, mt, num_tokens, mode='w'):
    total_num_tokens = 0
    print(f'Writing to {file_path}...')
    i = 0
    with open(file_path, mode, encoding='utf-8') as f_out:
        for i, text in enumerate(text_iter):

            num_tokens_article = 0  # count the number of tokens in an article
            tokenized_paragraphs = []
            paragraphs = text.split('\n')

            for paragraph in paragraphs:
                tokenized = mt.tokenize(paragraph.strip(), return_str=True)
                tokenized_paragraphs.append(tokenized)

                tokens = tokenized.split(' ')  # split on whitespace to keep newlines
                # don't count empty lines
                tokens = [token for token in tokens if token]

                # calculate length based on tokens; add 1 for newline
                num_tokens_article += len(tokens) + 1

            if num_tokens_article < 100:
                # only use articles that have at least 100 tokens
                continue

            for tokenized in tokenized_paragraphs:
                f_out.write(tokenized + '\n')

            total_num_tokens += num_tokens_article + 1
            if total_num_tokens > num_tokens:
                break
            if i % 10000 == 0 and i > 0:
                print('Processed {:,} documents. Total # tokens: {:,}.'.format(i, total_num_tokens))
    print('{}. # documents: {:,}. # tokens: {:,}.'.format(
        file_path, i, total_num_tokens))


def main(args):

    input_path = Path(args.input)
    output = Path(args.output)
    assert input_path.exists(), f'Error: {input_path} does not exist.'
    output.mkdir(exist_ok=True)

    mt = MosesTokenizer(args.lang)

    sml_wiki = output / f'{args.lang}-2'
    lrg_wiki = output / f'{args.lang}-100'
    sml_wiki.mkdir(exist_ok=True)
    lrg_wiki.mkdir(exist_ok=True)

    text_iter = get_texts(input_path)

    splits = ['train', 'valid', 'test']
    token_nums = [2000000, 200000, 200000]
    for split, token_num in zip(splits, token_nums):
        sml_file_path = sml_wiki / f'{args.lang}.wiki.{split}.tokens'
        write_wikitext(sml_file_path, text_iter, mt, token_num)
        lrg_file_path = lrg_wiki / f'{args.lang}.wiki.{split}.tokens'

        # copy the content of the small file to the large file
        print(f'Copying {sml_file_path} to {lrg_file_path}.')
        copyfile(sml_file_path, lrg_file_path)

    # add the new articles to the existing ones
    lrg_wiki_train = lrg_wiki / f'{args.lang}.wiki.train.tokens'
    write_wikitext(lrg_wiki_train, text_iter, mt, 98000000, mode='a')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='the directory where the Wikipedia data extracted '
                             'with WikiExtractor.py is located. Consists of '
                             'directories AA, AB, AC, etc.')
    parser.add_argument('-o', '--output', required=True,
                        help='the output directory where the merged Wikipedia '
                             'documents should be saved')
    parser.add_argument('-l', '--lang', required=True,
                        help='the iso code of the language of the Wikipedia '
                             'documents, e.g. en, fr, de, etc.')
    args = parser.parse_args()
    main(args)
