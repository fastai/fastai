#!/usr/bin/env bash
set -e

if [[ $# -ne 1 ]] ; then
  echo 'Please pass a language idenfier (e.g. "en")'
  exit 1
fi

LANG=$1
WIKI_DIR="${LANG}wiki"
mkdir -p $WIKI_DIR
cd $WIKI_DIR

BASE_FILE="${LANG}wiki-latest-pages-articles.xml"
DUMP_FILE="${BASE_FILE}.bz2"
if [ ! -f $DUMP_FILE ]; then wget -c "https://dumps.wikimedia.org/${LANG}wiki/latest/${DUMP_FILE}"; fi
if [ ! -f $BASE_FILE ]; then bunzip2 $DUMP_FILE; fi
if [ ! -d wikiextractor ]; then git clone https://github.com/attardi/wikiextractor.git; fi

python wikiextractor/WikiExtractor.py --processes 4 --no_templates --min_text_length 1800 \
  --filter_disambig_pages --log_file log -b 100G -q $BASE_FILE

mv text/AA/wiki_00 $WIKI_DIR
rm -rf text
echo "Saving data in $WIKI_DIR"
