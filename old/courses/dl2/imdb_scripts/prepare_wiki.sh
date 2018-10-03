#!/usr/bin/env bash
# Script to download a Wikipedia dump

# Script is partially based on https://github.com/facebookresearch/fastText/blob/master/get-wikimedia.sh
ROOT="data"
DUMP_DIR="${ROOT}/wiki_dumps"
EXTR_DIR="${ROOT}/wiki_extr"
WIKI_DIR="${ROOT}/wiki"
EXTR="wikiextractor"
mkdir -p "${ROOT}"
mkdir -p "${DUMP_DIR}"
mkdir -p "${EXTR_DIR}"
mkdir -p "${WIKI_DIR}"

echo "Saving data in ""$ROOT"
read -r -p "Choose a language (e.g. en, bh, fr, etc.): " choice
LANG="$choice"
echo "Chosen language: ""$LANG"
DUMP_FILE="${LANG}wiki-latest-pages-articles.xml.bz2"
DUMP_PATH="${DUMP_DIR}/${DUMP_FILE}"

if [ ! -f "${DUMP_PATH}" ]; then
  read -r -p "Continue to download (WARNING: This might be big and can take a long time!) (y/n)? " choice
  case "$choice" in
    y|Y ) echo "Starting download...";;
    n|N ) echo "Exiting";exit 1;;
    * ) echo "Invalid answer";exit 1;;
  esac
  wget -c "https://dumps.wikimedia.org/""${LANG}""wiki/latest/""${DUMP_FILE}""" -P "${DUMP_DIR}"
else
  echo "${DUMP_PATH} already exists. Skipping download."
fi

# Check if directory exists
if [ ! -d "${EXTR}" ]; then
  git clone https://github.com/attardi/wikiextractor.git
  cd "${EXTR}"
  python setup.py install
fi

EXTR_PATH="${EXTR_DIR}/${LANG}"
if [ ! -d "${EXTR_PATH}" ]; then
  read -r -p "Continue to extract Wikipedia (WARNING: This might take a long time!) (y/n)? " choice
  case "$choice" in
    y|Y ) echo "Extracting ${DUMP_PATH} to ${EXTR_PATH}...";;
    n|N ) echo "Exiting";exit 1;;
    * ) echo "Invalid answer";exit 1;;
  esac
  python wikiextractor/WikiExtractor.py -s --json -o "${EXTR_PATH}" "${DUMP_PATH}"
else
  echo "${EXTR_PATH} already exists. Skipping extraction."
fi

OUT_PATH="${WIKI_DIR}/${LANG}"
read -r -p "Continue to merge Wikipedia articles (y/n)? " choice
case "$choice" in
y|Y ) echo "Merging articles from ${EXTR_PATH} to ${OUT_PATH}...";;
n|N ) echo "Exiting";exit 1;;
* ) echo "Invalid answer";exit 1;;
esac
python merge_wiki.py -i "${EXTR_PATH}" -o "${OUT_PATH}"
