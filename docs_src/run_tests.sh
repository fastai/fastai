#!/bin/bash

# make sure spacy's en default model is installed before tests are run
python -c 'import spacy; spacy.load("en")' >/dev/null 2>&1 || python -m spacy download en

python -m pytest -p nbval "$@"
