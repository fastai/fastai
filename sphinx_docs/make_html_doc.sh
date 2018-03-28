#!/bin/bash

#bash script to create html documentation under build/html folder
phinx-apidoc -e -f -o source/ ../fastai
make html
