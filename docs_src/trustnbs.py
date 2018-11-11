#!/usr/bin/env python

import os, glob, nbformat.sign

# Iterate over notebooks and sign each of them as trusted
for fname in glob.glob("*.ipynb"):
    with open(fname) as f:
        nb = nbformat.read(f, as_version=4)
        nbformat.sign.NotebookNotary().sign(nb)

