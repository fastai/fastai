.ONESHELL:
SHELL := /bin/bash

SRC = $(wildcard nbs/*.ipynb)

all: fastai 

both: fastai docs

help:
	cat Makefile

fastai: $(SRC)
	nbdev_clean_nbs
	nbdev_build_lib

update_lib:
	pip install nbdev --upgrade

docs_serve: docs
	cd docs && bundle exec jekyll serve

docs: $(SRC)
	rsync -a docs_src/ docs
	nbdev_build_docs

test:
	nbdev_test_nbs --pause 0.5 --flags ''

testmore:
	nbdev_test_nbs --pause 0.5 --flags 'cpp cuda' --n_workers 8

testall:
	nbdev_test_nbs --pause 0.5 --flags 'cpp cuda slow' --n_workers 4

release: pypi
	sleep 3
	fastrelease_conda_package --mambabuild --upload_user fastai
	fastrelease_bump_version
	nbdev_build_lib | tail

conda_release:
	fastrelease_conda_package --mambabuild --upload_user fastai

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist

