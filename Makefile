.ONESHELL:
SHELL := /bin/bash

SRC = $(wildcard nbs/*.ipynb)

all: fastai docs

help:
	cat Makefile

fastai: $(SRC)
	nbdev_clean_nbs
	nbdev_build_lib
	touch fastai

update_lib:
	pip install nbdev --upgrade

docs_serve: docs
	cd docs && bundle exec jekyll serve

docs: $(SRC)
	nbdev_build_docs
	cd docs
	git commit -am docs && git push
	cd -
	touch docs

test:
	nbdev_test_nbs

release: pypi
	nbdev_conda_package --upload_user fastai --build_args '-c pytorch -c fastai'
	nbdev_bump_version

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist

