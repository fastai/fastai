SRC = $(wildcard nbs/*.ipynb)

all: fastai2 docs test

fastai2: $(SRC)
	nbdev_clean_nbs
	nbdev_build_lib
	touch fastai2

update_lib:
	pip install nbdev --upgrade

docs_serve: docs
	cd docs && bundle exec jekyll serve

docs: $(SRC)
	nbdev_build_docs
	touch docs

test:
	nbdev_test_nbs

release: pypi
	nbdev_bump_version

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist

