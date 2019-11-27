SRC = $(wildcard nbs/*.ipynb)

all: fastai2 docs

fastai2: $(SRC)
	nbdev_build_lib
	touch fastai2

docs: $(SRC)
	nbdev_build_docs
	touch docs

test:
	nbdev_test_nbs

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist