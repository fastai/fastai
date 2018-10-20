# usage: make help

.PHONY: clean clean-test clean-pyc clean-build docs help clean-pypi clean-build-pypi clean-pyc-pypi clean-test-pypi dist-pypi release-pypi clean-conda clean-build-conda clean-pyc-conda clean-test-conda test tag bump bump-minor bump-major bump-dev bump-minor-dev bump-major-dev commit-tag git-pull git-not-dirty test-install

version_file = fastai/version.py
version = $(shell python setup.py --version)

.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help: ## this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ PyPI

clean-pypi: clean-build-pypi clean-pyc-pypi clean-test-pypi ## remove all build, test, coverage and python artifacts

clean-build-pypi: ## remove pypi build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc-pypi: ## remove python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test-pypi: ## remove pypi test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

dist-pypi: clean-pypi ## build pypi source and wheel package
	@echo "\n\n*** Building pypi packages"
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

release-pypi: ## release pypi package
	@echo "\n\n*** Uploading" dist/* "to pypi\n"
	twine upload dist/*


##@ Conda

clean-conda: clean-build-conda clean-pyc-conda clean-test-conda ## remove all build, test, coverage and python artifacts

clean-build-conda: ## remove conda build artifacts
	@echo "\n\n*** conda build purge"
	conda build purge-all
	@echo "\n\n*** rm -fr conda-dist/"
	rm -fr conda-dist/

clean-pyc-conda: ## remove conda python file artifacts

clean-test-conda: ## remove conda test and coverage artifacts

dist-conda: clean-conda ## build conda package
	@echo "\n\n*** Building conda package"
	mkdir "conda-dist"
	conda-build ./conda/ -c pytorch -c fastai/label/main --output-folder conda-dist
	ls -l conda-dist/noarch/*tar.bz2

release-conda: ## release conda package
	@echo "\n\n*** Uploading" conda-dist/noarch/*tar.bz2 "to fastai@anaconda.org\n"
	anaconda upload conda-dist/noarch/*tar.bz2 -u fastai



##@ Combined (pip and conda)

# XXX: until conda-build supports skipping dirs, we have to reduce the size of the
# checked out directory, otherwise it make take forever to copy the folder
clean: clean-pypi clean-conda ## clean pip && conda package
	find ./data -type d -and -not -regex "^./data$$" -prune -exec rm -rf {} \;

dist: clean dist-pypi dist-conda ## build pip && conda package

release: release-pypi release-conda ## release pip && conda package

install: clean ## install the package to the active python's site-packages
	python setup.py install

test: ## run tests with the default python
	python setup.py --quiet test

tools-update: ## install/update build tools
	@echo "\n\n*** Updating build tools"
	conda install -y conda-verify conda-build anaconda-client
	pip install -U twine


##@ git helpers

git-pull: ## git pull
	@echo "\n\n*** Making sure we have the latest checkout"
	git pull
	git status

git-not-dirty:
	@echo "*** Checking that everything is committed"
	@if [ -n "$(git status -s)" ]; then\
		echo "uncommitted git files";\
		false;\
    fi

prev-branch-switch:
	@echo "*** Switching to prev branch"
	git checkout -

release-branch-create:
	@echo "*** Creating branch release-$(version)"
	git checkout -b release-$(version)

release-branch-switch:
	@echo "*** Switching to branch release-$(version)"
	git checkout release-$(version)

master-branch-switch:
	@echo "*** Switching to master branch: version $(version)"
	git checkout master

commit-dev-cycle-push: ## commit version and CHANGES and push
	@echo "\n\n*** Start new dev cycle: $(version)"
	git commit -m "new dev cycle: $(version)" $(version_file) CHANGES.md

	@echo "\n\n*** Push all changes"
	git push

commit-version: ## commit and tag the release
	@echo "\n\n*** Start release branch: $(version)"
	git commit -m "starting release branch: $(version)" $(version_file)

commit-tag-push: ## commit and tag the release
	@echo "\n\n*** Commit CHANGES"
	git commit -m "version $(version) release" CHANGES.md || echo "no changes to commit"

	@echo "\n\n*** Tag $(version) version"
	git tag -a $(version) -m "$(version)" && git push --tags

	@echo "\n\n*** Push all changes"
	git push --set-upstream origin release-$(version)

# check whether there any commits besides fastai/version.py from the
# point of branching of release-$(version) till its HEAD. If there are
# then probably there are things to backport.
backport-check: ## backport to master check
	@echo "*** Checking if anything needs to be backported"
	$(eval start_rev := $(shell git rev-parse --short $$(git merge-base --fork-point master origin/release-$(version))))
	@echo "*** branching point: $(start_rev)"
	$(eval log := $(shell git log --oneline $(start_rev)..origin/release-$(version) -- . ":(exclude)fastai/version.py"))
	@if [ -n "$(log)" ]; then\
		echo "!!! These commits may need to be backported:\n\n$(log)\n\nuse 'git show <commit>' to review or go to https://github.com/fastai/fastai/compare/release-$(version) to do it visually";\
	else\
		echo "Nothing to backport";\
    fi


##@ Testing new package installation

test-install: ## test conda/pip package by installing that version them
	@echo "\n\n*** Install/uninstall $(version) pip version"
	@pip uninstall -y fastai
	pip install fastai==$(version)
	pip uninstall -y fastai

	@echo "\n\n*** Install/uninstall $(version) conda version"
	@# skip, throws error when uninstalled @conda uninstall -y fastai
	conda install -y -c fastai fastai==$(version)
	@# leave conda package installed: conda uninstall -y fastai


##@ Version bumping

# Support semver, but using python's .dev0 instead of -dev0

bump-patch: ## bump patch-level unless has .devX, then don't bump, but remove .devX
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=$$5 ? join(".", $$2, $$3, $$4) :join(".", $$2, $$3, $$4+1); print STDERR "*** Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump: bump-patch ## alias to bump-patch (as it's used often)

bump-minor: ## bump minor-level unless has .devX, then don't bump, but remove .devX
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=$$5 ? join(".", $$2, $$3, $$4) :join(".", $$2, $$3+1, $$4); print STDERR "*** Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump-major: ## bump major-level unless has .devX, then don't bump, but remove .devX
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=$$5 ? join(".", $$2, $$3, $$4) :join(".", $$2+1, $$3, $$4); print STDERR "*** Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump-patch-dev: ## bump patch-level and add .dev0
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=join(".", $$2, $$3, $$4+1, "dev0"); print STDERR "*** Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump-dev: bump-patch-dev ## alias to bump-patch-dev (as it's used often)

bump-minor-dev: ## bump minor-level and add .dev0
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=join(".", $$2, $$3+1, $$4, "dev0"); print STDERR "*** Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump-major-dev: ## bump major-level and add .dev0
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=join(".", $$2+1, $$3, $$4, "dev0"); print STDERR "*** Changing version: $$o => $$n\n"; $$n |e' $(version_file)


##@ Coverage

coverage: ## check code coverage quickly with the default python
	coverage run --source fastai -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html
