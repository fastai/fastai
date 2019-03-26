# usage: make help

# notes:
# 'target: | target1 target2' syntax enforces the exact order

.PHONY: bump bump-dev bump-major bump-major-dev bump-minor bump-minor-dev bump-post-release clean clean-build clean-build-conda clean-build-pypi clean-conda clean-pyc clean-pyc-conda clean-pyc-pypi clean-pypi clean-test clean-test-conda clean-test-pypi commit-release-push commit-hotfix-push commit-tag dist-conda dist-pypi dist-pypi-bdist dist-pypi-sdist docs sanity-check git-pull help release tag-version-push test test-cpu test-install-conda test-install test-install-pyp upload upload-conda upload-pypi install-conda-local git-clean-check sanity-check-hotfix release-hotfix

define get_cur_branch
$(shell git branch | sed -n '/\* /s///p')
endef

define echo_cur_branch
@echo Now on [$(call get_cur_branch)] branch
endef

version_file = fastai/version.py
version = $(shell python setup.py --version)
cur_branch = $(call get_cur_branch)

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

define WAIT_TILL_PIP_VER_IS_AVAILABLE_BASH =
# note that when:
# bash -c "command" arg1
# is called, the first argument is actually $0 and not $1 as it's inside bash!
#
# is_pip_ver_available "1.0.14"
# returns (echo's) 1 if yes, 0 otherwise
#
# since pip doesn't have a way to check whether a certain version is available,
# here we are using a hack, calling:
# pip install fastai==
# which doesn't find the unspecified version and returns all available
# versions instead, which is what we search
function is_pip_ver_available() {
    local ver="$$0"
    local out="$$(pip install fastai== |& grep $$ver)"
    if [[ -n "$$out" ]]; then
        echo 1
    else
        echo 0
    fi
}

function wait_till_pip_ver_is_available() {
    local ver="$$1"
    if [[ $$(is_pip_ver_available $$ver) == "1" ]]; then
        echo "fastai-$$ver is available on pypi"
        return 0
    fi

    COUNTER=0
    echo "waiting for fastai-$$ver package to become visible on pypi:"
    while [[ $$(is_pip_ver_available $$ver) != "1" ]]; do
        echo -en "\\rwaiting: $$COUNTER secs"
        COUNTER=$$[$$COUNTER +5]
	    sleep 5
    done
    sleep 5 # wait a bit longer if we hit a different cache on install
    echo -e "\rwaited: $$COUNTER secs    "
    echo -e "fastai-$$ver is now available on pypi"
}

echo "checking version $$0"
wait_till_pip_ver_is_available "$$0"
endef
export WAIT_TILL_PIP_VER_IS_AVAILABLE_BASH

.DEFAULT_GOAL := help

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

dist-pypi-bdist: ## build pypi wheel package
	@echo "\n\n*** Building pypi wheel package"
	python setup.py bdist_wheel

dist-pypi-sdist: ## build pypi source package
	@echo "\n\n*** Building pypi source package"
	python setup.py sdist

dist-pypi: | clean-pypi dist-pypi-sdist dist-pypi-bdist ## build pypi source and wheel package
	ls -l dist

upload-pypi: ## upload pypi package
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

dist-conda: | clean-conda dist-pypi-sdist ## build conda package
	@echo "\n\n*** Building conda package"
	mkdir "conda-dist"
	conda-build ./conda/ -c pytorch -c fastai/label/main --output-folder conda-dist
	ls -l conda-dist/noarch/*tar.bz2

upload-conda: ## upload conda package
	@echo "\n\n*** Uploading" conda-dist/noarch/*tar.bz2 "to fastai@anaconda.org\n"
	anaconda upload conda-dist/noarch/*tar.bz2 -u fastai

install-conda-local: ## install the locally built conda package
	@echo "\n\n*** Installing the local build of" conda-dist/noarch/*tar.bz2
	conda install -y -c ./conda-dist/ -c fastai fastai==$(version)

##@ Combined (pip and conda)

# currently, no longer needed as we now rely on sdist's tarball for conda source, which doesn't have any data in it already
# find ./data -type d -and -not -regex "^./data$$" -prune -exec rm -rf {} \;

clean: clean-pypi clean-conda ## clean pip && conda package

dist: clean dist-pypi dist-conda ## build pip && conda package

upload: upload-pypi upload-conda ## upload pip && conda package

install: clean ## install the package to the active python's site-packages
	python setup.py install

test: ## run tests with the default python
	python setup.py --quiet test

test-fast: ## run tests in parallel (requires pip install pytest-xdist)
	pytest -n 3

test-full: ## run all tests, including slow ones, print summary
	pytest --runslow -ra

test-cpu: ## run tests with the default python and CUDA_VISIBLE_DEVICES=""
	CUDA_VISIBLE_DEVICES="" python setup.py --quiet test

tools-update: ## install/update build tools
	@echo "\n\n*** Updating build tools"
	conda install -y conda-verify conda-build anaconda-client
	pip install -U twine

docs: ## update docs
	tools/build-docs -f

log_file := release-`date +"%Y-%m-%d-%H-%M-%S"`.log
release: ## do it all (other than testing)
	@echo "\n\n*** logging to $(log_file)"
	( \
	${MAKE} tools-update && \
	${MAKE} master-branch-switch && \
	${MAKE} sanity-check && \
	${MAKE} test && \
	${MAKE} bump && \
	${MAKE} changes-finalize && \
	${MAKE} release-branch-create && \
	${MAKE} commit-version && \
	${MAKE} master-branch-switch && \
	${MAKE} bump-dev && \
	${MAKE} changes-dev-cycle && \
	${MAKE} commit-dev-cycle-push && \
	${MAKE} prev-branch-switch && \
	${MAKE} commit-release-push && \
	${MAKE} tag-version-push && \
	${MAKE} dist && \
	${MAKE} upload && \
	${MAKE} test-install && \
	${MAKE} backport-check && \
	${MAKE} master-branch-switch && \
	echo "Done" \
	) 2>&1 | tee $(log_file)

log_file_hotfix := release-hotfix-`date +"%Y-%m-%d-%H-%M-%S"`.log
release-hotfix: ## do most of the hotfix release process
	@echo "\n\n*** logging to $(log_file)"
	( \
	${MAKE} sanity-check-hotfix && \
	${MAKE} test && \
	${MAKE} bump-post-release && \
	${MAKE} commit-hotfix-push && \
	${MAKE} tag-version-push && \
	${MAKE} dist && \
	${MAKE} upload && \
	${MAKE} test-install && \
	${MAKE} master-branch-switch && \
	echo "Done" \
	) 2>&1 | tee $(log_file_hotfix)

##@ git helpers

git-pull: ## git pull
	@echo "\n\n*** Making sure we have the latest checkout"
	git checkout master
	git pull
	git status

git-clean-check:
	@echo "\n\n*** Checking that everything is committed"
	@if [ -n "$(shell git status -s)" ]; then\
		echo "git status is not clean. You have uncommitted git files";\
		exit 1;\
	else\
		echo "git status is clean";\
    fi

git-check-remote-origin-url:
	@echo "\n\n*** Checking `git config --get remote.origin.url`"
	@perl -le '$$_=shift; $$u=q[git@github.com:fastai/fastai.git]; $$_ eq $$u ? print "Correct $$_" : die "Expecting $$u, got $$_"' $(shell git config --get remote.origin.url)

sanity-check: git-clean-check git-check-remote-origin-url
	@echo "\n\n*** Checking master branch version: should always be: X.Y.Z.dev0"
	@perl -le '$$_=shift; $$v="initial version: $$_"; /\.dev0$$/ ? print "Good $$v" : die "Bad $$v, expecting .dev0"' $(version)

sanity-check-hotfix: git-clean-check git-check-remote-origin-url
	@echo "\n\n*** Checking branch name: expecting release-X.Y.Z"
	@perl -le '$$_=shift; $$br="current branch: $$_"; /^release-\d+\.\d+\.\d+/ ? print "Good $$br" : die "Bad $$br, expecting release-X.Y.Z"' $(cur_branch)

prev-branch-switch:
	@echo "\n\n*** [$(cur_branch)] Switching to prev branch"
	git checkout -
	$(call echo_cur_branch)

# also do a special sanity check for broken git setups that switch to private fork on branch
release-branch-create:
	@echo "\n\n*** [$(cur_branch)] Creating release-$(version) branch"
	git checkout -b release-$(version)
	$(call echo_cur_branch)
	$(MAKE) git-check-remote-origin-url

release-branch-switch:
	@echo "\n\n*** [$(cur_branch)] Switching to release-$(version) branch"
	git checkout release-$(version)
	$(call echo_cur_branch)

master-branch-switch:
	@echo "\n\n*** [$(cur_branch)] Switching to master branch: version $(version)"
	git checkout master
	$(call echo_cur_branch)

commit-version: ## commit and tag the release
	@echo "\n\n*** [$(cur_branch)] Start release branch: $(version)"
	git commit -m "starting release branch: $(version)" $(version_file)
	$(call echo_cur_branch)

# in case someone managed to push something into master since this process
# started, it's now safe to git pull (which would avoid the merge error and
# break 'make release'), as we are no longer on the release branch and new
# pulled changes won't affect the release branch
commit-dev-cycle-push: ## commit version and CHANGES and push
	@echo "\n\n*** [$(cur_branch)] pull before commit to avoid interactive merges"
	git pull

	@echo "\n\n*** [$(cur_branch)] Start new dev cycle: $(version)"
	git commit -m "new dev cycle: $(version)" $(version_file) CHANGES.md

	@echo "\n\n*** [$(cur_branch)] Push changes"
	git push

commit-release-push: ## commit CHANGES.md, push/set upstream
	@echo "\n\n*** [$(cur_branch)] Commit CHANGES.md"
	git commit -m "version $(version) release" CHANGES.md || echo "no changes to commit"

	@echo "\n\n*** [$(cur_branch)] Push changes"
	git push --set-upstream origin release-$(version)

commit-hotfix-push: ## commit version and CHANGES and push
	@echo "\n\n*** [$(cur_branch)] Complete hotfix: $(version)"
	git commit -m "hotfix: $(version)" $(version_file) CHANGES.md

	@echo "\n\n*** [$(cur_branch)] Push changes"
	git push

tag-version-push: ## tag the release
	@echo "\n\n*** [$(cur_branch)] Tag $(version) version"
	git tag -a $(version) -m "$(version)" && git push origin tag $(version)

# check whether there any commits besides fastai/version.py and CHANGES.md
# from the point of branching of release-$(version) till its HEAD. If
# there are any, then most likely there are things to backport.
backport-check: ## backport to master check
	@echo "\n\n*** [$(cur_branch)] Checking if anything needs to be backported"
	$(eval start_rev := $(shell git rev-parse --short $$(git merge-base master origin/release-$(version))))
	@if [ ! -n "$(start_rev)" ]; then\
		echo "*** failed, check you're on the correct release branch";\
		exit 1;\
	fi
	$(eval log := $(shell git log --oneline $(start_rev)..origin/release-$(version) -- . ":(exclude)fastai/version.py" ":(exclude)CHANGES.md"))
	@if [ -n "$(log)" ]; then\
		echo "!!! These commits may need to be backported:\n\n$(log)\n\nuse 'git show <commit>' to review or go to https://github.com/fastai/fastai/compare/release-$(version) to do it visually\nFor backporting see: https://docs-dev.fast.ai/release#backporting-release-branch-to-master";\
	else\
		echo "Nothing to backport";\
    fi


##@ Testing new package installation

test-install: ## test installing this version of the conda/pip packages
	@echo "\n\n*** [$(cur_branch)] Branch check (needing release branch)"
	@if [ "$(cur_branch)" = "master" ]; then\
		echo "Error: you are not on the release branch, to switch to it do:\n  git checkout release-1.0.??\nafter adjusting the version number. Also possible that:\n  git checkout - \nwill do the trick, if you just switched from it. And then repeat:\n  make test-install\n";\
		exit 1;\
	else\
		echo "You're on the release branch, good";\
	fi

	${MAKE} test-install-pypi
	${MAKE} test-install-conda

	@echo "\n\n*** Install the editable version to return to dev work"
	pip install -e .[dev]

test-install-pypi: ## test installing this version of the pip package
	@echo "\n\n*** Install/uninstall $(version) pip version"
	@pip uninstall -y fastai

	@echo "\n\n*** waiting for $(version) pip version to become visible"
	bash -c "$$WAIT_TILL_PIP_VER_IS_AVAILABLE_BASH" $(version)

	pip install fastai==$(version)
	pip uninstall -y fastai

test-install-conda: ## test installing this version of the conda package
	@echo "\n\n*** Install/uninstall $(version) conda version"
	@# skip, throws error when uninstalled @conda uninstall -y fastai

	@echo "\n\n*** waiting for $(version) conda version to become visible"
	@perl -e '$$v=shift; $$p="fastai"; $$|++; sub ok {`conda search -c fastai $$p==$$v >/dev/null 2>&1`; return $$? ? 0 : 1}; print "waiting for $$p-$$v to become available on conda\n"; $$c=0; while (not ok()) { print "\rwaiting: $$c secs"; $$c+=5;sleep 5; }; sleep 5; print "\n$$p-$$v is now available on conda\n"' $(version)

	conda install -y -c fastai fastai==$(version)
	conda uninstall -y fastai

##@ CHANGES.md file targets

changes-finalize: ## fix the version and stamp the date
	@echo "\n\n*** [$(cur_branch)] Adjust '## version (date)' in CHANGES.md"
	perl -pi -e 'use POSIX qw(strftime); BEGIN{$$date=strftime "%Y-%m-%d", localtime};s|^##.*Work In Progress\)|## $(version) ($$date)|' CHANGES.md

changes-dev-cycle: ## insert new template + version
	@echo "\n\n*** [$(cur_branch)] Install new template + version in CHANGES.md"
	perl -0777 -pi -e 's|^(##)|## $(version) (Work In Progress)\n\n### New:\n\n### Changed:\n\n### Fixed:\n\n\n\n$$1|ms' CHANGES.md


##@ Version bumping

# Support semver, but using python's .dev0/.post0 instead of -dev0/-post0

bump-major: ## bump major level; remove .devX if any
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=join(".", $$2+1, 0, 0); print STDERR "\n\n*** [$(cur_branch)] Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump-minor: ## bump minor level; remove .devX if any
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=join(".", $$2, $$3+1, 0); print STDERR "\n\n*** [$(cur_branch)] Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump-patch: ## bump patch level unless has .devX, then don't bump, but remove .devX
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=$$5 ? join(".", $$2, $$3, $$4) :join(".", $$2, $$3, $$4+1); print STDERR "\n\n*** [$(cur_branch)] Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump: bump-patch ## alias to bump-patch (as it's used often)

bump-post-release: ## add .post1 or bump post-release level .post2, .post3, ...
	@perl -pi -e 's{((\d+\.\d+\.\d+)(\.\w+\d+)?)}{do { $$o=$$1; $$b=$$2; $$l=$$3||".post0"}; $$l=~s/(\d+)$$/$$1+1/e; $$n="$$b$$l"; print STDERR "\n\n*** [$(cur_branch)] Changing version: $$o => $$n\n"; $$n}e' $(version_file)

bump-major-dev: ## bump major level and add .dev0
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=join(".", $$2+1, 0, 0, "dev0"); print STDERR "\n\n*** [$(cur_branch)] Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump-minor-dev: ## bump minor level and add .dev0
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=join(".", $$2, $$3+1, 0, "dev0"); print STDERR "\n\n*** [$(cur_branch)] Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump-patch-dev: ## bump patch level and add .dev0
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=join(".", $$2, $$3, $$4+1, "dev0"); print STDERR "\n\n*** [$(cur_branch)] Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump-dev: bump-patch-dev ## alias to bump-patch-dev (as it's used often)



##@ Coverage

coverage: ## check code coverage quickly with the default python
	coverage run --source fastai -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html
