#!/bin/bash

# check links and anchors of docs.fast.ai, but locally as fastai/docs/ dir

# make sure to run tools/build-docs

# usage:
# ./checklink-docs-local.sh

cur=`pwd`
base=`dirname "$cur"`
log="checklink-docs-local.log"
site="docs/_site/"

echo -e "\n\n*** Updating docs/_site/"
cd ../docs
bundle exec jekyll build
cd -

echo -e "\n\n\n*** Checking docs.fast.ai against local fs at $base/$site"
echo "Logging into $log"
echo "This will take a few minutes. The process will be silent unless problems are encountered"

# have to hack the urls using a modified masquerade feature to support full regex in the argument.
./fastai-checklink --depth 50 --quiet --broken -e --timeout 60 --connection-cache 3 --exclude "github.com|test.pypi.org|ycombinator.com|anaconda.org|google.com|microsoft.com" --masquerade "file:///(?=[^/]+.html) file://$base/$site" ../$site | tee "$log"

# the script will give no output if all is good, so let's give a clear indication of success
if [[ ! -s $log ]]; then echo "No broken links were found"; fi
