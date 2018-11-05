#!/bin/bash

# run a doc link checker on fast.ai docs sites.

# usage:
# ./fastai-checklink-run.sh docs
# ./fastai-checklink-run.sh docs-dev
me=`basename "$0"`
if [[ ! $1 ]];
then echo "usage: $me <docset> (docset: either 'docs' or 'docs-dev')"
     exit 1;
fi

set="$1"
log="checklink-$set.log"

# throttle to 2-sec per request
sleep_secs=2

echo "Checking ${set}.fast.ai"
echo "Logging into $log"
echo "This will take a few minutes. The process will be silent unless problems are encountered"
./fastai-checklink --depth 50 --quiet --broken -e --sleep "$sleep_secs" --timeout 60 --connection-cache 3 --exclude "github.com|test.pypi.org|ycombinator.com|anaconda.org|google.com|microsoft.com" --cookies cookies.txt "https://$set.fast.ai" | tee "$log"

# the script will give no output if all is good, so let's give a clear indication of success
if [[ ! -s $log ]]; then echo "No broken links were found"; fi
