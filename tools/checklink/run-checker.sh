#!/bin/bash

# this is a helper script for the CI build

# usage:
# ./run-checker.sh docs
# ./run-checker.sh docs-dev
me=`basename "$0"`
if [[ ! $1 ]];
then echo "usage: $me <docset> (docset: either 'docs' or 'docs-dev')"
     exit 1;
fi

log="checklink-$1.log"

if [[ ! -f $log ]];
then
    echo "Program didn't run";
    exit 1;
else
    if [[ -s $log ]];
    then echo "Got broken links:";
         cat $log;
         exit 1;
    else echo "No broken links were found";
    fi
fi
