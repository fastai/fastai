# fastai Link Checker

This is https://github.com/w3c/link-checker with some custom tweaks.

## Prerequisites

* Debian/Ubuntu:

   ```
   sudo apt install w3c-linkchecker
   ```
   
* Mac OS:

   ```bash
   brew install perl # Make sure that we have the most updated perl version installed
   perl -MCPAN -e 'install W3C::LinkChecker' # Install the link checker
   ```

See "More on Prerequisites" below for other ways to install this prerequisite if the above doesn't work.

To run this link checker again a local checkout, you also need to install `jekyll`:

1. Install `ruby-bundler`

   ```
   sudo apt install ruby-bundler
   ```
2. Install jekyll

   When running this one it will ask for your user's password (basically running a sudo operation).

   ```
   bundle install jekyll
   ```

## Checking the site locally

This currently only works for `docs.fast.ai`. For the other `fast.ai` sites use the slower online checking tool.

This is an order of magnitude faster check, since the tool doesn't need to throttle itself not to get blocked by the webserver. The script creates the complete website under `docs/_site/` and the linkchecker checks against that.

Usage:
```
cd tools/checklink
./checklink-docs-local.sh
```

## Checking the online sites

Note, that if you have just committed changes to git, wait a few minutes for github pages to sync, otherwise you'll be testing an outdated live site.

Check `(docs*|course-v3).fast.ai` for broken links and anchors:

```
cd tools/checklink
./checklink-docs.sh
./checklink-course-v3.sh
```

Each file logs to console and also into `checklink-docs.log` and `checklink-course-v3.log`

If you're on windows w/o bash and assuming you have [perl installed](https://learn.perl.org/installing/windows.html), you can run it directly like:

```
perl fastai-checklink --depth 50 --quiet --broken -e --sleep 2 --timeout 60 --connection-cache 3 --exclude github.com --exclude test.pypi.org --exclude ycombinator.com --exclude anaconda.org --exclude google.com --cookies cookies.txt "https://docs.fast.ai"
```

The script is set to sleep for 2 secs between each request, so not to get blocked by github, so it takes some 5-10min to complete.

You can add `--html` inside those scripts if you prefer to have the html output (in which case change the scripts to `|tee checklink-docs-log.html` or similar, since it dumps the output to stdout.




## More on Prerequisites

If for any reason you don't have the apt packages for `w3c-linkchecker`, you can install those manually with:

```
sudo apt install cpanminus
sudo cpanm W3C::LinkChecker
```

or via CPAN shell:

```
sudo apt install cpanplus
perl -MCPAN -e shell
install install W3C::LinkChecker
```

OSX Install:
```
sudo cpan install CPAN
sudo cpan Mozilla::CA
sudo perl -MCPAN -e 'install W3C::LinkChecker'
```



## Portable sript

You can ignore the rest of this document, unless you'd like to build a perl executable with all the prerequisites built in. It's not part of the repo because of its significant size.

## Building the executable

If for any reason you need to create an executable (or make one for another platform), install the script's dependencies (previous step), the build tools and then make the executable.

## Build tools prerequisites

Install Perl PAR Packager:

```
sudo apt install libpar-packer-perl
```

or via cpanm:

```
cpanm pp
```

or via CPAN shell:

```
perl -MCPAN -e shell
install pp
```

## Build the portable version of the tool

This will build a portable executable version for your platform (it's portable in a sense that it doesn't need any of its many dependencies). e.g. for linux:

```
cd tools/checklink
pp -o checklink-linux checklink
```
