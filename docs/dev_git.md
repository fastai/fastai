---
title: git Notes
---

Chances are that you may need to know some git when using fastai - for example if you want to contribute to the project, or you want to undo some change in your code tree. This document has a variety of useful recipes that might be of help in your work.




## How to Make a Pull Request (PR)

While this guide is mostly suitable for creating PRs for any github project, it includes several steps specific to the `fastai` project repositories, which currently are:

* https://github.com/fastai/fastai
* https://github.com/fastai/course-v3
* https://github.com/fastai/fastprogress

If you already know how to make PRs, you only need to read: the "Step 3" and "Step 5" sections, since they are unique requirements for the fastai project.

The following instructions use `USERNAME` as a github username placeholder. The easiest way to follow this guide is to copy-n-paste the whole section into a file, replace `USERNAME` with your real username and then follow the steps.

The guide is written for those who want to contribute to the `fastai` repository.
If you'd like to contribute to other `fastai`-project repositories, just replace `fastai` with that other repository name in the instructions below.

For the purpose of these examples, we will clone into a folder `fastai-fork`, to differentiate from `fastai` which you most likely already checked out to install it.

Also don't get confused between the `fastai` github username, the `fastai` repository, and the `fastai` module directory, where the python code resides. The following url shows all three, in the order they have been mentioned:

```
https://github.com/fastai/fastai/tree/master/fastai
                     |       |                  |
                 username reponame        modulename
```

Below you will find detailed 5 steps towards creating a PR.

### Helper Program

There is a smart [program](https://github.com/fastai/fastai/blob/master/tools/fastai-make-pr-branch) that can do all the heavy lifting of the first 2 steps for you. Then you just need to do your work, commit changes and submit PR. To run it:

```
curl -O https://raw.githubusercontent.com/fastai/fastai/master/tools/fastai-make-pr-branch
chmod a+x fastai-make-pr-branch
./fastai-make-pr-branch https your-github-username fastai new-feature
```

For more details run:
```
./fastai-make-pr-branch
```

While this is new and experimental, you probably want to place that script somewhere in your `$PATH`, so that you could invoke it from anywhere. Once it is well tested, it'll probably get installed automatically with the `fastai` package.


### Step 1. Start With a Synced Fork Checkout

#### 1a. First time

If you made the fork of the desired repository already, proceed to section 1b.

If it's your first time, you just need to make a fork of the original repository:

1. Go to https://github.com/fastai/fastai and in the right upper corner click on `[Fork]`. This will generate a fork of this repository, and you will be redirected to
 github.com/USERNAME/fastai.

2. Clone the main repository fork. Click on `[Clone or download]` button to get the clone url and then clone your repository.

   * Choose the SSH option if you have SSH configured with github and run:

   ```
   git clone git@github.com:USERNAME/fastai.git fastai-fork
   ```
   * otherwise choose the HTTPS option:

   ```
   git clone https://github.com/USERNAME/fastai.git fastai-fork
   ```

   Make sure the url has your username in it. If the username is `fastai` you're cloning the original repo and not your fork. This will not do what you need.

   Then move into the newly created directory:

   ```
   cd fastai-fork
   ```

   and run the setup tool:
   ```
   tools/run-after-git-clone
   ```
   for any of the `fastai` project repositories, except `fastprogress` where it doesn't exist.

   Finally, let's setup this fork to track the upstream:

   ```
   git remote add upstream git@github.com:fastai/fastai.git
   ```

   You can check your setup:
   ```
   git remote -v
   ```

   It should show:
   ```
   origin  git@github.com:USERNAME/fastai.git (fetch)
   origin  git@github.com:USERNAME/fastai.git (push)
   upstream        git@github.com:fastai/fastai.git (fetch)
   upstream        git@github.com:fastai/fastai.git (push)
   ```

   You can now proceed to step 2.

#### 1b. Subsequent times

If you make a PR right after you made a fork of the original repository, the two repositories are aligned and you can easily create a PR. If time passes the original repository starts diverging from your fork, so when you work on your PRs you need to keep your master fork in sync with the original repository.

You can tell the state of your fork, by going to https://github.com/USERNAME/fastai and seeing something like:

```
This branch is 331 commits behind fastai:master.
```

So, let's synchronize the two:

1. Place yourself in the `master` branch of the forked repository:

   * Either you go back to a repository you checked out earlier and switch to the `master` branch:

   ```
   cd fastai-fork
   git checkout master
   ```

   * or you make a new clone

   ```
   git clone git://github.com/USERNAME/fastai.git fastai-fork
   cd fastai-fork
   git remote add upstream git@github.com:fastai/fastai.git
   ```

   and set things up as before (except for the `fastprogress` repository):
   ```
   tools/run-after-git-clone
   ```

   Use the https version https://github.com/USERNAME/fastai if you don't have ssh configured with github.

2. Sync the forked repository with the original repository:

   ```
   git fetch upstream
   git checkout master
   git merge --no-edit upstream/master
   git push
   ```

   Now you can branch off this synced `master` branch.

   Validate that your fork is in sync with the original repository by going to https://github.com/USERNAME/fastai and checking that it says:

   ```
   This branch is even with fastai:master.
   ```
   Now you can work on a new PR.


### Step 2. Create a Branch

It's very important that you **always work inside a branch**. If you make any commits into the `master` branch, you will not be able to make more than one PR at the same time, and you will not be able to synchronize your forked `master` branch with the original without doing a reset. If you made a mistake and committed to the `master` branch, it's not the end of the world, it's just that you made your life more complicated. This guide will explain how to deal with this situation.


1. Create a branch with any name you want, for example `new-feature-branch`, and switch to it. Then set this branch's upstream, so that you could do `git push` and other git commands without needing to pass any more arguments.

   ```
   git checkout -b new-feature-branch
   git push --set-upstream origin new-feature-branch
   ```

### Step 3. Prepare Your Checkout

1. Install the prerequisites.

   No matter which repository you contribute to, unless you have already done so install the developer prerequisites:

   Use an existing checkout, or:
   ```
   git clone https://github.com/fastai/fastai
   cd fastai
   ```
   and install the dev prerequisites:
   ```
   pip install -e .[dev]
   ```

2. Now configure the nbstripout filters if you haven't yet done so (the helper script does it automatically for you if you have used it to create the PR branch).

   Move into the root of the repository where your PR branch is and run:

   ```
   tools/run-after-git-clone
   ```

### Step 4. Write Your Code

This is where the magic happens.

Create new code, fix bugs, add/correct documentation.


### Step 5. Test Your Changes

Test that your changes don't break things. Choose one according to which project you are creating PR for:

* `fastai`

   In the `fastai` repository, if you made changes to the libraries under `fastai` or you added/changed anything under `tests`, move into the root of the repository and run:

   ```
   make test
   ```
   or if you don't have `make`, just:
   ```
   pytest
   ```

* `docs_src`

   In the `docs_src` folder, if you made changes to the notebooks, run:

   ```
   cd docs_src
   run_tests.sh
   ```
   You will need at least 8GB free GPU RAM to run these tests.

### Step 6. Push Your Changes

1. When you're happy with the results, commit the new code:

   ```
   git commit -a
   ```

   `-a` will automatically commit changes to any of the repository files.

   If you created new files, first tell git to track them:

   ```
   git add newfile1 newdir2 ...
   ```
   and then commit.

2. Finally, push the changes into the branch of your fork:

   ```
   git push
   ```

### Step 7. Submit Your PR

1. Go to github and make a new Pull Request:

   Usually, if you go to https://github.com/USERNAME/fastai github will notice that you committed to a new branch and will offer you to make a PR, so you don't need to figure out how to do it.

   If for any reason it's not working, go to https://github.com/USERNAME/fastai/tree/new-feature-branch (replace `new-feature-branch` with the real branch name, and click `[Pull Request]` in the right upper corner.

If you work on several unrelated PRs, make different directories for each one, ideally using the same directory name as the branch name, to simplify things.


### How to Keep Your Feature Branch Up-to-date

If you synced the `master` branch with the original repository and you have feature branches that you're still working on, now you want to update those. For example to update your previously existing branch `my-cool-feature`:

   ```
   git checkout master
   git pull
   git checkout my-cool-feature
   ```

### How To Reset Your Forked Master Branch

If you haven't been careful to create a branch, and committed to the `master` branch of your forked repository, you no longer will be able to sync it with the original repository, without resetting it. And when you will want to create a branch, it'll have issues during PR, since it will be made against a diverged origin.

Of course, the brute-force approach is to go to github, delete your fork (which will delete any of the work you have done on this fork, including any branches, so be very careful if you decided to do that, since there will be no way to recover your data).

A much safer approach is to reset the `HEAD` of your forked `master` with the `HEAD` of the original repository:

If you haven't setup up the upstream, do it now:
   ```
   git remote add upstream git@github.com:fastai/REPONAME.git
   ```

and then do the reset:
   ```
   git fetch upstream
   git update-ref refs/heads/master refs/remotes/upstream/master
   git checkout master
   git stash
   git reset --hard upstream/master
   git push origin master --force
   ```

### Where am I?

Now that you have the original repository, the forked repository and its branches how do you know which of the repository and the branch you are currently in?

* Which repository am I in?

   ```
   git config --get remote.origin.url | sed 's|^.*//||; s/.*@//; s/[^:/]\+[:/]//; s/.git$//'
   ```
   e.g.: `stas00/fastai`

* Which branch am I on?

   ```
   git branch | sed -n '/\* /s///p'
   ```
   e.g.: `new-feature-branch7`

* Combined:

   ```
   echo $(git config --get remote.origin.url | sed 's|^.*//||; s/.*@//; s/[^:/]\+[:/]//; s/.git$//')/$(git branch | sed -n '/\* /s///p')
   ```
   e.g.: `stas00/fastai/new-feature-branch7`

But that's not a very efficient process to constantly ask the system to tell you where you are. Why not make it automatic and integrate this into your bash prompt (assuming that use bash).

#### bash-git-prompt

Enter [`bash-git-prompt`](https://github.com/magicmonty/bash-git-prompt), which not only tells you which virtual environment you are in and which `branch` you're on, but it also provides very useful visual indications on the state of your git checkout - how many files have changed, how many commits are waiting to be pushed, whether there are any upstream changes, and much more.

This is not finalized yet, but there is a PR that incorporates `username` and `repository` into the prompt too. Until it gets merged into the parent repository, use this [fork](https://github.com/stas00/bash-git-prompt) and change your `bash-git-prompt` theme to include:

   ```
   GIT_PROMPT_PREFIX="[${Blue}_USERNAME_REPO_|" # start of the git info string
   ```
or have a look at this [theme](https://github.com/stas00/bash-git-prompt/blob/master/themes/Single_line_username_repo.bgptheme).

I currently work on 4 different `fastai` project repositories and 4 corresponding forks, and several branches in all of them, so I was very lost until I started using this tool. To give you a visual of various prompts I have as of this writing:

   ```
   (pytorch-dev) /fastai/ci-experiments [fastai/fastai:ci-experiments|·6]>

   (pytorch-dev) /fastai/linkcheck [fastai/fastai:master]>

   (pytorch-dev) /stas00/fork [stas00/fastai:master|·3]>

   (pytorch-dev) /fastai/wip [fastai/fastai:master|+2?10·3]>
   ```

The numbers after the branch are modified/untracked/stashed counts. The leading `(pytorch-dev)` is the currently activated conda env name.

If you're not using `bash` or `fish` shell, search for forks of this idea for other shells.



## hub

hub == hub helps you win at git

[`hub`](https://github.com/github/hub) is the command line GitHub. It provides integration between git and github in command line. One of the most useful commands is creating pull request by just typing `hub pull-request` in your terminal.

Installation:

There is a variety of [ways to install](https://github.com/github/hub#installation) this application (written in go), but the easiest is to download the latest binary for your platform at https://github.com/github/hub/releases/latest, un-archiving the package and running `./install`, for example for the `linux-64` build:

```
wget https://github.com/github/hub/releases/download/v2.5.1/hub-linux-amd64-2.5.1.tgz
tar -xvzf hub-linux-amd64-2.5.1.tgz
cd hub-linux-amd64-2.5.1
sudo ./install
```

You can add a prefix to install it to a different location, for example, under your home:

```
prefix=~ ./install
```

or say you wanted to install it inside your active conda environment:

```
prefix=`which conda | sed 's/\/bin\/conda//'` ./install
```

Either of the these two should give you the location of the your active conda environment:
```
which conda | sed 's/\/bin\/conda//'
conda info | grep 'location' | awk '{print $5}'
```
but the first one is more reliable, `conda info`'s output may change down the road.

HELP-WANTED: If you'd like to contribute a little tool, this process could be automated, by getting the json output of all platform-specific urls for the latest binary release:

```
curl https://api.github.com/repos/github/hub/releases/latest
```
identifying user's platform, retrieving the corresponding to that platform package, unarchiving it, identifying the conda base as shown above, and running `install` with that prefix. If you work on it, please write it in python, so that windows users w/o bash could use it too. It'd go into `tools/hub-install` in the `fastai` repo.


## Github Shortcuts

* show commits by author: `?author=github_username`

   You can filter commits by author in the commit view by appending param `?author=github_username`.

   For example, the link https://github.com/fastai/fastai/commits/master?author=jph00 shows a list of commits `jph00` commits to the fastai repository.

* show commits by range: `master@{time}..master`

   You can create a compare view in GitHub by using the URL `github.com/user/repo/compare/{range}`. Range can be two SHAs like sha1…sha2 or two branch names like `master…my-branch`. Range is also smart enough to take time into consideration.

   For example, you can filter a list of commits since yesterday by using format like `master@{1.day.ago}…master`. The link https://github.com/fastai/fastai/compare/master@{1.day.ago}…master, for example, gets all commits since yesterday for the `fastai` repository:

* show `.diff` & `.patch`

   Add `.diff` or `.patch` to the URLs of compare view, pull request or commit page to get the diff or patch in text format.

   For example, the link https://github.com/fastai/fastai/compare/master@{1.day.ago}…master.patch gets the patch for all the commits since yesterday in the `fastai` repository.

* line linking

   In any file view, when you click one line or multiple lines by pressing SHIFT, the URL will change to reflect your selections. You can tell others to look at a specific line of code, or a specific chunk of code, using just that link.

* delete a fork

   1. Go to github.com/USERNAME/FORKED-REPO-NAME/
   2. Hit Settings
   3. Scroll down and hit [Delete this repository]

   replace, `USERNAME` with your github username, and `FORKED-REPO-NAME` with the repository name




## Revisions

relative refs
```
^      - one commit at a time (parent of the specified commit)
master^  = the first parent of master
master^^ = the first grandparent of master
~<num> - several commits
```


## Operations


### add
```
git add [folder/file]
```


### remove
```
git rm [folder/file]
```

remove remote file copy only. e.g. remove database.yml that is already checked in but leaving the local copy untouched. This is intensively handy for removing ignored files that are already pushed without removing the local copies.
```
git rm --cached database.yml
```


### status
```
git status
```

brief status
```
git status -s
```


###  push
```
git push
```


dry-run (do everything except for the actually sending of the data)
```
git push --dry-run
```
but it doesn't show anything useful - see commands below for visual hints of what will happen


show which files have changed and view the diff compared to the remote master branch HEAD
```
git diff --stat --patch origin master
```

list of files to be pushed
```
git diff --stat --cached [remote/branch]
```

show code diff of the files to be pushed
```
git diff [remote repo/branch]
```

show full file paths of the files that will change
```
git diff --numstat [remote repo/branch]
```


### commit
```
git commit -a
```

`-a` is crucial as w/o it you need to `git add` every file that has changed!

There is also `-A`, but careful using it, as it'll add any tracked files, which is probably not what you want most of the time. Better forget about this option.


### authentication

cache auth
```
git config --global credential.helper cache
```

adjust caching time
```
git config --global credential.helper 'cache --timeout=36000'
```


### update
```
git pull
```


git pull is shorthand for
```
git fetch
git merge FETCH_HEAD
```


display the incoming/outgoing changes before pull/push
```
git log ^master origin/master
git log master ^origin/master
```


### search/replace


How to safely and efficiently search/replace files in git repo using CLI. The operation must not touch anything under .git/
```
find . -type d -name ".git" -prune -o -type f -exec perl -pi -e 's|OLDSTR|NEWSTR|g' {} \;
```
but it touch(1)es all files which slows down git-side

so we want to do it on files that actually contain the old pattern
```
grep --exclude-dir=.git -lIr "OLDSTR" . | xargs -n1 perl -pi -e 's|OLDSTR|NEWSTR|g'
```


### git GUI

git
```
git gui
```


gitk
```
gitk --all
```


### contributors


show a list of contributors ordered by number of commits. Similar to the contributors view of GitHub.
```
git shortlog -sn
```


### search git history


to find all commits where commit message contains given word, use
```
git log --grep=word_to_search_for
```

to search all of git history for a string
```
git log -Sword_to_search_for
```
this will find any commit that added or removed the string password. Here are a few extra options:

* `-p`: will show the diffs. If you provide a file (`-p file`), it will generate a patch for you.
* `-G`: looks for differences whose added or removed line matches the given regexp, as opposed to
* `-S`, which "looks for differences that introduce or remove an instance of string".
* `--all`: searches over all branches and tags; alternatively, use `--branches[=<pattern>]` or `--tags[=<pattern>]`

search and exclude certain paths from the results:

exclude subfolder foo
```
git log -- . ":(exclude)foo"
```
exclude several subfolders
```
git log -- . ":(exclude)foo" ":(exclude)bar"
```
exclude specific elements in that subfolder
```
git log -- . ":(exclude)foo/bar/file"
```
exclude any given file in that subfolder
```
git log -- . ":(exclude)foo/*file"
git log -- . ":(exclude,glob)foo/*file"
```
make exclude case insensitive
```
git log -- . ":(exclude,icase)FOO"
```


which branch contains a specified sha key
```
git branch –contains SHA
```

### cherry picking

choose a commit rev from one branch (e.g. PR) and merge it the current checkout
```
git show <commit>        # check that this is the right rev
git cherry-pick <commit> # merge it into the current checkout
git push
```


to merge a range of commits:
```
git cherry-pick <commit1>..<commitN>
```


cherry picking parts of a commit (only sections/hunks and not whole files)
```
git cherry-pick -n <commit> # get your patch, but don't commit (-n = --no-commit)
git reset                   # unstage the changes from the cherry-picked commit
git add -p                  # make all your choices (add the changes you do want)
git commit                  # make the commit!
```
similar to the above 4 commands - interactive picking (-p == --patch)
```
git checkout -p <commit>
```


and if only changes for specific files are wanted:
```
git checkout -p <commit> -- path/to/file_a path/to/file_b
```


cherry-pick another git repo (can use sha1 instead of FETCH_HEAD)
```
git fetch <remote-git-url> <branch> && git cherry-pick FETCH_HEAD
```


abort the started cherry-pick process, which will revert to the previous state
```
git cherry-pick --abort
```


### checkout



checkout a specific commit
```
git checkout <sha1>/or-short-hash
```

check out a specific branch
```
git clone https://github.com/vidartf/nbdime -b optimize-diff2
```

### overwrite local changes



If you want to remove all local changes from your working copy, simply stash them:
```
git stash push --keep-index
```


or if it's important you can name it
```
git stash push "your message here"
```


to merge the local changes saved with 'git stash push' after 'git pull'
```
git stash pop
```

if the merge fails, it doesn't get removed from the stash.

once merge conflict is manually removed, need to manually call:
```
git stash drop
```


If you don't need them anymore, you now can drop that stash:
```
git stash drop
```


to override all local changes and does not require an identity:
```
git reset --hard
git pull
```


or:
```
git checkout -t -f remote/branch
git pull
```


Discard local changes for a specific file
```
git checkout dirs-or-files
git pull
```


maintain current local commits by creating a branch from master before resetting
```
git checkout master
git branch new-branch-to-save-current-commits
git fetch --all
git reset --hard origin/master
```


pull from upstream and accept all changes blindly
```
git pull --strategy theirs
```


list existing stashes
```
git stash list
```


vies stashes:

latest
```
git stash show -p
```

specific stash
```
git stash show -p stash@{0}
```


show the contents of each stash with one command
```
git show $(git stash list | cut -d":" -f 1)
```


diff against a specific stash
```
git diff stash@{0}
```


diff against a specific stash's filename
```
git diff stash@{0} my/file.ipynb
```


diff 2 stashes:
```
git diff stash@{0}..stash@{1}
```

check out nbdime - diffing and merging of Jupyter Notebooks
https://nbdime.readthedocs.io/en/stable/





### branches



git branch removal (when not checkout'ed inside the branch that's about to be removed)
```
git branch -d branch_name
```

branch delete via github - after the branch has been merged into the master upsteam, can now delete the branch in my fork at github.com
```
1. https://github.com/stas00/fastai/branches
```


or go to https://github.com/stas00/fastai/ (and click [NN branches] above [New pull request] button
```
1. hit the trash button next to the branch to remove
```


list branches that are merged or not yet merged to current branch. It’s a useful check before any merging happens
```
git branch –merged
git branch –no-merged
```


switch back to last branch (like `cd -`)
```
git checkout -
```


`@{-1}` is a way to refer to the last branch you were on. '-' is shorthand for `@{-1}`
`git branch --track mybranch @{-1}`, `git merge @{-1}`, and `git rev-parse --symbolic-full-name @{-1}` would work as expected.



compare two branches in the same repo
```
git diff --stat --color master..branch_name
```
or:
```
git difftool -d master branch_name
```


find the diff from their common ancestor to test, you can use ... instead of ..:
```
git diff --stat --color master...branch_name
```


to compare just specific files
```
git diff branch1 branch2 -- myfile1.js myfile2.js
```


to compare a sub-directory or specific files across different commits
```
git diff <rev1>..<rev2> -- dir1 file2
```


compare two branches in different repos (e.g. original and github fork)



given 2 checkouts `/path/to/repoA` and `/path/to/repoB`
```
cd /path/to/repoA
GIT_ALTERNATE_OBJECT_DIRECTORIES=/path/to/repoB/.git/objects git diff $(git --git-dir=/path/to/repoB/.git rev-parse --verify HEAD) HEAD
```


another way using GUI with meld (apt install meld)
```
meld /f1/br/stas00/master/ /f1/br/fastai/master
```


find the best common ancestor between two branches, usually the branching point:
```
git merge-base master origin/branch_name
```


same, but returns a short rev instead of the long one
```
git rev-parse --short $(git merge-base master origin/branch_name)
```
alternative (doesn't always work):
```
git merge-base --fork-point master origin/branch_name
```
note that 'git merge-base' returns no output once that branch has been merged to master.



diff between the branching point and the HEAD of the branch
```
git diff $(git merge-base --fork-point master origin/branch_name)..origin/branch_name
```


commits between the branching point and the HEAD of the branch
```
git log  --oneline $(git merge-base --fork-point master origin/branch_name)..origin/branch_name
```


find branches the commit is on
```
git branch --contains <commit>
```


find when a commit was merged into one or more branches.

https://github.com/mhagger/git-when-merged
```
git when-merged [OPTIONS] COMMIT [BRANCH...]
```


some good docs on branching strategies:
```
https://nvie.com/posts/a-successful-git-branching-model/
```


### reverting/resetting/undoing

lots of scenarios here:
```
https://blog.github.com/2015-06-08-how-to-undo-almost-anything-with-git/
```


revert the last commit
```
git revert HEAD
```


revert everything from the HEAD back to the commit hash 0766c053
```
git revert --no-commit 0766c053..HEAD
git commit
```
this will revert everything from the HEAD back to the commit hash, meaning it will recreate that commit state in the working tree as if every commit since had been walked back. You can then commit the current tree, and it will create a brand new commit essentially equivalent to the commit you "reverted" to.

(the `--no-commit` flag lets git revert all the commits at once- otherwise you'll be prompted for a message for each commit in the range, littering your history with unnecessary new commits.)

this is a safe and easy way to rollback to a previous state. No history is destroyed, so it can be used for commits that have already been made public.

if merge happened earlier, revert could fail and ask for a specific parent branch via -m flag to specify which mainline to use

for details: http://schacon.github.io/git/howto/revert-a-faulty-merge.txt and https://stackoverflow.com/questions/5970889/why-does-git-revert-complain-about-a-missing-m-option


revert your repository to a specific revision
```
git checkout <rev>
```


revert only parts of your repository to a specific revision
```
git checkout <rev> -- dir1 dir2 file1 file2
```


### ignore

to temporarily ignore changes in a certain file, run:
```
git update-index --assume-unchanged <file>
```

track changes again:
```
git update-index --no-assume-unchanged <file>
```


### trace and debug



check which config comes from where
```
git config --list --show-origin
```


display git attributes for a specific path
```
git check-attr -a dev_nb/001b_fit.ipynb
```


more here: https://git-scm.com/book/en/v2/Git-Tools-Debugging-with-Git



trace
```
GIT_TRACE=1 git pull origin master
```


very verbose
```
set -x; GIT_TRACE=2 GIT_CURL_VERBOSE=2 GIT_TRACE_PERFORMANCE=2 GIT_TRACE_PACK_ACCESS=2 GIT_TRACE_PACKET=2 GIT_TRACE_PACKFILE=2 GIT_TRACE_SETUP=2 GIT_TRACE_SHALLOW=2 git pull origin master -v -v; set +x
```


different options:
```
    GIT_TRACE for general traces,
    GIT_TRACE_PACK_ACCESS for tracing of packfile access,
    GIT_TRACE_PACKET for packet-level tracing for network operations,
    GIT_TRACE_PERFORMANCE for logging the performance data,
    GIT_TRACE_SETUP for information about discovering the repository and environment it’s interacting with,
    GIT_MERGE_VERBOSITY for debugging recursive merge strategy (values: 0-5),
    GIT_CURL_VERBOSE for logging all curl messages (equivalent to curl -v),
    GIT_TRACE_SHALLOW for debugging fetching/cloning of shallow repositories.

possible values can include:

    true, 1 or 2 to write to stderr,
    an absolute path starting with / to trace output to the specified file.
```


### status and information

short form log of events
```
git log --oneline
```


show a graph of the tree, showing the branch structure of merges
```
git log --graph --decorate --pretty=oneline --abbrev-commit
```
add `--all` to show all branches


show all the commits in a branch that are not in HEAD. e.g. show all commits that are in master but not merged into the current feature branch yet.
```
git log ..master
```



### overriding git configuration
```
git -c http.proxy=someproxy clone https://github.com/user/repo.git
git -c user.email=email@domain.fr -c user.name='Your Name'
```

override git diff:
```
git diff --no-ext-diff
```
no such option exists for merge drivers.



## fixing things



to fix a bad merge
```
https://stackoverflow.com/questions/307828/how-do-you-fix-a-bad-merge-and-replay-your-good-commits-onto-a-fixed-merge
```


"fatal: Unknown index entry format 61740000".



when your index is broken you can normally delete the index file and reset it.
```
rm -f .git/index
git reset
```


or you clone the repo again.



## merge strategies



tell git not to merge certain files (i.e. keep the local version) by defining merge filter 'ours'.

https://stackoverflow.com/a/5895890/9201239



1) add to .gitattributes:
```
database.xml merge=ours
```


2) set git merge driver to do nothing but return success
```
git config merge.ours.name '"always keep ours" merge driver'
git config merge.ours.driver 'touch %A'
git config merge.ours.driver true
```


## workflows


working and updating the local checkout with upstream changes https://stackoverflow.com/questions/457927/git-workflow-and-rebase-vs-merge-questions?rq=1
```
clone the remote repository
git checkout -b my_new_feature
..work and commit some stuff
git rebase master
..work and commit some stuff
git rebase master
..finish the feature, commit
git rebase master
git checkout master
git merge --squash my_new_feature
git commit -m "added my_new_feature"
git branch -D my_new_feature
```


## Aliases

best to add manually with editor, but can use CLI
```
.gitconfig
   [alias]
```

e.g.
```
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
```

unstage a file (equivalent of: `git reset HEAD -- fileA`:
```
git config --global alias.unstage 'reset HEAD --'
```

see last commit
```
git config --global alias.last 'log -1 HEAD'
```

use `!` for non-git sub-commands in aliases, e.g.:
```
git config --global alias.visual '!gitk'
```


## git hooks and filters

### strip output from Jupyter and IPython notebooks


install nbstripout
```
pip install nbstripout
```


check it's in the path:
```
which nbstripout
```


switch to the repository you want to work in
```
cd fastai_v1/
```


#### automatic install for git commit and git diff
```
nbstripout --install
```


#### manual git commit instrumentation



add to .gitattributes or .git/info/attributes:
```
*.ipynb filter=nbstripout
```


these will modify .git/config
```
git config filter.nbstripout.clean `which nbstripout`
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true
```


#### manual git diff instrumentation



add to .gitattributes or .git/info/attributes:
```
*.ipynb diff=ipynb
```


this will modify .git/config
```
git config diff.ipynb.textconv "$(which nbstripout) -t"
```


### git filters

- before check in: clean filter
- before checkout: smudge filter


to check what the "clean" filter produced (to see the actual contents of the index)
```
git show :0:repo-relative/path/to/file
```

you can not usually use git diff for this since it also applies the filters.

report all attributes set on file
```
git check-attr -a repo-relative/path/to/file
```


#### useful git filters

git keyword expansion.
```
https://github.com/gistya/expandr
```



## Miscellaneous Recipes

* download a sub-directory from a git tree, e.g. https://github.com/buckyroberts/Source-Code-from-Tutorials/tree/master/Python

   1. replace tree/master => trunk
   2. svn co the new url
   ```
   svn co https://github.com/buckyroberts/Source-Code-from-Tutorials/trunk/Python
   ```


## Useful Resources

* https://learngitbranching.js.org/ - visual teaching with exercises
