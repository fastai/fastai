---
title: git Notes
---

## Revisions



relative refs
```
^      - one commit at a time (parent of the specified commit)
master^  = the first parent of master
master^^ = the first grandparent of master
~<num> - several commits
```


## Commands


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
2. hit the trash button next to the branch to remove
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


## trace and debug



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


## Status And Information

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


## reverting/resetting/undoing

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

(the --no-commit flag lets git revert all the commits at once- otherwise you'll be prompted for a message for each commit in the range, littering your history with unnecessary new commits.)

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



### Workflows


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


### Aliases

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





### GitHub-specific notes



### Github shortcuts

* show commits by range: master@{time}..master

   You can create a compare view in GitHub by using the URL github.com/user/repo/compare/{range}. Range can be two SHAs like sha1…sha2 or two branches’ name like master…my-branch. Range is also smart enough to take time into consideration.

   For example, you can filter a list of commits since yesterday by using format like master@{1.day.ago}…master. The link https://github.com/rails/rails/compare/master@{1.day.ago}…master, for example, gets all commits since yesterday for the Rails project:


* show commits by author: ?author=github_handle

   You can filter commits by author in the commit view by appending param ?author=github_handle.

   For example, the link https://github.com/dynjs/dynjs/commits/master?author=jingweno shows a list of my commits to the Dynjs project



* show .diff & .patch

   Add .diff or .patch to the URLs of compare view, pull request or commit page to get the diff or patch in text format.

   For example, the link https://github.com/rails/rails/compare/master@{1.day.ago}…master.patch gets the patch for all the commits since yesterday in the Rails project

* line linking

   In any file view, when you click one line or multiple lines by pressing SHIFT, the URL will change to reflect your selections. This is very handy for sharing the link to a chunk of code with your teammates

* hub

   Hub is the command line GitHub. It provides integration between Git and GitHub in command line. One of the most useful commands is creating pull request by just typing `hub pull-request` in your terminal. Detail of all other commands is available on its project readme.



* delete a fork

1. Go to github.com/stas00/[FORKED-REPO]/
2. Hit Settings
3. Scroll down and hit [Delete this repository]


### sync master fork with master repository

#### Syncing a Fork method

https://help.github.com/articles/syncing-a-fork/


step 1. Configuring a remote for a fork

```
git clone git://github.com/stas00/fastai sync

cd sync
git remote -v
git remote add origin   git@github.com:stas00/fastai.git
git remote add upstream git@github.com:fastai/fastai.git
git remote -v
```


non-SSH (password auth) version

```
git clone https://github.com/stas00/fastai sync
cd sync
git remote add upstream https://github.com/fastai/fastai.git
```

step 2. Syncing a fork
```
git fetch upstream
git checkout master
git merge --no-edit upstream/master
git push --set-upstream origin master
```

or one line:
```
git fetch upstream; git checkout master; git merge --no-edit upstream/master; git push --set-upstream origin master
```


#### Making a pull request

unless you know how to handle switching branches, it's the easiest to make a new clone for each new branch

1. Sync the forked version with master

see above

2. clone the main repository fork (or if it hasn't been done yet fork it first on github by going to github.com:fastai/fastai)

moreover - make sure that the repository fork is up-to-date by syncing it

```
git clone git://github.com/stas00/fastai branch_name
cd branch_name
git remote rm origin
git remote add origin git@github.com:stas00/fastai.git
```


3. create a branch
```
git checkout -b branch_name
```

if the master sync has since updated:
```
git rebase master
```


4. edit code and commit changes

may validate the changes:
```
jsonlint-php lesson7-cifar10.ipynb
```

commit
```
git commit -a
```


5. push the changes into the branch
```
git push origin branch_name
```


6. go to github and make pull request

https://github.com/stas00/fastai/tree/branch_name/
and click [Pull Request] on the right upper corner


## Useful Resources

* https://learngitbranching.js.org/ - visual teaching with exercises


## Extra Recipes

download a sub-directory from a git tree, e.g. https://github.com/buckyroberts/Source-Code-from-Tutorials/tree/master/Python

1. replace tree/master => trunk
2. svn co the new url
```
svn co https://github.com/buckyroberts/Source-Code-from-Tutorials/trunk/Python
```
