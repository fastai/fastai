# How to make a Pull Request(PR)
For this tutorial we are going to use GitHub desktop, github.com, and a text editor of your choice.

## 1. **GitHub**
Make sure you are logged in to your GitHub account. ([sign up](https://github.com/join?source=header-home) or [sign in](https://github.com/login?return_to=%2Fjoin%3Fsource%3Dheader-home) )

## 2. **GitHub Desktop**
Download([windows](https://desktop.github.com) or [macOS](https://central.github.com/deployments/desktop/desktop/latest/darwin)) or open GitHub Desktop

## 3. **Tutorials** 
If you are new to GitHub, a good place to start is GitHub's ["Hello World" tutorial](https://guides.github.com/activities/hello-world/) it cover some of the features/terms you are going to need to know([What is GitHub?](https://guides.github.com/activities/hello-world/#what), [Create a Repository(Repo](https://guides.github.com/activities/hello-world/#repository), [Create a Branch](https://guides.github.com/activities/hello-world/#branch), [Make a Commit](https://guides.github.com/activities/hello-world/#commit), [Open a Pull Request(PR)](https://guides.github.com/activities/hello-world/#pr), [Merge Pull Request](https://guides.github.com/activities/hello-world/#merge)). Some other options are [hubspot's git-and-github-tutorial-for-beginners](https://product.hubspot.com/blog/git-and-github-tutorial-for-beginners), or [github-flow](https://guides.github.com/introduction/flow/)
  
## 4. **Clone & Fork** 
I chose to [Clone](https://help.github.com/articles/cloning-a-repository/) as well as fork the fastai GitHub repo. The reason I chose to do this is I could use the official GitHub repo for reading/class-work, and use the forked one to make my proposed change(s) to the library. 

## 5. **Clone It**
![clone|690x338](images\pr_tutorial\clone.png)
To me cloning a repo is like borrowing a library book, it is possible to make changes/commits(write in it) when it is in your possession, but changes you make could cause problems/errors(conflicts) since you don't have permission to make changes/updates. The changes you make will only be on your local copy, and if you try to return-it/update-it, it might cause issues. Since most library and companies have a code review process, you can't create a pull request(PR) from a cloned copy. You need to do that from a Forked copy.

## 6. **Fork It**
![Fork|690x338](images\pr_tutorial\Fork.png)
As time goes by, chages will be made to the fastai library, and you will need to update your forked copy with the new updates by [syncing the fork](https://help.github.com/articles/syncing-a-fork/). If you very recently forked the repo you won't have to worry about this, but if it has been more than a couple days you will most likely want to update it before making your changes. See the [fastai-docs "start-with-a-synced-fork-checkout"](https://docs-dev.fast.ai/git.html#step-1-start-with-a-synced-fork-checkout) and [syncing-"subsequent-times"](https://docs-dev.fast.ai/git.html#1b-subsequent-times), personally I think it is easier for the beginner to just use GitHub-desktop to get updates/sync-fork.  

**Tip** - *You probably want a unique name for your fork because if you use fastai, it can be confusing because fastai/fastai and YourGitHubUserName/fastai will both show as fastai in the current repository section*

## 7. **Open GitHub-desktop**
![navigation|589x500](images\pr_tutorial\navigation.png) 
Every repo has a "master" "branch", this is your main version. The "repository"(repo) could be the source, or it could be a "fork" of that source. The source(repo forked from) is considered to be"upstream", its main version would be considered "upstream/master". 

## 8. **Checking for updates/changes/syncing-fork**
![Merge%20|690x454](images\pr_tutorial\Merge%20.png) 
![merge_detail|500x400](images\pr_tutorial\merge_detail.png) 

## 9. **Branches**
Branches are a very important part of the git/github process. Generally speaking, you want to create a new branch for each new feature. 
> Fastai dev docs:
"It’s very important that you  **always work inside a branch** . If you make any commits into the  `master`  branch, you will not be able to make more than one PR at the same time, and you will not be able to synchronize your forked  `master`  branch with the original without doing a reset. If you made a mistake and committed to the  `master`  branch, it’s not the end of the world, it’s just that you made your life more complicated."

![create_new_branch.png|539x500](images\pr_tutorial\create_new_branch.png) 


## 10.  **Make your changes in the text editor of  your choice**
Github-desktop tracks the local folder that your repo is in, and if you make any changes to the files or folder in that folder, GitHub-desktop keeps track of all the changes, so when you log back into GitHub-desktop, it knows what needs to be committed/updated. 
![commit%20your%20chages%20to%20your%20master|539x500](images\pr_tutorial\commit%20your%20chages%20to%20your%20master.png) 
Once you commit to master, It will now be updated/committed on github.com as well. You can then submit a Pull Request on gitHub.com, in GitHub-Desktop, or in the [terminal](https://docs-dev.fast.ai/git.html#step-7-submit-your-pr).

***Keep in mind if you are updating the Jupiter-notebooks(docs) you will need to perform [additional steps](https://docs-dev.fast.ai/develop#stripping-out-jupyter-notebooks).***

## 11. **Pull Request via GitHub.com**
Log-in to github.com, go into your forked repo.
 ![pull%20request|690x361](images\pr_tutorial\pull%20request.png)
Compare changes
![PR%20Detail|690x406](images\pr_tutorial\PR%20Detail.png)
Add in description and documentation
![PR%20DESC|690x415](images\pr_tutorial\PR%20DESC.png) 

More information is available in the [fastai developer documentation](https://docs-dev.fast.ai/develop)

**Now Git to It!**