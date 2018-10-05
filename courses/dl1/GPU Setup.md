# How To Set Up Your GPU Notebook

There are various ways to go about setting up your Python Notebook to run on the GPU.

Below I've linked those those mentioned in the lectures and relatively quick to set up. They do however require payment based on useage
basis.

* [AWS](https://github.com/reshamas/fastai_deeplearn_part1/blob/master/tools/aws_ami_gpu_setup.md)
* [Paperspace](https://github.com/reshamas/fastai_deeplearn_part1/blob/master/tools/paperspace.md)
* [Crestle](https://github.com/reshamas/fastai_deeplearn_part1/blob/master/tools/crestle_run.md)

Another option is to use Google Colab which is free and gives a good way to get a course taster.

The process for set up is simple enough. Going to https://colab.research.google.com/ you are presented with a Jupyter notebook where you 
can add the various dependencies, install the fast.ai library and you are good to go!

## Below are the instructions for setting up on Google Colab to run the lessons.

As usual, install what you may need:

### [Optional: [magic functions](https://stackoverflow.com/questions/20961287/what-is-pylab)]

e.g. 

![Imgur](https://i.imgur.com/vronrOn.png)

### Install fast.ai library

```python
!pip install fastai
```

The '!' tells the Python notebook to run a command like that in your terminal.


### Import the functions you need from your modules

e.g.

```python
from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics
```

### Set up your path as relevant 

In the course, Jeremy Howard specifies what path he uses. Remember the '!' command and set up the path like you would on Terminal. You can
also type 'files.upload()' if you would like to upload any files, use the terminal command to unzip any files you may possibly want. But 
otherwise also import from Google Drive or Github a notebook and run it.
