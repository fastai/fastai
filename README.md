# fastai

The fast.ai deep learning library, lessons, and tutorials.

Copyright 2017 onwards, Jeremy Howard. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. A copy of the License is provided in the LICENSE file in this repository.

## Current Status

This is an alpha version. Most of the library is quite well tested since many students have used it to complete the [Practical Deep Learning for Coders](http://course.fast.ai). course. However it hasn't been widely used yet outside of the course, so you may find some missing features or rough edges. If you're interested in using the library in your own projects, we're happy to help support any bug fixes or feature additions you need&mdash;please use http://forums.fast.ai to discuss.

Recommended installation approach is to clone fastai using `git`:

```sh
git clone https://github.com/fastai/fastai.git
```
Then, `cd` to the fastai folder and create the python environment:

```sh
cd fastai
conda env update
```
This downloads all of the dependencies and then all you have to do is:

```sh
conda activate fastai
```

To update everything at any time, cd to your repo and:

```sh
git pull
conda env update
```

To install a cpu only environment instead:
```sh
cd fastai
conda env update -f environment-cpu.yml
```

You can also install this library in the local environment using ```pip```

```sh
pip install fastai
```

However this is not currently the recommended approach, since the library is being updated much more frequently than the pip release, fewer people are using and testing the pip version, and pip needs to compile many libraries from scratch (which can be slow). 

