import matplotlib
matplotlib.use('agg')
import cv2,torch

# the above imports are fixing the TLS issue:
# ```ImportError: dlopen: cannot load any more object with static TLS```
# they were set after experimenting with the test sets on ubuntu 16.04
