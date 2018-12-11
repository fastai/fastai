---
title: Performance Tips and Tricks
---

This document will show you how to speed things up and get more out of your GPU/CPU.

# Mixed Precision Training

Combined FP16/FP32 training can tremendously improve training speed and use less GPU RAM. For theory behind it see:

https://forums.fast.ai/t/mixed-precision-training/20720/3

To deploy it see: http://docs.fast.ai/callbacks.fp16.html
