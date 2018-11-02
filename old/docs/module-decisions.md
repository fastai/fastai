# Module Decisions

## Introduction

There are many ways of doing one thing in programming. Instead of getting into debates about the one right way of doing things, in `fastai` library we would like to make decisions and then stick with them. This page is to list down any such decisions made.

### Image Data
- Coordinates
 - Computer vision uses coordinates in format `(x, y)`. e.g. PIL
 - Maths uses `(y, x)`. e.g. Numpy, PyTorch
 - `fastai` will use `(y, x)`
- Bounding Boxes
 - Will use `(coordinates top right, coordinates bottom right)` instead of `(coordinates top right, (height, width))`
