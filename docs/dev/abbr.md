---
title: fastai Abbreviation Guide
---

As mentioned in the [fastai style](style.md),
we name symbols following the *Huffman Coding* principle, which basically means

> Commonly used and generic concepts should be named shorter. You shouldn't waste short sequences on less common concepts.

fastai also follows the life-cycle naming principle:
> The shorter life a symbol, the shorter name it should have.

which means:
- **Aggressive Abbreviations** are used in *list comprehensions*, *lambda* functions, *local helper* functions.
- **Aggressive Abbreviations** are sometimes used for *local temporary variables* inside a function.
- **Common Abbreviations** are used most elsewhere, especially for *function arguments*, *function names*, and *variables*
- **Light or No Abbreviations** are used for *module names*, *class names* or *constructor methods*, since they basically live forever.
However, when a class or module is very popular, we could consider using abbreviations to shorten its name.

This document lists abbreviations of common concepts that are consistently used across the whole fastai project. For naming of
domain-specific concepts, you should check their corresponding module documentations. Concepts are grouped and listed by semantic order.
Note that there are always exceptions, especially when we try to comply with the naming convention in a library.

|                  | **Concept**                         | **Abbr.**      | **Combination Examples**                         |
|------------------|-------------------------------------|----------------|--------------------------------------------------|
| **Suffix**       |                                     |                |                                                  |
|                  | multiple of something (plural)      | s              | xs, ys, tfms, args, ss                           |
|                  | internal property or method         | _              | data_, V_()                                      |
| **Prefix**       |                                     |                |                                                  |
|                  | check if satisfied                  | is_            | is_reg, is_multi, is_single, is_test, is_correct |
|                  | On/off a feature                    | use_           | use_bn                                           |
|                  | Number of something (plural)        | n_             | n_embs, n_factors, n_users, n_items              |
|                  | count something                     | num_           | num_features(), num_gpus()                       |
|                  | convert to something                | to_            | to_gpu(), to_cpu(), to_np()                      |
| **Infix**        |                                     |                |                                                  |
|                  | Convert between concepts            | 2              | name2idx(), label2idx(), seq2seq                 |
| **Aggressive**   |                                     |                |                                                  |
|                  | function                            | f              |                                                  |
|                  | torch input                         | x              |                                                  |
|                  | key, value                          | k,v            | for k,v in d.items()                             |
|                  | other pairs of short scope          | p,q            | listify(p,q) (same as python's stdlib)           |
|                  | index                               | i              |                                                  |
|                  | generic object argument             | o              | [o for o in list], lambda o: o                   |
|                  | variable                            | v              | V(), VV()                                        |
|                  | tensor                              | t              | T()                                              |
|                  | array                               | a              | A()                                              |
|                  | use first letter                    |                | *w*eight -> w, *m*odel -> m                      |
| **Generic**      |                                     |                |                                                  |
|                  | function                            | fn             | opt_fn, init_fn, reg_fn                          |
|                  | process                             | proc           | proc_col                                         |
|                  | transform                           | tfm            | tfm_y, TfmType                                   |
|                  | evaluate                            | eval           | eval()                                           |
|                  |                                     |                |                                                  |
|                  | argument                            | arg            |                                                  |
|                  | input                               | x              |                                                  |
|                  | input / output                      | io             |                                                  |
|                  | object                              | obj            |                                                  |
|                  | string                              | s              |                                                  |
|                  | class                               | cl             | cl, classes                                      |
|                  | source                              | src            |                                                  |
|                  | destination                         | dst            |                                                  |
|                  | directory                           | dir            |                                                  |
|                  | percentage                          | p              |                                                  |
|                  | ratio, proportion of something      | r              |                                                  |
|                  | count                               | cnt            |                                                  |
|                  |                                     |                |                                                  |
|                  | configuration                       | cfg            |                                                  |
|                  | random                              | rand           |                                                  |
|                  | utility                             | util           |                                                  |
|                  | filename                            | fname          |                                                  |
|                  |                                     |                |                                                  |
|                  | threshold                           | thresh         |                                                  |
| **Data**         |                                     |                |                                                  |
|                  | number of elements                  | n              |                                                  |
|                  | length                              | len            |                                                  |
|                  | size                                | sz             |                                                  |
|                  | array                               | arr            | label_arr                                        |
|                  | dictionary                          | dict           |                                                  |
|                  | sequence                            | seq            |                                                  |
|                  |                                     |                |                                                  |
|                  | dataset                             | ds             | train_ds                                         |
|                  | dataloader                          | dl             | train_dl                                         |
|                  | dataframe                           | df             | train_df                                         |
|                  | train                               | train          | train_ds, train_dl, train_x, train_y             |
|                  | validation                          | valid          | valid_ds, valid_dl, valid_x, valid_y             |
|                  | test                                | test           | test_ds,  test_dl                                |
|                  | number of classes                   | c              |                                                  |
|                  | batch                               | b              |                                                  |
|                  | batch's x parts                     | xb             |                                                  |
|                  | batch's y parts                     | yb             |                                                  |
|                  | batch size                          | bs             |                                                  |
|                  | multiple targets                    | multi          | is_multi                                         |
|                  | regression                          | reg            | is_reg                                           |
|                  | iterate, iterator                   | iter           | train_iter, valid_iter                               |
|                  |                                     |                |                                                  |
|                  | torch input                         | x              |                                                  |
|                  | target                              | y              |                                                  |
|                  | dependent var tensor                | dep            |                                                  |
|                  | independent var tensor              | indep          |                                                  |
|                  | prediction                          | pred           |                                                  |
|                  | output                              | out            |                                                  |
|                  | column                              | col            | dep_col                                          |
|                  | continuous var                      | cont           | conts                                            |
|                  | category var                        | cat            | cat, cats                                        |
|                  | continuous columns                  | cont_cols      |                                                  |
|                  | category  columns                   | cat_cols       |                                                  |
|                  | dependent column                    | dep_col        |                                                  |
|                  |                                     |                |                                                  |
|                  | index                               | idx            |                                                  |
|                  | identity                            | id             |                                                  |
|                  | first element                       | head           |                                                  |
|                  | last element                        | tail           |                                                  |
|                  |                                     |                |                                                  |
|                  | unique                              | uniq           |                                                  |
|                  | residual                            | res            |                                                  |
|                  | label                               | lbl            | (not common)                                     |
|                  | augment                             | aug            |                                                  |
|                  | padding                             | pad            |                                                  |
|                  |                                     |                |                                                  |
|                  | probability                         | pr             |                                                  |
|                  | image                               | img            |                                                  |
|                  | rectangle                           | rect           |                                                  |
|                  | color                               | colr           |                                                  |
|                  | anchor box                          | anc            |                                                  |
|                  | bounding box                        | bb             |                                                  |
|                  |                                     |                |                                                  |
| **Modeling**     |                                     |                |                                                  |
|                  | initialize                          | init           |                                                  |
|                  | language model                      | lm             |                                                  |
|                  | recurrent neural network            | rnn            |                                                  |
|                  | convolutional neural network        | convnet        |                                                  |
|                  |                                     |                |                                                  |
|                  | model data                          | md             |                                                  |
|                  | linear                              | lin            |                                                  |
|                  | embedding                           | emb            |                                                  |
|                  | batch norm                          | bn             |                                                  |
|                  | dropout                             | drop           |                                                  |
|                  | fully connected                     | fc             |                                                  |
|                  | convolution                         | conv           |                                                  |
|                  | hidden                              | hid            |                                                  |
|                  |                                     |                |                                                  |
|                  | optimizer (e.g. Adam)               | opt            |                                                  |
|                  | layer group learning rate optimizer | layer_opt      |                                                  |
|                  | criteria                            | crit           |                                                  |
|                  | weight decay                        | wd             |                                                  |
|                  | momentum                            | mom            |                                                  |
|                  | cross validation                    | cv             |                                                  |
|                  | learning rate                       | lr             |                                                  |
|                  | schedule                            | sched          |                                                  |
|                  | cycle length                        | cl             |                                                  |
|                  | multiplier                          | mult           |                                                  |
|                  | activation                          | actn           |                                                  |
|                  |                                     |                |                                                  |
| **CV**           | computer vision                     |                |                                                  |
|                  | figure                              | fig            |                                                  |
|                  | image                               | im             |                                                  |
|                  | transform image using opencv        | _cv            | zoom_cv(), rotate_cv(), stretch_cv()             |
| **NLP**          | natural language processing (nlp)   |                |                                                  |
|                  | token                               | tok            |                                                  |
|                  | sequence length                     | sl             |                                                  |
|                  | back propagation through time       | bptt           |                                                  |
