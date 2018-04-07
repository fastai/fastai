## Fastai Abbreviation Guide
As mentioned in the [fastai style](https://github.com/fastai/fastai/blob/master/docs/style.md), 
we name symbols following the *Huffman Coding* principle, which basically means 

> Commonly used and generic concepts should be named shorter. You shouldn't waste short sequences on less common concepts. 

Fastai also follows the life-cycle naming principle:
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
|                  | transform image using opencv        | _cv            | zoom_cv(), rotate_cv(), stretch_cv()             |
| **Prefix**       |                                     |                |                                                  |
|                  | check if satisfied                  | is_            | is_reg, is_multi, is_single, is_test, is_correct |
|                  | On/off a feature                    | use_           | use_bn                                           |
|                  | Number of something (plural)        | n_             | n_embs, n_factors, n_users, n_items              |
|                  | count something                     | num_           | num_features(), num_gpus()                       |
|                  | convert to something                | to_            | to_gpu(), to_cpu(), to_np()                      |
| **Inflix**       |                                     |                |                                                  |
|                  | Convert between concepts            | 2              | name2idx(), label2idx(), seq2seq                 |
| **Aggressive**   |                                     |                |                                                  |
|                  | function                            | f              |                                                  |
|                  | input                               | x              |                                                  |
|                  | key                                 | k              |                                                  |
|                  | value                               | v              |                                                  |
|                  | index                               | i              |                                                  |
|                  | object                              | o              |                                                  |
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
|                  | class                               | cls            |                                                  |
|                  | source                              | src            |                                                  |
|                  | destination                         | dst            |                                                  |
|                  | directory                           | dir            |                                                  |
|                  | percentage                          | p              |                                                  |
|                  | ratio, proportion of something      | r              |                                                  |
|                  |                                     |                |                                                  |
|                  | figure                              | fig            |                                                  |
|                  | image                               | im             |                                                  |
|                  | configuration                       | cfg            |                                                  |
|                  | random                              | rand           |                                                  |
|                  | numpy                               | np             |                                                  |
|                  | pandas                              | pd             |                                                  |
|                  | utility                             | util           |                                                  |
|                  |                                     |                |                                                  |
|                  | temporary                           | tmp            | tmp_name, tmp_path                               |
|                  | extra                               | xtra           |                                                  |
|                  | tiny number                         | eps            |                                                  |
|                  | threshold                           | thresh         |                                                  |
|                  |                                     |                |                                                  |
|                  | less than                           | lt             |                                                  |
|                  | less or equal                       | le             |                                                  |
|                  | equal                               | eq             |                                                  |
|                  | not equal                           | ne             |                                                  |
|                  | greater than                        | gt             |                                                  |
|                  | greater or equal                    | ge             |                                                  |
| **Data**         |                                     |                |                                                  |
|                  | number of elements                  | n              |                                                  |
|                  | length                              | len            |                                                  |
|                  | size                                | sz             |                                                  |
|                  | array                               | arr            | label_arr                                        |
|                  | dictionary                          | dict           |                                                  |
|                  | sequence                            | seq            |                                                  |
|                  |                                     |                |                                                  |
|                  | dataset                             | ds             |                                                  |
|                  | dataloader                          | dl             |                                                  |
|                  | dataframe                           | df             |                                                  |
|                  | train                               | trn            | trn_ds                                           |
|                  | validation                          | val            | val_ds                                           |
|                  | test                                | test           | test_ds                                          |
|                  | number of classes                   | c              |                                                  |
|                  | batch size                          | bs             |                                                  |
|                  | multiple targets                    | multi          | is_multi                                         |
|                  | regression                          | reg            | is_reg                                           |
|                  | iterate, iterator                   | iter           | trn_iter, val_iter                               |
|                  |                                     |                |                                                  |
|                  | multiple inputs                     | xs             |                                                  |
|                  | target                              | y              |                                                  |
|                  | prediction                          | pred           |                                                  |
|                  | output                              | out            |                                                  |
|                  | column                              | col            |                                                  |
|                  | continuous                          | cont           | conts, cont_cols                                 |
|                  | category                            | cat            | cats, cat_cols                                   |
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
|                  |                                     |                |                                                  |
| **Modeling**     |                                     |                |                                                  |
|                  | initialize                          | init           |                                                  |
|                  | natural language processing         | nlp            |                                                  |
|                  | language model                      | lm             |                                                  |
|                  | recurrent neural network            | rnn            |                                                  |
|                  | convolutional neural network        | convnet        |                                                  |
|                  |                                     |                |                                                  |
|                  | model                               | m              | (not common)                                     |
|                  | model data                          | md             |                                                  |
|                  | linear                              | lin            |                                                  |
|                  | embedding                           | emb            |                                                  |
|                  | batch norm                          | bn             |                                                  |
|                  | dropout                             | drop           |                                                  |
|                  | fully connected                     | fc             |                                                  |
|                  | convolution                         | conv           |                                                  |
|                  | hidden                              | hid            |                                                  |
|                  |                                     |                |                                                  |
|                  | optimizer (usually e.g. Adam)       | opt            |                                                  |
|                  | layer group learning rate optimizer | layer_opt      |                                                  |
|                  | criteria                            | crit           |                                                  |
|                  | weight decay                        | wd             |                                                  |
|                  | cross validation                    | cv             |                                                  |
|                  | learning rate                       | lr             |                                                  |
|                  | schedule                            | sched          |                                                  |
|                  | multiplier                          | mult           |                                                  |
|                  | activation                          | actn           |                                                  |
|                  |                                     |                |                                                  |
| **Text**         | string                              | s              |                                                  |
