## Instructions

### 0. Preparing Wikipedia

If you want to train your own language model on a Wikipedia in your chosen language,
run `prepare_wiki.sh`. The script will ask for a language and will then
download, extract, and prepare the latest version of Wikipedia for the chosen language.
Note that for English (due to the size of the English Wikipedia), the extraction process
takes quite long.

Example command: `bash prepare_wiki.sh`

This will create a `data` folder in this directory and `wiki_dumps`, `wiki_extr`, and
`wiki` subfolders. In each subfolder, it will furthermore create a folder `LANG`
where `LANG` is the language of the Wikipedia. The prepared files are stored in
`wiki/LANG` as `train.csv` and `val.csv` to match the format used for text
classification datasets. By default, `train.csv` contains around 100 million tokens
and `val.csv` is 10% the size of `train.csv`.

### 1. Tokenization

Run `create_toks.py` to tokenize the input texts.

Example command: `python create_toks.py data/imdb`

Usage:

```
create_toks.py DIR_PATH [CHUNKSIZE] [N_LBLS] [LANG]
create_toks.py --dir-path DIR_PATH [--chunksize CHUNKSIZE] [--n-lbls N_LBLS] [--lang LANG]
```

- `DIR_PATH`: the directory where your data is located
- `CHUNKSIZE`: the size of the chunks when reading the files with pandas; use smaller sizes with less RAM
- `LANG`: the language of your corpus.

The script expects `train.csv` and `val.csv` files to be in `DIR_PATH`. Each file should be in
CSV format. If the data is labeled, the first column should consist of the label as an integer.
The remaining columns should consist of text or features, which will be concatenated to form
each example. If the data is unlabeled, the file should just consist of a single text column.
The script will then save the training and test tokens and labels as arrays to binary files in NumPy format
in a `tmp` in the above path in the following files:
`tok_trn.npy`, `tok_val.npy`, `lbl_trn.npy`, and `lbl_val.npy`.
In addition, a joined corpus containing white space-separated tokens is produced in `tmp/joined.txt`.

### 2. Mapping tokens to ids

Run `tok2id.py` to map the tokens in the `tok_trn.npy` and `tok_val.npy` files to ids.

Example command: `python tok2id.py data/imdb`

Usage:
```
tok2id.py PREFIX [MAX_VOCAB] [MIN_FREQ]
tok2id.py --prefix PREFIX [--max-vocab MAX_VOCAB] [--min-freq MIN_FREQ]
```
- `PREFIX`: the file path prefix in `data/nlp_clas/{prefix}`
- `MAX_VOCAB`: the maximum vocabulary size
- `MIN_FREQ`: the minimum frequency of words that should be kept

### (3a. Pretrain the Wikipedia language model)

Before fine-tuning the language model, you can run `pretrain_lm.py` to create a
pre-trained language model using WikiText-103 (or whatever base corpus you prefer).

Example command: `python pretrain_lm.py data/wiki/de/ 0 --lr 1e-3 --cl 12`

Usage:
```
pretrain_lm.py DIR_PATH CUDA_ID [CL] [BS] [BACKWARDS] [LR] [SAMPLED] [PRETRAIN_ID]
pretrain_lm.py --dir-path DIR_PATH --cuda-id CUDA_ID [--cl CL] [--bs BS] [--backwards BACKWARDS] [--lr LR] [--sampled SAMPLED] [--pretrain-id PRETRAIN_ID]
```
- `DIR_PATH`: the directory that contains the Wikipedia files
- `CUDA_ID`: the id of the GPU that should be used;
- `CL`: the # of epochs to train
- `BS`: the batch size
- `BACKWARDS`: whether a backwards LM should be trained
- `LR`: the learning rate
- `SAMPLED`: whether a sampled softmax should be used (default: `True`)
- `PRETRAIN_ID`: the id used for saving the trained LM

You might have to adapt the learning rate and the # of epochs to maximize performance.

### 3b. Fine-tune the LM

Alternatively, you can download the pre-trained models [here](http://files.fast.ai/models/wt103/). Before,
create a directory `wt103`. In `wt103`, create a `models` and a `tmp` folder. Save the model files
in the `models` folder and `itos_wt103.pkl`, the word-to-token mapping, to the `tmp` folder.

Then run `finetune_lm.py` to fine-tune a language model pretrained on WikiText-103 data on the target task data.

Example command: `python finetune_lm.py data/imdb data/wt103 1 25 --lm-id pretrain_wt103`

Usage:
```
finetune_lm.py DIR_PATH PRETRAIN_PATH [CUDA_ID] [CL] [PRETRAIN_ID] [LM_ID] [BS] [DROPMULT] [BACKWARDS] [LR] [PRELOAD] [BPE] [STARTAT] [USE_CLR] [USE_REGULAR_SCHEDULE] [USE_DISCRIMINATIVE] [NOTRAIN] [JOINED] [TRAIN_FILE_ID] [EARLY_STOPPING]
finetune_lm.py --dir-path DIR_PATH --pretrain-path PRETRAIN_PATH [--cuda-id CUDA_ID] [--cl CL] [--pretrain-id PRETRAIN_ID] [--lm-id LM_ID] [--bs BS] [--dropmult DROPMULT] [--backwards BACKWARDS] [--lr LR] [--preload PRELOAD] [--bpe BPE] [--startat STARTAT] [--use-clr USE_CLR] [--use-regular-schedule USE_REGULAR_SCHEDULE] [--use-discriminative USE_DISCRIMINATIVE] [--notrain NOTRAIN] [--joined JOINED] [--train-file-id TRAIN_FILE_ID] [--early-stopping EARLY_STOPPING]
```
- `DIR_PATH`: the directory where the `tmp` and `models` folder are located
- `PRETRAIN_PATH`: the path where the pretrained model is saved; if using the downloaded model, this is `wt103`
- `CUDA_ID`: the id of the GPU used for training the model
- `CL`: number of epochs to train the model
- `PRETRAIN_ID`: the id of the pretrained model; set to `wt103` per default
- `LM_ID`: the id used for saving the fine-tuned language model
- `BS`: the batch size used for training the model
- `DROPMULT`: the factor used to multiply the dropout parameters
- `BACKWARDS`: whether a backwards LM is trained
- `LR`: the learning rate
- `PRELOAD`: whether we load a pretrained LM (`True` by default)
- `BPE`: whether we use byte-pair encoding (BPE)
- `STARTAT`: can be used to continue fine-tuning a model; if `>0`, loads an already fine-tuned LM; can also be used to indicate the layer at which to start the gradual unfreezing (`1` is last hidden layer, etc.); in the final model, we only used this for training the classifier
- `USE_CLR`: whether to use slanted triangular learning rates (STLR) (`True` by default)
- `USE_REGULAR_SCHEDULE`: whether to use a regular schedule (instead of STLR)
- `USE_DISCRIMINATIVE`: whether to use discriminative fine-tuning (`True` by default)
- `NOTRAIN`: whether to skip fine-tuning
- `JOINED`: whether to fine-tune the LM on the concatenation of training and validation data
- `TRAIN_FILE_ID`: can be used to indicate different training files (e.g. to test training sizes)
- `EARLY_STOPPING`: whether to use early stopping

The language model is fine-tuned using warm-up reverse annealing and triangular learning rates. For IMDb,
we set `--cl`, the number of epochs to `50` and used a learning rate `--lr` of `4e-3`.

### 4. Train the classifier

Run `train_clas.py` to train the classifier on top of the fine-tuned language model with gradual unfreezing,
discriminative fine-tuning, and slanted triangular learning rates.

Example command: `python train_clas.py data/imdb 0 --lm-id pretrain_wt103 --clas-id pretrain_wt103 --cl 50`

Usage:
```
train_clas.py DIR_PATH CUDA_ID [LM_ID] [CLAS_ID] [BS] [CL] [BACKWARDS] [STARTAT] [UNFREEZE] [LR] [DROPMULT] [BPE] [USE_CLR] [USE_REGULAR_SCHEDULE] [USE_DISCRIMINATIVE] [LAST] [CHAIN_THAW] [FROM_SCRATCH] [TRAIN_FILE_ID]
train_clas.py --dir-path DIR_PATH --cuda-id CUDA_ID [--lm-id LM_ID] [--clas-id CLAS_ID] [--bs BS] [--cl CL] [--backwards BACKWARDS] [--startat STARTAT] [--unfreeze UNFREEZE] [--lr LR] [--dropmult DROPMULT] [--bpe BPE] [--use-clr USE_CLR] [--use-regular-schedule USE_REGULAR_SCHEDULE] [--use-discriminative USE_DISCRIMINATIVE] [--last LAST] [--chain-thaw CHAIN_THAW] [--from-scratch FROM_SCRATCH] [--train-file-id TRAIN_FILE_ID]
```
- `DIR_PATH`: the directory where the `tmp` and `models` folder are located
- `CUDA_ID`: the id of the GPU used for training the model
- `LM_ID`: the id of the fine-tuned language model that should be loaded
- `CLAS_ID`: the id used for saving the classifier
- `BS`: the batch size used for training the model
- `CL`: the number of epochs to train the model with all layers unfrozen
- `BACKWARDS`: whether a backwards LM is trained
- `STARTAT`: whether to use gradual unfreezing (`0`) or load the pretrained model (`1`)
- `UNFREEZE`: whether to unfreeze the whole network (after optional gradual unfreezing) or train only the final classifier layer (default is `True`)
- `LR`: the learning rate
- `DROPMULT`: the factor used to multiply the dropout parameters
- `BPE`: whether we use byte-pair encoding (BPE)
- `USE_CLR`: whether to use slanted triangular learning rates (STLR) (`True` by default)
- `USE_REGULAR_SCHEDULE`: whether to use a regular schedule (instead of STLR)
- `USE_DISCRIMINATIVE`: whether to use discriminative fine-tuning (`True` by default)
- `LAST`: whether to fine-tune only the last layer of the model
- `CHAIN_THAW`: whether to use chain-thaw
- `FROM_SCRATCH`: whether to train the model from scratch (without loading a pretrained model)
- `TRAIN_FILE_ID`: can be used to indicate different training files (e.g. to test training sizes)

For fine-tuning the classifier on IMDb, we set `--cl`, the number of epochs to `50`. `--lr .005`, works well if `--backwards` is set to True, otherwise the default .01 is fine.

### 5. Evaluate the classifier

Run `eval_clas.py` to get the classifier accuracy and confusion matrix.

This requires the files produced during the training process: itos.pkl and the classifier (named clas_1.h5 by default), as well as the `npy` files containing the evaluation samples and labels. 

Example command: `python eval_clas.py data/imdb 0 --lm-id pretrain_wt103 --clas-id pretrain_wt103`

Usage:
```
eval_clas.py DIR_PATH CUDA_ID [LM_ID] [CLAS_ID] [BS] [BACKWARDS] [BPE]
eval_clas.py --dir-path DIR_PATH --cuda-id CUDA_ID [--lm-id LM_ID] [--clas-id CLAS_ID] [--bs BS] [--bpe BPE]
```
- `DIR_PATH`: the directory where the `tmp` and `models` folder are located
- `CUDA_ID`: the id of the GPU used for training the model
- `LM_ID`: the id of the fine-tuned language model that should be loaded
- `CLAS_ID`: the id used for saving the classifier
- `BS`: the batch size used for training the model
- `BACKWARDS`: whether a backwards LM is trained
- `BPE`: whether we use byte-pair encoding (BPE)

### 6. Try the classifier on text

Run `predict_with_classifier.py` to predict against free text entry.

This requires two files produced during the training process: the id-to-token mapping `itos.pkl` and the classifier (named `clas_1.h5` by default)

Example command: `python predict_with_classifier.py trained_models/itos.pkl trained_models/classifier_model.h5`

It is suggested to customize this script to your needs.
