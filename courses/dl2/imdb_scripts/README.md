## Instructions

### 0. Preparing Wikipedia

If you want to train your own language model on a Wikipedia in your chosen language,
run `prepare_wiki.sh`. The script will ask for a language and will then
download, extract, and prepare the latest version of Wikipedia for the chosen language.

Example command: `bash prepare_wiki.sh`

This will create a `data` folder in this directory and `wiki_dumps`, `wiki_extr`, and
`wiki` subfolders. In each subfolder, it will furthermore create a folder `LANG`
where `LANG` is the language of the Wikipedia. The prepared files are stored in
`wiki/LANG` as `train.csv` and `val.csv` to match the format used for text
classification datasets. By default, `train.csv` contains around 100 million tokens
and `val.csv` is 10% the size of `train.csv`.

### 1. Tokenization

Run `create_toks.py` to tokenize the input texts.

Example command: `python create_toks.py imdb`

Usage:

```
create_toks.py DIR_PATH [CHUNKSIZE] [N_LBLS] [LANG]
create_toks.py --dir-path DIR_PATH [--chunksize CHUNKSIZE] [--n-lbls N_LBLS] [--lang LANG]
```

- `DIR_PATH`: the directory where your data is located
- `CHUNKSIZE`: the size of the chunks when reading the files with pandas; use smaller sizes with less RAM
- `LANG`: the language of your corpus. 

`train.csv` and `val.csv` files should be in `DIR_PATH`. The script will then save the
training and test tokens and labels as arrays to binary files in NumPy format in a `tmp`
in the above path in the following files:
`tok_trn.npy`, `tok_val.npy`, `lbl_trn.npy`, and `lbl_val.npy`.
In addition, a joined corpus containing white space-separated tokens is produced in `tmp/joined.txt`.

### 2. Mapping tokens to ids

Run `tok2id.py` to map the tokens in the `tok_trn.npy` and `tok_val.npy` files to ids.

Example command: `python tok2id.py imdb`

Usage:
```
tok2id.py PREFIX [MAX_VOCAB] [MIN_FREQ]
tok2id.py --prefix PREFIX [--max-vocab MAX_VOCAB] [--min-freq MIN_FREQ]
```
- `PREFIX`: the file path prefix in `data/nlp_clas/{prefix}`
- `MAX_VOCAB`: the maximum vocabulary size
- `MIN_FREQ`: the minimum frequency of words that should be kept

### 3. Fine-tune the LM

First run `train_tri_wt.py` to create a pre-trained language model using WikiText-103 (or whatever base corpus you prefer)
or use one of the pre-trained models.

Example command: `python train_tri_wt.py data/wiki/de/ 0 --lr 1e-3 --cl 12`

Usage:
```
train_tri_wt.py DIR_PATH CUDA_ID [CL] [BS] [BACKWARDS] [LR] [SAMPLED]
train_tri_wt.py --dir-path DIR_PATH --cuda-id CUDA_ID [--cl CL] [--bs BS] [--backwards BACKWARDS] [--lr LR] [--sampled SAMPLED]
```
- `DIR_PATH`: the directory that contains the Wikipedia files
- `CUDA_ID`: the id of the GPU that should be used; `-1` if no GPU should be used
- `CL`: the # of epochs to train
- `BS`: the batch size
- `BACKWARDS`: whether a backwards LM should be trained
- `LR`: the learning rate
- `SAMPLED`: whether a sampled softmax should be used (default: `True`)

Then run `train_tri_lm.py` to fine-tune a language model pretrained on WikiText-103 data on the target task data.

Example command: `python train_tri_lm.py imdb 0 5`

Usage:
```
train_tri_lm.py PREFIX CUDA_ID [NC] [BS] [BACKWARDS] [STARTAT]
train_tri_lm.py --prefix PREFIX --cuda-id CUDA_ID [--nc NC] [--bs BS] [--backwards BACKWARDS] [--startat STARTAT]
```
- `PREFIX`: the file path prefix in `data/nlp_clas/{prefix}`
- `CUDA_ID`: the id of the GPU used for training the model; `-1` if no GPU is used
- `NC`: number of cycles with all layers unfrozen
- `BS`: the batch size used for training the model
- `BACKWARDS`: whether to fine-tune a backwards language model (default is `False`)
- `STARTAT`: the id of the layer at which to start the gradual unfreezing (`1` is last hidden layer, etc.)

The language model is fine-tuned using warm-up reverse annealing and gradual unfreezing. For IMDb,
we set `--cl`, the number of epochs to `50` and used a learning rate `--lr` of `4e-3`.

### 4. Train the classifier

Run `train_clas.py` to train the classifier on top of the fine-tuned language model with gradual unfreezing and
discriminative fine-tuning.

Example command: `python train_clas.py imdb 0 --nc 4 --bs 48`

Usage:
```
train_clas.py PREFIX CUDA_ID [BS] [NC] [BACKWARDS] [STARTAT] [UNFREEZE] [PRETRAIN] [BPE]
train_clas.py --prefix PREFIX --cuda-id CUDA_ID [--bs BS] [--cl CL] [--backwards BACKWARDS] [--startat STARTAT] [--unfreeze UNFREEZE] [--pretrain PRETRAIN] [--bpe BPE]
```
- `BS`: the batch size used for training the model
- `CL`: the number of cycles with all layers unfrozen
- `BACKWARDS`: whether to fine-tune a backwards language model
- `STARTAT`: whether to use gradual unfreezing (`0`) or load the pretrained model (`1`)
- `UNFREEZE`: whether to unfreeze the whole network (after optional gradual unfreezing) or train only the final
              classifier layer (default is `True`)
- `PRETRAIN`: whether we use a pretrained model
- `BPE`: whether we use BPE

For fine-tuning the classifier on IMDb, we set `--cl`, the number of epochs to `50`. 

### 5. Evaluate the classifier

