## Instructions

### 1. Tokenization

Run `create_toks.py` to tokenize the input texts.

Example command: `python create_toks.py imdb imdb`

Usage:

```
create_toks.py PREFIX PR_ABBR [CHUNKSIZE]
create_toks.py --prefix PREFIX --pr-abbr PR_ABBR [--chunksize CHUNKSIZE]
```

- `PREFIX`: the file path prefix in `data/nlp_clas/{prefix}`
- `PR_ABBR`: the file path abbreviation used for designating the joined corpus
- `CHUNKSIZE`: the size of the chunks when reading the files with pandas; use smaller sizes with less RAM

`train.csv` and `test.csv` files should be in `data/nlp_clas/{prefix}`. The script will then save the
training and test tokens and labels as arrays to binary files in NumPy format in a `tmp`
in the above path in the following files:
`tok_trn.npy`, `tok_val.npy`, `lbl_trn.npy`, and `lbl_val.npy`.
In addition, a joined corpus containing white space-separated tokens is produced in `tmp/{pr_abbr}_joined.txt`.

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

First run `train_tri_wt.py` to create a pre-trained language model using WikiText-103 (or whatever base corpus you prefer).

Then run `train_lm.py` to fine-tune a language model pretrained on WikiText-103 data on the target task data.

Example command: `python train_lm.py imdb 0 5`

Usage:
```
train_lm.py PREFIX CUDA_ID [NC] [BS] [BACKWARDS] [STARTAT]
train_lm.py --prefix PREFIX --cuda-id CUDA_ID [--nc NC] [--bs BS] [--backwards BACKWARDS] [--startat STARTAT]
```
- `PREFIX`: the file path prefix in `data/nlp_clas/{prefix}`
- `CUDA_ID`: the id of the GPU used for training the model; `-1` if no GPU is used
- `NC`: number of cycles with all layers unfrozen
- `BS`: the batch size used for training the model
- `BACKWARDS`: whether to fine-tune a backwards language model (default is `False`)
- `STARTAT`: the id of the layer at which to start the gradual unfreezing (`1` is last hidden layer, etc.)

The language model is fine-tuned using warm-up reverse annealing and gradual unfreezing.

### 4. Train the classifier

Run `train_clas.py` to train the classifier on top of the fine-tuned language model with gradual unfreezing and
discriminative fine-tuning.

Example command: `python train_clas.py imdb 0 --nc 4 --bs 48`

Usage:
```
train_clas.py PREFIX CUDA_ID [BS] [NC] [BACKWARDS] [STARTAT] [UNFREEZE] [PRETRAIN] [BPE]
train_clas.py --prefix PREFIX --cuda-id CUDA_ID [--bs BS] [--nc NC] [--backwards BACKWARDS] [--startat STARTAT] [--unfreeze UNFREEZE] [--pretrain PRETRAIN] [--bpe BPE]
```
- `BS`: the batch size used for training the model
- `NC`: the number of cycles with all layers unfrozen
- `BACKWARDS`: whether to fine-tune a backwards language model
- `STARTAT`: whether to use gradual unfreezing (`0`) or load the pretrained model (`1`)
- `UNFREEZE`: whether to unfreeze the whole network (after optional gradual unfreezing) or train only the final
              classifier layer (default is `True`)
- `PRETRAIN`: whether we use a pretrained model
- `BPE`: whether we use BPE

### 5. Evaluate the classifier

