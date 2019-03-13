import pytest
from fastai.gen_doc.doctest import this_tests
from fastai.text import *

def getAllBatches(data,epochs=1):
    x=None
    for i in range(epochs):
        data.on_epoch_begin()
        countIte=0
        for xb,yb in data:
            countIte += 1
            d= xb.data.numpy()
            if x is None:
                x = xb.data.numpy().copy()
            else:
                x = np.concatenate((x, xb.data.numpy().copy()),axis=1)
            continue
        data.on_epoch_end()
    return x,countIte

def jaggedWithConsecutiveNumbers(bs,sentence_len,iterations,minTokens):
    "create jagged array with random layout and filled with consequetive numbers"
    jagged = []
    count = 0
    total = bs*sentence_len*iterations
    while count < total:
        nb = total-count if total-count<sentence_len else minTokens+int(np.random.random() * sentence_len)
        jagged.append(np.arange(count+1,count+1+nb))
        count = jagged[-1][-1]
    jagged = np.asarray(jagged)
    return jagged, count

def verify_datadirection( bs,seq_len,sentence_len, iterations,minTokens, backwards=False, nbTests=1000):
    for i in range(nbTests):
        jagged,countTokens = jaggedWithConsecutiveNumbers(bs,sentence_len,iterations,minTokens)

        trainIDS = validIDS = jagged
        db   = TextLMDataBunch.from_ids( ".", None, trainIDS, validIDS, bptt=seq_len, bs=bs,no_check=True)
        data = LanguageModelPreLoader(db.train_ds, bs=bs, bptt=seq_len, backwards=backwards, shuffle=False)
        dl   = DataLoader(data, bs, shuffle=False)
        batches, countIte = getAllBatches(dl)

        assert countIte==len(dl), f"number of iteration does not match: countIte:{countIte}!= len(data):{len(dl)} "

        #The diff from one to the next column must be 1 for aligned mini-batches with forward indexing of the data
        #(forward is default for LanguageModelLoader ie.: backwards=False)
        b_diff = batches[:,1:] - batches[:,0:-1]
        diff = -1 if backwards else 1
        assert (b_diff.flatten()==diff).all(), "the sequences of batch rows are not contiguous"

        ix = np.arange(1,len(batches))
        assert np.all(batches[ix-1,-1]+diff == batches[ix,0]), f"last token i row-1 {batches[ix-1,-1]}+{diff} must be equal to first element in row:{batches[ix,0]}"

def test_forward_minibatch():
    this_tests('na')
    bs           = 4
    seq_len      = 3
    sentence_len = 20*seq_len
    iterations   = 2
    minTokens    = 1
    verify_datadirection( bs, seq_len, sentence_len, iterations, minTokens, backwards=False, nbTests=1000)

def test_backwards_minibatch():
    this_tests('na')
    bs           = 4
    seq_len      = 3
    sentence_len = 20*seq_len
    iterations   = 2
    minTokens    = 1
    verify_datadirection( bs, seq_len, sentence_len, iterations, minTokens, backwards=True, nbTests=1000)
