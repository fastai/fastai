from fastai import *
from fastai.text import *

def test_should_load_backwards_lm():
    # GIVEN
    df = pd.DataFrame([{0: 0, "text": "fast ai is a cool project"}, {0: 0, 'text': "hello world"}])
    text_ds = TextDataset.from_df('/tmp/', df, label_cols=[0], txt_cols=["text"], min_freq=0, tokenizer=Tokenizer(BaseTokenizer))
    # WHEN
    lml = LanguageModelLoader(text_ds, bs=1, backwards=True)
    # THEN
    batch = lml.get_batch(0, 70)

    as_text = [text_ds.vocab.itos[x] for x in batch[0]]
    np.testing.assert_array_equal(as_text[:2], ["world", "hello"])
