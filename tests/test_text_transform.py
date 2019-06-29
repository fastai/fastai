import pytest
from fastai.gen_doc.doctest import this_tests
from fastai.text import *

def test_rules():
    this_tests(fix_html, replace_rep, replace_wrep, rm_useless_spaces, spec_add_spaces, replace_all_caps, deal_caps)
    assert fix_html("Some HTML&nbsp;text<br />") == "Some HTML& text\n"
    assert replace_rep("I'm so excited!!!!!!!!") == "I'm so excited xxrep 8 ! "
    assert replace_wrep("I've never ever ever ever ever ever ever ever done this.") == "I've never  xxwrep 7 ever  done this."
    assert rm_useless_spaces("Inconsistent   use  of     spaces.") == "Inconsistent use of spaces."
    assert spec_add_spaces('I #like to #put #hashtags #everywhere!') == "I  # like to  # put  # hashtags  # everywhere!"
    assert replace_all_caps(['Mark','CAPITALIZED','Only']) == ['Mark', 'xxup', 'capitalized', 'Only']
    assert deal_caps(['Mark','Capitalized','lower', 'All']) == ['xxmaj', 'mark', 'xxmaj', 'capitalized', 'lower', 'xxmaj', 'all']

def test_tokenize():
    this_tests(Tokenizer)
    texts = ['one two three four', 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.', "I'm suddenly SHOUTING FOR NO REASON"]
    tokenizer = Tokenizer(BaseTokenizer)
    toks = tokenizer.process_all(texts)
    assert toks[0] == ['one', 'two', 'three', 'four']
    assert toks[1][:6] == ['xxmaj', 'lorem', 'ipsum', 'dolor', 'sit', 'amet,']
    assert ' '.join(toks[2]) == "xxmaj i'm suddenly xxup shouting xxup for xxup no xxup reason"

def test_tokenize_handles_empty_lines():
    this_tests(Tokenizer)
    texts = ['= Markdown Title =\n\nMakrdown Title does not have spaces around']
    tokenizer = Tokenizer(BaseTokenizer)
    toks = tokenizer.process_all(texts)
    assert toks[0] == ['=', 'xxmaj', 'markdown', 'xxmaj', 'title', '=', '\n', '\n',
                       'xxmaj', 'makrdown', 'xxmaj', 'title', 'does', 'not', 'have', 'spaces', 'around']

def test_tokenize_ignores_extraneous_space():
    this_tests(Tokenizer)
    texts = ['test ']
    tokenizer = Tokenizer(BaseTokenizer)
    toks = tokenizer.process_all(texts)
    assert toks[0] == ['test']

def test_numericalize_and_textify():
    toks = [['ok', '!', 'xxmaj', 'nice', '!', 'anti', '-', 'virus'], ['!', 'xxmaj', 'meg', 'xxmaj', 'nice', 'meg']]
    vocab = Vocab.create(toks, max_vocab=20, min_freq=2)
    this_tests(vocab.numericalize, vocab.textify)
    assert vocab.numericalize(toks[0]) == [0, 9, 5, 10, 9, 0, 0, 0]
    assert vocab.textify([0, 3, 10, 11, 9]) == 'xxunk xxeos nice meg !'
    print(vocab.stoi)
    with pytest.raises(IndexError):
        vocab.textify([16])
