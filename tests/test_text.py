import pytest
from fastai import *
from fastai.text import *

def test_rules():
    assert fix_html("Some HTML&nbsp;text<br />") == "Some HTML& text\n"
    assert replace_rep("I'm so excited!!!!!!!!") == "I'm so excited xxrep 8 ! "
    assert replace_wrep("I've never ever ever ever ever ever ever ever done this.") == "I've never  xxwrep 7 ever  done this."
    assert rm_useless_spaces("Inconsistent   use  of     spaces.") == "Inconsistent use of spaces."
    assert spec_add_spaces('I #like to #put #hashtags #everywhere!') == "I  # like to  # put  # hashtags  # everywhere!"

def test_tokenize():
    texts = ['one two three four', 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.', "I'm suddenly SHOUTING FOR NO REASON"]
    tokenizer = Tokenizer(BaseTokenizer)
    toks = tokenizer.process_all(texts)
    assert toks[0] == ['one', 'two', 'three', 'four']
    assert toks[1][:6] == ['xxmaj', 'lorem', 'ipsum', 'dolor', 'sit', 'amet,']
    assert ' '.join(toks[2]) == "xxmaj i'm suddenly xxup shouting xxup for xxup no xxup reason"

    