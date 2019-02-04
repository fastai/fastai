import pytest
from fastai.vision import *

def test_rle_encode_with_array():
    fake_img = np.array([[0, 0, 1], [0, 1, 0], [1, 0 ,0]])
    answer = '3 1 5 1 7 1'
    assert rle_encode(fake_img) == answer

def test_rle_encode_all_zero_array():
    fake_img = np.array([[0, 0, 0], [0, 0, 0], [0, 0 ,0]])
    answer = ''
    assert rle_encode(fake_img) == answer

def test_rle_decode_with_str():
    encoded_str = '3 1 5 1 7 1'
    ans = np.array([[0, 0, 1], [0, 1, 0], [1, 0 ,0]])
    assert np.alltrue(rle_decode(encoded_str,(3,3)) == ans)

def test_rle_decode_empty_str():
    encoded_str = ''
    ans = np.array([[0, 0, 0], [0, 0, 0], [0, 0 ,0]])
    assert np.alltrue(rle_decode(encoded_str,(3,3)) == ans)

def test_tis2hw_int():
    size = 224
    assert(tis2hw(size) == [224,224])

def test_tis2hw_3dims():
    size = (3, 224, 224)
    assert(tis2hw(size) == [224,224])

def test_tis2hw_2dims():
    size = (224, 224)
    assert(tis2hw(size) == [224,224])

def test_tis2hw_str_raises_an_erorr():
    with pytest.raises(RuntimeError) as e:
        tis2hw("224")
