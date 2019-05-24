import pytest
from fastai.gen_doc.doctest import this_tests
from fastai.vision import *

def test_rle_encode_with_array():
    this_tests(rle_encode)
    fake_img = np.array([[0, 0, 1], [0, 1, 0], [1, 0 ,0]])
    answer = '3 1 5 1 7 1'
    assert rle_encode(fake_img) == answer

def test_rle_encode_all_zero_array():
    this_tests(rle_encode)
    fake_img = np.array([[0, 0, 0], [0, 0, 0], [0, 0 ,0]])
    answer = ''
    assert rle_encode(fake_img) == answer

def test_rle_decode_with_str():
    this_tests(rle_decode)
    encoded_str = '3 1 5 1 7 1'
    ans = np.array([[0, 0, 1], [0, 1, 0], [1, 0 ,0]])
    assert np.alltrue(rle_decode(encoded_str,(3,3)) == ans)

def test_rle_decode_empty_str():
    this_tests(rle_decode)
    encoded_str = ''
    ans = np.array([[0, 0, 0], [0, 0, 0], [0, 0 ,0]])
    assert np.alltrue(rle_decode(encoded_str,(3,3)) == ans)

def test_tis2hw_int():
    this_tests(tis2hw)
    size = 224
    assert(tis2hw(size) == [224,224])

def test_tis2hw_3dims():
    this_tests(tis2hw)
    size = (3, 224, 224)
    assert(tis2hw(size) == [224,224])

def test_tis2hw_2dims():
    this_tests(tis2hw)
    size = (224, 224)
    assert(tis2hw(size) == [224,224])

def test_tis2hw_str_raises_an_error():
    this_tests(tis2hw)
    with pytest.raises(RuntimeError) as e:
        tis2hw("224")

def test_image_resize_same_size_shortcut():
    this_tests(Image.resize)
    px = torch.Tensor([[[1, 2,], [3, 4]]])
    image = Image(px)
    old_size = image.size
    image = image.resize(px.size()) 
    assert(image is not None and (old_size == image.size))
