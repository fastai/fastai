import numpy as np

from fastai.text import SortSampler, SortishSampler


def test_sort_sampler_sorts_all_descending():
    bs = 4
    n = bs*100
    data = 2 * np.arange(n)
    samp = list(SortSampler(data, lambda i: data[i]))

    # The sample is a permutation of the indices.
    assert sorted(samp) == list(range(n))
    # And that "permutation" is for descending data order.
    assert all(s1 > s2 for s1, s2 in zip(samp, samp[1:]))


def test_sortish_sampler_sorts_each_batch_descending():
    bs = 4
    n = bs*100
    data = 2 * np.arange(n)
    samp = list(SortishSampler(data, lambda i: data[i], bs))

    # The sample is a permutation of the indices.
    assert sorted(samp) == list(range(n))
    # And that permutation is kind of reverse sorted.
    assert all(
        s1 > s2 or (i+1) % bs == 0  # don't check batch boundaries
        for i, (s1, s2) in enumerate(zip(samp, samp[1:]))
    )
    assert samp[0] == max(samp)
