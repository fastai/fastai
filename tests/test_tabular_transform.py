from pandas.core.dtypes.dtypes import CategoricalDtype
import pytest
from fastai import *
from fastai.tabular import *

def test_categorify():
    cat_names = ['A']
    cont_names = ['X']
    train_df = pd.DataFrame({'A': ['a', 'b'],
                             'X': [0., 1.]})
    valid_df = pd.DataFrame({'A': ['b', 'a'],
                             'X': [1., 0.]})
    original_cont_train_df = train_df[cont_names].copy()
    original_cont_valid_df = valid_df[cont_names].copy()
    
    categorify_transform = Categorify(cat_names, cont_names)
    categorify_transform.apply_train(train_df)
    categorify_transform.apply_test(valid_df)

    # Make sure all categorical columns have the right type
    assert all([isinstance(dt, CategoricalDtype) for dt
                in train_df[cat_names].dtypes])
    assert all([isinstance(dt, CategoricalDtype) for dt
                in valid_df[cat_names].dtypes])
    # Make sure continuous columns have not changed
    assert train_df[cont_names].equals(original_cont_train_df)
    assert valid_df[cont_names].equals(original_cont_valid_df)

def test_default_fill_strategy_is_median():
    fill_missing_transform = FillMissing([], [])

    assert fill_missing_transform.fill_strategy is FillStrategy.MEDIAN

def test_fill_missing_leaves_no_na_values():
    cont_names = ['A']
    train_df = pd.DataFrame({'A': [0., np.nan, np.nan]})
    valid_df = pd.DataFrame({'A': [np.nan, 0., np.nan]})
    
    fill_missing_transform = FillMissing([], cont_names)
    fill_missing_transform.apply_train(train_df)
    fill_missing_transform.apply_test(valid_df)

    assert train_df.isna().values.sum() == 0
    assert valid_df.isna().values.sum() == 0

def test_fill_missing_returns_correct_medians():
    train_df = pd.DataFrame({'A': [0., 1., 1., np.nan, np.nan]})
    valid_df = pd.DataFrame({'A': [0., 0., 1., np.nan, np.nan]})
    expected_filled_train_df = pd.DataFrame({'A': [0., 1., 1., 1., 1.]})
    expected_filled_valid_df = pd.DataFrame({'A': [0., 0., 1., 1., 1.]})

    fill_missing_transform = FillMissing([], ['A'], add_col=False)
    fill_missing_transform.apply_train(train_df)
    fill_missing_transform.apply_test(valid_df)

    # Make sure the train median is used in both cases
    assert train_df.equals(expected_filled_train_df)
    assert valid_df.equals(expected_filled_valid_df)