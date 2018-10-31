from pandas.core.dtypes.dtypes import CategoricalDtype
import pytest
from fastai import *
from fastai.tabular import *

def test_categorify():
    path = untar_data(URLs.ADULT_SAMPLE)
    df = pd.read_csv(path / 'adult.csv')
    cat_names = ['workclass', 'education', 'marital-status', 'occupation',
                 'relationship', 'race', 'sex', 'native-country']
    cont_names = [column_name for column_name in df.columns
                     if column_name not in cat_names]
    train_df, valid_df = df[:1024].copy(),df[1024:1260].copy()
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
    path = untar_data(URLs.ADULT_SAMPLE)
    df = pd.read_csv(path / 'adult.csv')
    cont_names = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                  'capital-loss', 'hours-per-week']
    df = df[cont_names]
    train_df, valid_df = df[:1024].copy(),df[1024:1260].copy()
    
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