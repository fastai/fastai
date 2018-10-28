from pandas.core.dtypes.dtypes import CategoricalDtype
import pytest
from fastai import *
from fastai.tabular import *

def test_categorify():
    path = untar_data(URLs.ADULT_SAMPLE)
    df = pd.read_csv(path / 'adult.csv')
    cat_names = ['workclass', 'education', 'marital-status', 'occupation',
                 'relationship', 'race', 'sex', 'native-country']
    df = df[cat_names]
    train_df, valid_df = df[:1024].copy(),df[1024:1260].copy()
    categorify_transform = Categorify(cat_names, [])

    categorify_transform.apply_train(train_df)
    categorify_transform.apply_test(valid_df)

    assert all([isinstance(dt, CategoricalDtype) for dt in train_df.dtypes])
    assert all([isinstance(dt, CategoricalDtype) for dt in valid_df.dtypes])

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

    train_df.isna().values.sum() == 0
    valid_df.isna().values.sum() == 0

def test_fill_missing_returns_correct_median():
    path = untar_data(URLs.ADULT_SAMPLE)
    df = pd.read_csv(path / 'adult.csv')
    cont_names = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                  'capital-loss', 'hours-per-week']
    df = df[cont_names]
    train_df, valid_df = df[:1024].copy(),df[1024:1260].copy()
    expected_train_median = train_df.median()
    expected_valid_median = valid_df.median()
    fill_missing_transform = FillMissing([], cont_names, add_col=False)

    fill_missing_transform.apply_train(train_df)
    fill_missing_transform.apply_test(valid_df)
    train_median = train_df.median()
    valid_median = valid_df.median()

    assert np.array_equal(train_median.values, expected_train_median.values)
    assert np.array_equal(valid_median.values, expected_valid_median.values)
