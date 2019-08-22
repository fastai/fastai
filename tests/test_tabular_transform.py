from fastai.gen_doc.doctest import this_tests
from fastai.tabular import *
from pandas.core.dtypes.dtypes import CategoricalDtype


def test_categorify():
    this_tests(Categorify)
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
    this_tests(FillMissing)


def test_fill_missing_leaves_no_na_values():
    cont_names = ['A']
    train_df = pd.DataFrame({'A': [0., np.nan, np.nan]})
    valid_df = pd.DataFrame({'A': [np.nan, 0., np.nan]})

    fill_missing_transform = FillMissing([], cont_names)
    fill_missing_transform.apply_train(train_df)
    fill_missing_transform.apply_test(valid_df)
    this_tests(fill_missing_transform.apply_train, fill_missing_transform.apply_test)

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
    this_tests(fill_missing_transform.apply_train, fill_missing_transform.apply_test)

    # Make sure the train median is used in both cases
    assert train_df.equals(expected_filled_train_df)
    assert valid_df.equals(expected_filled_valid_df)


def test_cont_cat_split():
    this_tests(cont_cat_split)
    cat_names = ['A']
    cont_names = ['F', 'I']
    df = pd.DataFrame({'A': ['a', 'b'], 'F': [0., 1.], 'I': [0, 1]})

    # Make sure the list and number of columns is correct
    res_cont, res_cat = cont_cat_split(df, max_card=0)
    assert res_cont == cont_names
    assert res_cat == cat_names
    assert len(res_cont) == len(cont_names)
    assert len(res_cat) == len(cat_names)
    # Make sure `max_card` is correctly used to differentiate int as categorical
    res_cont, res_cat = cont_cat_split(df)
    assert res_cont == ['F']
    assert res_cat == ['A', 'I']


def test_normalize():
    this_tests(Normalize)

    cont_names = ['height']
    normalize_transform = Normalize([], cont_names)
    train_df = pd.DataFrame({'height': [180, 140, 122], 'age': [24, 33, 45]})
    test_df = pd.DataFrame({'height': [184, 102, 113], 'age': [44, 31, 19]})
    normalize_transform(train_df)
    normalize_transform(test_df, test=True)

    expected_test_df = pd.DataFrame({'height': [1.235, -1.527, -1.157], 'age': [44, 31, 19]})

    np.testing.assert_array_almost_equal(expected_test_df["height"], test_df["height"],decimal=3)
    np.testing.assert_array_equal(expected_test_df["age"], test_df["age"])


def test_add_datepart():
    this_tests(add_datepart)

    df = pd.DataFrame({'transaction_date': ['02/03/2017', '02/04/2017', '02/05/2017'], 'category': ['a', 'b', 'a']})
    add_datepart(df, 'transaction_date')

    expected_test_df = pd.DataFrame(
        {'category': ['a', 'b', 'a'],
         'transaction_Year': [2017, 2017, 2017],
         'transaction_Month': [2, 2, 2],
         'transaction_Week': [5, 5, 5],
         'transaction_Day': [3, 4, 5],
         'transaction_Dayofweek': [4, 5, 6],
         'transaction_Dayofyear': [34, 35, 36],
         'transaction_Is_month_end': [False, False, False],
         'transaction_Is_month_start': [False, False, False],
         'transaction_Is_quarter_end': [False, False, False],
         'transaction_Is_quarter_start': [False, False, False],
         'transaction_Is_year_end': [False, False, False],
         'transaction_Is_year_start': [False, False, False],
         'transaction_Elapsed': [1486080000, 1486166400, 1486252800]

         })

    for col in df.columns:
        np.testing.assert_array_equal(df[col], expected_test_df[col])
