import pytest
from fastai.structured import proc_df
import pandas as pd
import numpy as np

def test_proc_df_fix_missing():
    y_col = 'target'

    df_train = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2], 'target': [1, 0, 1]})
    df_test = pd.DataFrame({'col1' : [1, 2, np.NaN], 'col2' : [5, np.NaN, 2]})

    # Assume test is the same as train but without target column
    assert len(set(df_train.columns) - set([y_col])) == len(set(df_test.columns))

    X_train, y_train, nas_train = proc_df(df_train, y_fld=y_col)
    # We are expecting nas_train to contain one column
    assert len(nas_train) == 1

    original_nas_length = len(nas_train)
    X_test, _, nas_test = proc_df(df_test, y_fld=None, na_dict=nas_train)
    # We expect nas_train to be unchanged
    assert len(nas_train) == original_nas_length

    # We are expecting nas_test to contain two columns
    assert len(nas_test) == 2

    # We are expecting the test set to have the same columns as train set because we have used the na_dict from train set
    assert set(X_train.columns) == set(X_test.columns)
