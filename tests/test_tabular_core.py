import pandas as pd
from fastai.tabular.core import FillMissing, TabularPandas

def test_fillna():
    # Mock data
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
    na_dict = {"a": 0, "b": -1}
    
    # Initialize TabularPandas with appropriate columns
    tab_pandas = TabularPandas(
        df, 
        procs=[],  # No preprocessing steps required
        cont_names=["a", "b"], 
        cat_names=[],
        y_names=[]
    )
    
    # Initialize FillMissing
    fill_missing = FillMissing(add_col=False)
    fill_missing.na_dict = na_dict  # Manually set the na_dict for testing
    
    # Apply the transformation
    fill_missing.encodes(tab_pandas)

    # Check results
    assert (tab_pandas["a"] == [1, 0, 3]).all()
    assert (tab_pandas["b"] == [4, 5, -1]).all()
