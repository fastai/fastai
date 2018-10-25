from fastai import *
from fastai.text import *

import pytest
@pytest.mark.skip(reason="per Jeremy's post on github issues #946")
def test_should_set_final_activations_to_n_labels():
    test_data = [
        {'lbl1': 0, "lbl2":1, 'lbl3': 0, "lbl4":1, "text": "fast ai is a cool project"}, 
        {'lbl1': 0, "lbl2":1, 'lbl3': 0, "lbl4":1, 'text': "hello world"}
    ]

    df = pd.DataFrame(test_data)
    data_clas_test = TextClasDataBunch.from_df('/tmp/', df, df, txt_cols=['text'], label_cols=['lbl1', 'lbl2'])
    learn = RNNLearner.classifier(data_clas_test, lin_ftrs=[50], ps=[0.1])
    
    x = learn.layer_groups[-1]
    layers = [l for l in x.modules()]

    np.testing.assert_equal(layers[-1].out_features, 4)
