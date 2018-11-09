import pytest, torch
from fastai.gen_doc import gen_notebooks
from fastai.gen_doc import nbdoc

def assert_link(docstr, expected, nb_cells=None, modules=None, msg=''):
    if modules is None: modules = gen_notebooks.get_imported_modules(nb_cells or [])
    linked = nbdoc.link_docstring(modules, docstr)
    assert linked == expected, f'{msg}\nExpected: {expected}\nActual  : {linked}'

def build_nb_cells(mod_names):
    return [{'cell_type': 'code', 'source': f'from {m} import *'} for m in mod_names]

@pytest.mark.skip(reason="need to update")
def test_torchvision():
    docstr   = 'Note that `tvm` is the namespace we use for `torchvision.models`.'
    expected = 'Note that [`tvm`](https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models) is the namespace we use for `torchvision.models`.'
    assert_link(docstr, expected, msg='Should match imported aliases')

def test_fastai_prefix():
    docstr   = "functions for your application (`fastai.vision`)"
    expected = "functions for your application ([`fastai.vision`](/vision.html#vision))"
    assert_link(docstr, expected, msg='Should match keywords prefixed with fastai. See `index.ipynb`')

def test_link_typedef():
    docstr   = "- `LayerFunc` = `Callable`\[`nn.Module`],`None`]"
    expected = "- `LayerFunc` = `Callable`\[[`nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)],`None`]"
    assert_link(docstr, expected, modules=[torch], msg='Type definitions to torch formatted incorrectly. See fastai_typing.ipynb')

def test_link_typedef_double_bt():
    docstr   = "- `ParamList` = `Collection`\[`nn`.`Parameter`]"
    expected = "- `ParamList` = `Collection`\[[`nn`](https://pytorch.org/docs/stable/nn.html#torch-nn).`Parameter`]"
    assert_link(docstr, expected)

def test_link_inner_class_functions():
    docstr   = "To train your model in mixed precision you just have to call `Learner.to_fp16`, which converts the model and modifies the existing `Learner` to add `MixedPrecision`."
    expected = "To train your model in mixed precision you just have to call [`Learner.to_fp16`](/train.html#to_fp16), which converts the model and modifies the existing [`Learner`](/basic_train.html#Learner) to add [`MixedPrecision`](/callbacks.fp16.html#MixedPrecision)."
    imports = 'from fastai.callbacks.fp16 import *'
    assert_link(docstr, expected, nb_cells=[gen_notebooks.get_code_cell(imports)])

def test_class_anchor():
    docstr   = "`DataBunch.create`, `DeviceDataLoader.proc_batch`"
    expected = "[`DataBunch.create`](/basic_data.html#DataBunch.create), [`DeviceDataLoader.proc_batch`](/basic_data.html#DeviceDataLoader.proc_batch)"
    imports = 'from fastai.basic_train import *'
    assert_link(docstr, expected, nb_cells=[gen_notebooks.get_code_cell(imports)])

def test_link_class_methods():
    docstr   = "`ImageDataBunch.from_csv`"
    expected = "[`ImageDataBunch.from_csv`](/vision.data.html#ImageDataBunch.from_csv)"
    imports = 'from fastai.vision.data import *'
    assert_link(docstr, expected, nb_cells=[gen_notebooks.get_code_cell(imports)])

def test_respects_import_order():
    docstr   = "`learner`"
    expected = "[`learner`](/vision.learner.html#vision.learner)"
    assert_link(docstr, expected, build_nb_cells(['fastai.text', 'fastai.vision']))

    expected_text = "[`learner`](/text.learner.html#text.learner)"
    assert_link(docstr, expected_text, build_nb_cells(['fastai.vision', 'fastai.text']))

def test_nb_module_name_has_highest_priority():
    # get_imported_modules.nb_module_name should have highest priority. This is the associated notebook module.
    # Ex: vision.transforms.ipynb is associated with fastai.vision.transforms
    docstr   = "`transform`"
    expected = "[`transform`](/text.transform.html#text.transform)"
    modules = gen_notebooks.get_imported_modules([], nb_module_name='fastai.text.transform')
    assert_link(docstr, expected, modules=modules)

    expected = "[`transform`](/tabular.transform.html#tabular.transform)"
    modules = gen_notebooks.get_imported_modules([], nb_module_name='fastai.tabular.transform')
    assert_link(docstr, expected, modules=modules)

@pytest.mark.skip(reason="need to update")
def test_application_links_top_level_modules():
    # Snippet taken from applications.ipynb
    docstr = """## Module structure
In each case (except for `collab`), the module is organized this way:
### `transform`
### `data`
### `models`
### `learner`"""
    expected = """## Module structure
In each case (except for [`collab`](/collab.html#collab)), the module is organized this way:
### [`transform`](/text.transform.html#text.transform)
### [`data`](/text.data.html#text.data)
### [`models`](/text.models.html#text.models)
### [`learner`](/text.learner.html#text.learner)"""
    assert_link(docstr, expected, msg='data, models should link to highest module. transform and learner links to first match')

def test_link_vision_learner_priority():
    # Edge case for vision.learner.ipynb
    imports = """from fastai.gen_doc.nbdoc import *
    from fastai.vision import *
    from fastai import *
    from fastai.vision import data
    """

    docstr   = "Pass in your `data`, calculated `preds`, actual `y`,"
    expected = "Pass in your [`data`](/vision.data.html#vision.data), calculated `preds`, actual `y`,"
    err_msg = "`data` should link to vision.data instead of text.data."
    modules = gen_notebooks.get_imported_modules([gen_notebooks.get_code_cell(imports)], nb_module_name='fastai.vision.learner')
    assert_link(docstr, expected, modules=modules, msg=err_msg)

