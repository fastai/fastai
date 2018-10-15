# this test checks that the core fastai library doesn't depend on unnecessary modules.

import sys

# packages that shouldn't be required to be installed for fastai to work
# this is basically dev_requirements from setup.py which won't be installed by default
unwanted_deps = 'jupyter_contrib_nbextensions distro'.split()

class CheckDependencyImporter(object):

    def find_spec(self, fullname, path, target=None):
        #print("spec: ", fullname, path, target)
        # catch if import of any unwanted dependencies gets triggered
        assert fullname not in unwanted_deps, f"detected unwanted dependency on '{fullname}'"
        return None

def test_unwanted_mod_dependencies():
    # save the original state
    mod_saved = sys.modules['fastai'] if 'fastai' in sys.modules else None
    meta_path_saved = sys.meta_path.copy

    # unload any candidates we want to test, including fastai, so we can test their import
    for mod in unwanted_deps + ['fastai']:
        if mod in sys.modules: del sys.modules[mod]

    # test
    try:
        sys.meta_path.insert(0, CheckDependencyImporter())
        import fastai
    finally:
        # restore the original state
        del sys.meta_path[0]
        if mod_saved is not None: sys.modules['fastai'] = mod_saved
