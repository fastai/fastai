# this test checks that the core fastai library doesn't depend on unnecessary modules.

import sys

# packages that shouldn't be required to be installed for fastai to work:
unwanted_deps = ['jupyter_contrib_nbextensions']

class CheckDependencyImporter(object):

    def find_spec(self, fullname, path, target=None):
        #print("spec: ", fullname, path, target)
        # catch if any of the unwanted dependencies gets triggered
        for mod in unwanted_deps:
            if mod == fullname:
                print(f"detected unwanted dependency on {mod}")
                sys.exit(1)
        return None

sys.meta_path.insert(0, CheckDependencyImporter())

def test_unwanted_mod_dependencies():
    # unload any candidates we want to test, including fastai, so we can test import
    for mod in unwanted_deps + ['fastai']:
        if mod in sys.modules: del sys.modules[mod]
    # test
    import fastai
