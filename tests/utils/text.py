""" Helper functions for dealing with testing text outputs """

import sys, re
from io import StringIO

# When any function contains print() calls that get overwritten, like progress bars,
# a special care needs to be applied, since under pytest -s captured output (capsys
# or contextlib.redirect_stdout) contains any temporary printed strings, followed by
# \r's. This helper function ensures that the buffer will contain the same output
# with and without -s in pytest, by turning:
# foo bar\r tar mar\r final message
# into:
# final message
# it can handle a single string or a multiline buffer
def apply_print_resets(buf):
    return re.sub(r'^.*\r', '', buf, 0, re.M)

def assert_screenout(out, what):
    out_pr = apply_print_resets(out).lower()
    match_str = out_pr.find(what.lower())
    assert match_str != -1, f"expecting to find {what} in output: f{out_pr}"

class CaptureStdout():
    """ Context manager to capture stdout, clean it up and make it available via obj.out or str(obj).

    Example:

    with CaptureStdout() as cs:
        print("Secret message")
    print(f"captured: {cs.out}")
    # or via its stringified repr:
    print(f"captured: {cs}")

    """
    def __init__(self):
        self.buffer = StringIO()
        self.out = 'error: CaptureStdout context is unfinished yet, called too early'

    def __enter__(self):
        self.old = sys.stdout
        sys.stdout = self.buffer
        return self

    def __exit__(self, *exc):
        sys.stdout = self.old
        self.out = apply_print_resets(self.buffer.getvalue())

    def __repr__(self):
        return self.out
