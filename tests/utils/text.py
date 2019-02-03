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

class CaptureStd():
    """ Context manager to capture:
    stdout, clean it up and make it available via obj.out
    stderr, and make it available via obj.err

    init arguments:
    - out - capture stdout: True/False, default True
    - err - capture stdout: True/False, default True

    Examples:

    with CaptureStdout() as cs:
        print("Secret message")
    print(f"captured: {cs.out}")

    import sys
    with CaptureStdout() as cs:
        print("Warning: ", file=sys.stderr)
    print(f"captured: {cs.err}")

    # to capture just one of the streams, but not the other
    with CaptureStdout(err=False) as cs:
        print("Secret message")
    print(f"captured: {cs.out}")
    # but best use the stream-specific subclasses

    """
    def __init__(self, out=True, err=True):
        if out:
            self.out_buf = StringIO()
            self.out = 'error: CaptureStd context is unfinished yet, called too early'
        else:
            self.out_buf = None
            self.out = 'not capturing stdout'

        if err:
            self.err_buf = StringIO()
            self.err = 'error: CaptureStd context is unfinished yet, called too early'
        else:
            self.err_buf = None
            self.err = 'not capturing stderr'

    def __enter__(self):
        if self.out_buf:
            self.out_old = sys.stdout
            sys.stdout = self.out_buf

        if self.err_buf:
            self.err_old = sys.stderr
            sys.stderr = self.err_buf

        return self

    def __exit__(self, *exc):
        if self.out_buf:
            sys.stdout = self.out_old
            self.out = apply_print_resets(self.out_buf.getvalue())

        if self.err_buf:
            sys.stderr = self.err_old
            self.err = self.err_buf.getvalue()

    def __repr__(self):
        msg = ''
        if self.out_buf: msg += f"stdout: {self.out}\n"
        if self.err_buf: msg += f"stderr: {self.err}\n"
        return msg

# in tests it's the best to capture only the stream that's wanted, otherwise
# it's easy to miss things, so unless you need to capture both streams, use the
# subclasses below (less typing). Or alternatively, configure `CaptureStd` to
# disable the stream you don't need to test.

class CaptureStdout(CaptureStd):
    """ Same as CaptureStd but captures only stdout """
    def __init__(self):
        super().__init__(err=False)

class CaptureStderr(CaptureStd):
    """ Same as CaptureStd but captures only stderr """
    def __init__(self):
        super().__init__(out=False)
