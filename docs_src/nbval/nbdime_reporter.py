"""
pytest ipython plugin modification - Nbdime reporter

Authors: V.T. Fauske

"""

# import the pytest API
import pytest
from _pytest.main import EXIT_OK, EXIT_TESTSFAILED, EXIT_INTERRUPTED, \
    EXIT_USAGEERROR, EXIT_NOTESTSCOLLECTED

import re
import copy
import tempfile
import os
import shutil
import io

import nbformat
import nbdime
from nbdime.webapp.nbdiffweb import run_server, browse

from .plugin import IPyNbCell, bcolors

nbdime.log.set_nbdime_log_level('ERROR')

_re_nbval_nodeid = re.compile('.*\.ipynb::Cell \d+')


class NbdimeReporter:
    def __init__(self, config, file=None):
        self.config = config
        self.verbosity = self.config.option.verbose
        self._numcollected = 0

        self.nbval_items = []

        self.nb_ref = nbformat.v4.new_notebook()
        self.nb_test = nbformat.v4.new_notebook()

        self.stats = {}

    # ---- These functions store captured test items and reports ----

    def pytest_runtest_logreport(self, report):
        """Store all test reports for evaluation on finish"""
        rep = report
        res = self.config.hook.pytest_report_teststatus(report=rep)
        cat, letter, word = res
        self.stats.setdefault(cat, []).append(rep)

    def pytest_collectreport(self, report):
        """Store all collected nbval tests for evaluation on finish
        """
        items = [x for x in report.result if isinstance(x, IPyNbCell)]
        self.nbval_items.extend(items)
        self._numcollected += len(items)


    # ---- Code below writes up report ----

    @pytest.hookimpl(hookwrapper=True)
    def pytest_sessionfinish(self, exitstatus):
        """Called when test session has finished.
        """
        outcome = yield
        outcome.get_result()
        summary_exit_codes = (
            EXIT_OK, EXIT_TESTSFAILED, EXIT_INTERRUPTED, EXIT_USAGEERROR,
            EXIT_NOTESTSCOLLECTED)
        if exitstatus in summary_exit_codes:
            # We had some failures that might need reporting
            self.make_report(outcome)

    def make_report(self, outcome):
        """Make report in form of two notebooks.

        Use nbdime diff-web to present the difference between reference
        cells and test cells.
        """
        failures = self.getreports('failed')
        if not failures:
            return
        for rep in failures:
            # Check if this is a notebook node
            msg = self._getfailureheadline(rep)
            lines = rep.longrepr.splitlines()
            if len(lines) > 1:
                self.section(msg, lines[1])
            self._outrep_summary(rep)
        tmpdir = tempfile.mkdtemp()
        try:
            ref_file = os.path.join(tmpdir, 'reference.ipynb')
            test_file = os.path.join(tmpdir, 'test_result.ipynb')
            with io.open(ref_file, "w", encoding="utf8") as f:
                nbformat.write(self.nb_ref, f)
            with io.open(test_file, "w", encoding="utf8") as f:
                nbformat.write(self.nb_test, f)
            run_server(
                port=0,     # Run on random port
                cwd=tmpdir,
                closable=True,
                on_port=lambda port: browse(
                    port, ref_file, test_file, None))
        finally:
            shutil.rmtree(tmpdir)

    #
    # summaries for sessionfinish
    #
    def getreports(self, name):
        l = []
        for x in self.stats.get(name, []):
            if not hasattr(x, '_pdbshown'):
                l.append(x)
        return l

    def section(self, title, details):
        # Create markdown cell with title
        source = "## " + title
        if details:
            details = details.replace(bcolors.OKBLUE, '')
            source += "\n\n**" + details + '**'
        header = nbformat.v4.new_markdown_cell(source)
        # Add markdown in both ref and test
        self.nb_ref.cells.append(header)
        self.nb_test.cells.append(header)

    def _outrep_summary(self, rep):
        # Find corresponding item
        for item in self.nbval_items:
            if item.nodeid == rep.nodeid:
                # item found, output
                # Sanitize reference cell
                ref_cell = item.cell
                ref_cell.outputs = item.sanitize_outputs(ref_cell.outputs)
                self.nb_ref.cells.append(item.cell)
                test_cell = copy.copy(item.cell)
                if item.test_outputs:
                    test_cell.outputs = item.sanitize_outputs(item.test_outputs)
                self.nb_test.cells.append(test_cell)

    def _getfailureheadline(self, rep):
        if hasattr(rep, 'location'):
            domain = rep.location[2]
            return domain
        else:
            return "test session"  # XXX?
