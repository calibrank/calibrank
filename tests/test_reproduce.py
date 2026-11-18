"""The committed results/ are exactly what the records produce.

Regenerates the run tables, every LaTeX table and figure of the paper, and the
claims check from records/ into a temporary directory and compares them with the
committed files, byte for byte: the main grid from records/runs, the extended
evaluation from records/extended. ``make results`` rewrites results/ in place.

The paired statistics of the extended evaluation (results/extended/run_stats.json)
take about ten minutes to recompute, so the regular test reads the committed file;
``pytest -m slow`` recomputes it as well.
"""

import contextlib
import filecmp
import io
import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

RECORDS = os.path.join(ROOT, "records")
RESULTS = os.path.join(ROOT, "results")
STATS = os.path.join(RESULTS, "extended", "run_stats.json")


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    import aggregate
    import claims
    import make_report
    import make_report_ext
    out = tmp_path_factory.mktemp("results")
    ext = out / "extended"
    with contextlib.redirect_stdout(io.StringIO()):
        assert aggregate.main(["--records", RECORDS, "--out", str(out)]) == 0
        assert make_report.main(["--records", RECORDS, "--results", str(out)]) == 0
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        assert claims.main(["--records", RECORDS, "--results", str(out)]) == 0
    (out / "claims.txt").write_text(text.getvalue().replace(RECORDS, "records"))
    with contextlib.redirect_stdout(io.StringIO()):
        assert aggregate.main(["--records", os.path.join(RECORDS, "extended"), "--out", str(ext)]) == 0
        shutil.copy(STATS, ext / "run_stats.json")
        assert make_report_ext.main(["--records", os.path.join(RECORDS, "extended"),
                                     "--main", str(out), "--results", str(ext)]) == 0
    return str(out)


def _files(root):
    return sorted(os.path.relpath(os.path.join(d, f), root)
                  for d, _, files in os.walk(root) for f in files)


def test_same_files(regenerated):
    assert _files(regenerated) == _files(RESULTS)


@pytest.mark.parametrize("name", ["runs.csv", "dense.csv", "claims.txt",
                                  "extended/runs.csv", "extended/dense.csv"])
def test_tables_and_claims_match(regenerated, name):
    assert filecmp.cmp(os.path.join(regenerated, name), os.path.join(RESULTS, name), shallow=False)


@pytest.mark.parametrize("sub", ["tables", "figures", "extended/tables", "extended/figures"])
def test_paper_tables_and_figures_match(regenerated, sub):
    names = sorted(os.listdir(os.path.join(RESULTS, sub)))
    _, mismatch, errors = filecmp.cmpfiles(os.path.join(regenerated, sub),
                                           os.path.join(RESULTS, sub), names, shallow=False)
    assert not mismatch and not errors, (sub, mismatch, errors)


@pytest.mark.slow
def test_paired_statistics_match(tmp_path):
    import run_stats
    out = tmp_path / "run_stats.json"
    with contextlib.redirect_stdout(io.StringIO()):
        assert run_stats.main(["--records", os.path.join(RECORDS, "extended"), "--out", str(out)]) == 0
    assert filecmp.cmp(out, STATS, shallow=False)
