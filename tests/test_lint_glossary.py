import importlib.util
import json
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / ".github/skills/domain-glossary/lint_glossary.py"
_spec = importlib.util.spec_from_file_location("lint_glossary", _path)
lint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lint)

GLOSSARY = """
| Term | Avoid | Code | Source |
|---|---|---|---|
| `within_corr2` | `within_R2`, `corr2` | `metrics.within_corr2_pooled` | x |
| a \\| b | `old name` | `col` | y |

| Other | Table |
|---|---|
| `ignored` | `also ignored` |
"""


def _hits(path):
    aliases = lint.load_aliases(GLOSSARY)
    return list(lint.find_hits(path, aliases, lint.compile_patterns(aliases)))


def test_load_aliases_reads_only_avoid_column():
    assert lint.load_aliases(GLOSSARY) == {
        "within_R2": "within_corr2", "corr2": "within_corr2", "old name": r"a \| b"}


def test_aliases_match_whole_identifiers_only(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("within_corr2 and pooled_corr2 are fine\nbut corr2 is not; nor OLD\n  NAME\n")
    assert [(loc, found) for loc, found, _ in _hits(f)] == [("2", "corr2")]
    f.write_text("an old   name here")
    assert [found for _, found, _ in _hits(f)] == ["old   name"]


def test_notebook_scans_markdown_cells_only(tmp_path):
    nb = tmp_path / "nb.ipynb"
    nb.write_text(json.dumps({"cells": [
        {"cell_type": "code", "source": ["df['corr2'] = 1\n"]},
        {"cell_type": "markdown", "source": ["# Title\n", "reports within_R2\n"]},
    ]}))
    assert [(loc, found) for loc, found, _ in _hits(nb)] == [("cell 1:line 2", "within_R2")]
