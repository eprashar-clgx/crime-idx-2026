#!/usr/bin/env python3
"""Flag 'Avoid' aliases from the glossary table in docs/CONTEXT.md.

Usage: python3 .github/skills/domain-glossary/lint_glossary.py [PATH ...]
With no paths, scans git-tracked .md/.sql/.yml files and the markdown cells of tracked
notebooks. Prints `file:line: 'alias' -> use 'Canonical'` and exits 1 on any hit.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
GLOSSARY = ROOT / "docs" / "CONTEXT.md"
SUFFIXES = {".md", ".sql", ".yml", ".yaml", ".ipynb"}
# ADRs are dated records and may quote retired names; the glossary and this skill define them.
EXCLUDE = {GLOSSARY, ROOT / "docs" / "adr", ROOT / ".github" / "skills" / "domain-glossary"}


def cells(line):
    return [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def load_aliases(text):
    """Map each backticked alias in an `Avoid` column to its canonical `Term`."""
    aliases, term_i, avoid_i = {}, None, None
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            term_i = avoid_i = None
            continue
        row = cells(line)
        if "Avoid" in row and "Term" in row:
            term_i, avoid_i = row.index("Term"), row.index("Avoid")
            continue
        if avoid_i is None or len(row) <= avoid_i or set(row[0]) <= set("-: "):
            continue
        canonical = re.sub(r"[*`]", "", row[term_i])
        for alias in re.findall(r"`([^`]+)`", row[avoid_i]):
            aliases[alias] = canonical
    return aliases


def compile_patterns(aliases):
    return {
        a: re.compile(r"(?<![A-Za-z0-9_])" + r"\s+".join(map(re.escape, a.split())) + r"(?![A-Za-z0-9_])", re.I)
        for a in aliases
    }


def file_lines(path):
    """Yield (location, text); for notebooks only markdown cells, located as `cell N:line M`."""
    if path.suffix != ".ipynb":
        for n, text in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            yield str(n), text
        return
    for i, cell in enumerate(json.loads(path.read_text()).get("cells", [])):
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        for n, text in enumerate(src.splitlines(), 1):
            yield f"cell {i}:line {n}", text


def find_hits(path, aliases, patterns):
    for loc, text in file_lines(path):
        for alias, pat in patterns.items():
            for m in pat.finditer(text):
                yield loc, m.group(0), aliases[alias]


def target_files(args):
    if args:
        paths = []
        for a in args:
            p = Path(a).resolve()
            paths += [f for f in p.rglob("*") if f.is_file()] if p.is_dir() else [p]
    else:
        out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
        paths = [ROOT / f for f in out.splitlines()]
    return [p for p in paths
            if p.suffix in SUFFIXES and not any(p == e or e in p.parents for e in EXCLUDE)]


def main():
    aliases = load_aliases(GLOSSARY.read_text())
    patterns = compile_patterns(aliases)
    hits = 0
    for path in sorted(target_files(sys.argv[1:])):
        for loc, found, canonical in find_hits(path, aliases, patterns):
            hits += 1
            print(f"{path.relative_to(ROOT)}:{loc}: '{found}' -> use '{canonical}'")
    print(f"{hits} alias hit(s), {len(aliases)} aliases checked", file=sys.stderr)
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
