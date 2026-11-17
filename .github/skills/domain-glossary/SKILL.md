---
name: domain-glossary
description: Maintain and apply the repo's shared vocabulary in docs/CONTEXT.md. Use when writing or reviewing docs, notebooks, SQL, issues or PRs, when a new domain concept or naming question comes up, or to check files for non-canonical terms.
---

# Domain glossary

`docs/CONTEXT.md` is the single source of truth for domain vocabulary: modules, metrics, scope words and domain nouns. Its **Terms to avoid** table lists aliases that are always wrong, and the linter checks for them.

## Apply (every time you write or review)

1. Read `docs/CONTEXT.md` before writing docs, notebook markdown, SQL comments, issues or PR text.
2. Use canonical terms and exact code names. Don't coin synonyms or invent new labels or column names for concepts that already exist. A correlation is never called "R²"; always say which scope (within-city, between-city, pooled).
3. Check a definition in code or docs before using it; don't infer it from a column name.
4. If a concept you need isn't in `CONTEXT.md`, use **Capture** below. Don't silently introduce a new word.

## Capture (new or changed concept)

Use when a chat introduces a new concept, when two words are being used for one idea, or when a term's meaning changes.

1. Check whether the concept already exists under another name. If it does, use that term and offer to add the new word to **Terms to avoid**.
2. Propose the entry to the user: the canonical term, a one-to-three-sentence definition, and the exact code name (column, function, constant). Wait for the user to confirm the canonical term.
3. Add the definition to the right section of `CONTEXT.md` (Modules, Metric vocabulary, Domain nouns). If the user names aliases that are always wrong, add a row to **Terms to avoid**: **Term**, **Avoid** (backticked exact aliases), **Code**, **Source** (file or ADR).
4. When renaming a term, move the old name into **Avoid**. Don't delete it.
5. If the concept reflects a decision, check whether an ADR in `docs/adr/` needs updating.
6. Glossary edits follow the `issue-workflow` skill: make them on the current issue's branch if related, otherwise open a `repo` issue.

### Avoid-table rules
- Put aliases in backticks, comma-separated. The linter reads only backticked text in the **Avoid** column.
- Only list aliases that are always wrong. Words that are right in some contexts (for example "within-city R²", which is correct for `r2_city`) belong in the definition, not in **Avoid**.
- Escape pipes in cells as `\|`.

## Lint (consistency check)

```bash
python3 .github/skills/domain-glossary/lint_glossary.py                 # tracked .md/.sql/.yml + notebook markdown
python3 .github/skills/domain-glossary/lint_glossary.py docs/ notebooks/regression_modelling/models/05_predictor_diagnostics.ipynb
```

The script prints `file:line: 'alias' -> use 'Canonical'` (notebooks as `file:cell N:line M`) and exits 1 on any hit. It skips `docs/CONTEXT.md` itself and `docs/adr/`, because ADRs are dated records that may quote retired names. Code cells and `.py` files are not scanned.

- Before opening a PR, lint the files you changed and fix any hits in them.
- Renaming hits in files outside the current issue's scope (especially column names that cached tables or notebooks depend on) is out of scope. Report them, or open a follow-up issue, instead of fixing them on the current branch.
