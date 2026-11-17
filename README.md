# crime-idx-2026

Proof of concept for a block-group crime-risk model. Start with
`.github/copilot-instructions.md`, `docs/CONTEXT.md` (vocabulary) and `docs/architecture.md`.

## Setup

```bash
poetry install --no-root
poetry run pytest -q
```

Data under `data/` is gitignored; see `data/README.md`.

## Workflow

Every change goes **issue → branch → PR → squash merge**; nothing is committed straight to `main`.

1. Open work lives in GitHub Issues. Each issue has Goal, Scope, Done when and a
   **Context** brief for the session that picks it up (`.github/ISSUE_TEMPLATE/`).
2. `bin/start-issue N` creates a worktree at `../crime-idx-2026.worktrees/N-slug` from
   `origin/main`, shares `data/` and `.venv` with this checkout, opens VS Code and starts a
   Copilot session named after the branch. Re-run it to resume. `--dry-run` shows the steps.
3. Open a PR that says `Closes #N` (`.github/pull_request_template.md`); squash-merge once
   approved, then remove the worktree from the root checkout.

Details are in the Copilot skills: `.github/skills/issue-workflow/` and
`.github/skills/domain-glossary/` (vocabulary rules and `lint_glossary.py`).

To call `start-issue` from any folder, put a dispatcher on your `PATH` that runs the
current repo's `bin/start-issue`:

```bash
cat > ~/.local/bin/start-issue <<'SH'
#!/usr/bin/env bash
top="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)/.." || top=""
[[ -x "$top/bin/start-issue" ]] || { echo "start-issue: not in a repo with bin/start-issue" >&2; exit 1; }
exec "$top/bin/start-issue" "$@"
SH
chmod +x ~/.local/bin/start-issue
```
