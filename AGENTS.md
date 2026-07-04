# Instructions for Agents

## Research Focus

This repository is a research prototype, not a production codebase. Keep the scope tight. We do not need a large feature surface; we need the minimum work required to prove or disprove concrete research hypotheses.

## Tooling & Sandbox

### Using Python

- Always invoke Python through `uv run python ...`; do not run `python`/`python3` directly.
- When running `uv` commands in the sandbox, set `UV_CACHE_DIR` to a writable directory under `/tmp` by default so verification stays local and non-escalated; the default `uv` cache location is read-only in this environment.
- Use `uv add <package>` to add dependencies — never use `pip install` or `pip`.
- Use `uv lock` and `uv sync` as appropriate.
- Refer to uv documentation for any packaging or environment questions.

### Protobuf generated files

Never edit generated protobuf files (`*_pb2.py`, `*_pb2_grpc.py`, `*_pb2.pyi`) directly. Instead:

1. Edit the `.proto` source files.
2. Regenerate with `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/generate_protobuf.py`.
3. Keep the generated runtime files and typing files in sync by committing the regenerated outputs together.

Use the repo-owned generation script rather than ad hoc `protoc` commands or manual post-generation edits.


### Plotting notebooks

When authoring a notebook to produce plots:

- Include a single toggle near the top (e.g. a `SAVE_OUTPUT` boolean) that, when disabled, suppresses writing PDF plots and CSVs to disk so the notebook only renders inline.
- Set x-axis label and tick font size to at least 11.
- Keep the notebook minimal: only the cells needed to load data and render the plot.

### Git in the sandbox

- Use read-only Git commands such as `git status`, `git diff`, and `git show` without escalation.
- For Git commands that write repository metadata under `.git`, such as `git add` and `git commit`, use escalation from the start because the sandbox blocks writes like creating `index.lock`.
- If the user explicitly asks to make a Git commit, treat that as permission to exit Plan Mode and perform the commit flow directly instead of spending a turn planning it.

## Code Style

- Do not add default values to function parameters unless truly necessary. Avoid littering the codebase with defaults — prefer requiring arguments explicitly so call sites are clear about what they pass.
- When making multiple changes to the same file, group them into as few edit operations as possible so the user doesn't have to approve each change individually. Use your judgement for large changes where splitting may improve clarity.
- When code changes, also update tests, `README`, and plans as needed so documentation and implementation remain aligned. 
- Be thoughtful when adding tests: prefer targeted tests for meaningful behavior changes and regressions, and do not add tests for every minor or mechanical change.

### Plans

- Keep plans concise so they are quick to review.
- When a plan adds or changes RPC calls, clearly call out which RPC calls are added/changed, when they occur in the flow, and their expected frequency (per request, per batch, periodic, retry-driven, etc.). 

### PRs

When asked to draft a PR title and description:

- Output the title and description in raw Markdown format. The user copies the title and description into GitHub.
- Have sections: `Summary`, `Changelog`, and `Tests`.
- Put the "why" rationale inline in the `Summary` instead of as a separate section.
- Base the title and description on the actual branch diff against `origin/main`.
- Run the relevant tests before generating the title and description to verify the Tests section accurately reflects what was tested.
- Keep it brief.
