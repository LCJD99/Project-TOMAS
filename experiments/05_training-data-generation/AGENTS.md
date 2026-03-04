# AGENT PLAYBOOK

This document grounds automated contributors working in `experiments/05_training-data-generation`. Read it fully before shipping code.

## Mission
- Generate tooling for analyzing, augmenting, and scheduling Tool DAG workloads (see README.md, README_PLANNING.md).
- Keep data artifacts deterministic: respect seeds, preserve augmented IDs, and never commit regenerated JSON unless asked.
- Scripts are built for Python 3.10+ with minimal deps; prefer reproducibility over speed.

## Repo Snapshot
- `src/models`: Tool DAG primitives (`tool_dag.py`, `tool_node.py`).
- `src/planning`: Scheduler stack (dag analyzer, resource profiler, execution language, scenario generator, plan generator).
- `src/schema`: Token and resource naming (`token_schema.py`).
- `scripts`: CLI entry points for augmentation, analysis, and plan generation.
- `data`: Large JSON/CSV inputs and outputs (treat as read-only unless user requests modifications).

## Environment Setup
1. Use Python 3.10 or newer with venv/conda.
2. Install core dependencies (no requirements.txt provided):
   ```bash
   pip install --upgrade pip
   pip install pandas tqdm
   ```
   Optional: `pip install numpy` for experimentation; do not add unless required.
3. Add repo root to `PYTHONPATH` or run scripts via `python -m` to satisfy intra-package imports.
4. Keep `scripts/` runnable via `python scripts/<name>.py` from repo root.

## Data Assets
- `data/tasks.json`: Original ToolBench-like tasks, never edited in place.
- `data/tasks_augmented.json`: Output from `scripts/augment_tasks.py`; ~1866 tasks.
- `data/system.json`: Resource cap template consumed by planners.
- `data/profiling.csv`: Performance matrix needed by `ResourceProfiler`.
- `data/gt_stage*/`: Batched plan exports. Heavy files; avoid re-checking in.
- Treat `.zip` archives as immutable test fixtures.

## Build & Validation Commands
- Environment smoke test: `python -m compileall src`.
- Type drift probe (lightweight): `python -m py_compile $(git ls-files '*.py')`.
- Lint baseline (if ruff/flake8 unavailable): run `python scripts/analyze_tasks.py` to validate imports & IO.
- Dependency sanity: `python scripts/augment_tasks.py --help` (argparse wiring + paths).
- Add new commands to this list when introducing tooling; keep instructions reproducible.

## Targeted Execution / Single-Test Guidance
- No pytest suite exists; use focused script invocations instead.
- Single augmented sample run:
  ```bash
  python scripts/augment_tasks.py --input data/tasks.json --output /tmp/tasks_augmented_test.json --num-augmented 1 --merge-count 2 --seed 7
  ```
- Single-plan dry run (acts as a "unit test" for planning stack):
  ```bash
  python scripts/generate_execution_plans.py \
    --tasks data/tasks_augmented.json \
    --system data/system.json \
    --profiling data/profiling.csv \
    --output /tmp/plans.json \
    --max-tasks 1 \
    --workers 1 \
    --scenarios 1
  ```
- Quick stats validation:
  ```bash
  python scripts/analyze_tasks.py
  ```
- For iterative development on `src/planning`, run modules directly (each file has a `__main__` guard); eg `python src/planning/resource_profiler.py`.

## Running Full Pipelines
1. **Augment tasks**
   ```bash
   python scripts/augment_tasks.py --input data/tasks.json --output data/tasks_augmented.json --num-augmented 1000 --merge-count 3 --seed 42
   ```
2. **Analyze augmented dataset** (adjust path in script if using alt output):
   ```bash
   python scripts/analyze_augmented.py
   ```
3. **Generate execution plans**
   ```bash
   python scripts/generate_execution_plans.py \
     --tasks data/tasks_augmented.json \
     --system data/system.json \
     --profiling data/profiling.csv \
     --output data/execution_plans_stage1.json \
     --batch-dir data/gt_stage1 \
     --batch-size 100 \
     --scenarios 3 \
     --workers 8 \
     --seed 42
   ```
4. **Regenerate staged outputs** only on request; they are large and versioned externally.

## Coding Standards
- **Python version**: target 3.10 features (pattern matching unused today; dataclasses + type hints required).
- **Imports**: 3 blocks (stdlib, third-party, local). Keep alphabetical ordering within blocks. Use explicit imports (`from typing import Dict, List`). Relative imports stay inside `src` packages; scripts manipulate `sys.path` explicitly — follow existing pattern.
- **Formatting**: PEP 8 line width <= 100 chars. Prefer double quotes for human-facing strings, single quotes acceptable for short tokens. Keep docstrings triple-double-quoted.
- **Types**: All new public functions must have typing annotations. Reuse `typing` aliases already in files; avoid `Any` unless bridging JSON. When returning structured data, define `@dataclass` (see `ToolDAG`, `ResourceConfig`).
- **Mutability**: Avoid in-place mutation of arguments unless documented. Use copies when editing dicts sourced from JSON (see `TaskAugmentor.add_start_node_to_dag`).
- **Seed control**: Always expose `seed` arguments for stochastic flows; default to 42 to align with existing CLI.
- **Logging**: Scripts rely on `print` for progress (tqdm optional). Keep logs concise, prefix multi-step phases with `Step N:`.
- **Error handling**: Raise `RuntimeError` for scheduling failures (see `scheduler.py`). Wrap CLI entrypoints with explanatory messages before `sys.exit(1)`.
- **CLI ergonomics**: Use `argparse.ArgumentDefaultsHelpFormatter` for new scripts. Provide `--help` friendly descriptions.
- **Data safety**: When writing JSON, use `ensure_ascii=False` only if necessary (current scripts do). Always `indent=2` to match repo norm.

## Naming Conventions
- Tool nodes: maintain original `task` strings; when merging, prepend `T{i}_` exactly as `TaskAugmentor` does.
- Generated task IDs: `AUG_XXXXXXXX` zero-padded to 8 digits.
- Scenario structs: camel-cased keys inside JSON (`SYSTEM_STATE`, `PLAN_START`). Python-side types use snake_case (`system_state`, `tool_nodes`).
- Resource levels: `low`, `medium`, `high` with uppercase abbreviations only inside virtual tokens (see `token_schema.py`).
- File/dir names: prefer lowercase with underscores. Scripts remain snake_case.

## Imports & Package Layout Expectations
- `src/planning/__init__.py` exports the public API; new modules must be added there.
- Scripts manipulate `sys.path` by inserting `Path(__file__).parent.parent`; mirror this pattern when introducing new entrypoints.
- Keep circular imports out by isolating shared constants (eg `token_schema`).

## Data Processing Patterns
- Use `json.loads(json.dumps(...))` trick (already present) for safe deep copies when editing DAG nodes.
- Always recompute `n_tools` after mutating node lists.
- Maintain `instruction_list` cleanup semantics when merging tasks—never persist helper keys to disk.
- When adding START edges, insert new links at the beginning to keep deterministic ordering.

## Performance & Parallelism
- Default to sequential execution for debuggability. Only enable `--workers > 1` after validating sequential correctness.
- When using `ProcessPoolExecutor`, guard heavy objects behind `if __name__ == '__main__'` and ensure worker arguments are picklable (PlanGenerator already demonstrates pattern).
- Manage tqdm imports carefully inside workers (lazy import inside exception branches to avoid cross-process contention).

## Testing Philosophy
- Unit-style coverage achieved by running module `__main__` blocks (every major file includes one). Keep new modules consistent.
- For deterministic regression checks, snapshot a small subset of tasks and compare plan counts / latency totals. Do not check fixtures into git; store under `/tmp`.
- When adding dependencies, document verification commands here and in README.md.

## Documentation Expectations
- Update README.md and README_PLANNING.md when surface area changes (new CLI flags, new data columns, etc.).
- Keep QUICKSTART.md bilingual section intact (Chinese headings). If adding translations, mirror the structure.
- Include inline comments only when logic is non-obvious (per repo guidance). Favor descriptive helper names instead.

## External Tooling Rules
- No Cursor or Copilot rule files exist in this repo; if they appear later (`.cursor/rules/`, `.cursorrules`, `.github/copilot-instructions.md`), summarize them here and align behavior accordingly.
- Until then, default to this playbook plus existing Markdown guides.

## Working With Data Files
- Large JSON/CSV artifacts can exceed diff limits. When editing, script the change and describe verification steps in PRs.
- Never commit regenerated `data/execution_plans_stage*.json` unless explicitly requested; they are considered derived assets.
- For experimental outputs, write to `/tmp` or `data/tmp/` (gitignored) and mention paths in notes.

## Pull Request Hygiene
- Follow repo commit style: imperative mood, summary of intent (e.g., "add plan generator batching knobs").
- Run at least one validation command from this file before requesting review; cite the exact command + output summary.
- Reference relevant docs (`README_PLANNING.md`, `EBNF.txt`) inside PR descriptions when touching scheduling logic.

## Security & Privacy
- No secrets in repo. If a script requires credentials later, load via environment variables and document placeholders here.
- Do not embed dataset snippets that may contain user text beyond what already ships in `data/`.

## Checklist Before You Ship
1. Dependencies installed? (`pandas`, `tqdm`)
2. Lint/build commands executed?
3. Single-task dry run passing?
4. Docs updated (if CLI signature changed)?
5. Large outputs avoided or justified?

Stay deterministic, stay explicit, and narrate your automation.
