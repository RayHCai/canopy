# CLAUDE.md

Guidance for Claude Code and subagents working in this repository.

## What Canopy is

A simulated drone swarm that maps a procedurally generated house, finds the
electric meter, and emits a Site Survey Review (SSR) packet. `spec.md` is the
product/architecture source of truth; read it before proposing design changes,
and treat its "Key architecture decisions" list as settled unless the user says
otherwise.

Layout:

| Path | Holds |
| --- | --- |
| `src/canopy/contracts.py` | Every cross-module data type. An interface, not an implementation detail. |
| `src/canopy/errors.py` | The deliberate-failure exception hierarchy. |
| `src/canopy/{worldgen,sim,perception,mapping,planning,site,viz}/` | The pipeline stages, in dependency order. |
| `src/canopy/cli/` | Entry points (`canopy-fly`). No loose scripts. |
| `docs/adr/` | Architecture decision records. |

Module boundaries (ADR 0009, enforced by `tests/test_architecture.py`):

- Each stage's `__init__.py` is its public API: re-exports plus a sorted
  `__all__`. Another package imports `from canopy.sim import SimWorld`, never
  `canopy.sim.world`. Exporting a new name is how a helper becomes cross-stage
  API, so do it deliberately.
- Inside a stage, import sibling submodules directly; never import your own
  package's facade.
- Stages depend only on core (`contracts`, `errors`, `log`, `mathutil`,
  `config`) and earlier stages. If a later stage has something an earlier one
  needs, move it down; don't import upward.

## Commands

```bash
uv sync                      # deps + dev group
uv run ruff format .         # format
uv run ruff check --fix .    # lint
uv run mypy                  # typecheck (files configured in pyproject)
uv run pytest                # tests
uv run pytest -m "not slow"  # fast loop
```

Run `ruff format`, `ruff check`, `mypy` and `pytest` before declaring work done.
Never `# type: ignore` or `# noqa` to get green — fix the cause, or if the tool
is genuinely wrong, narrow the ignore to one rule code and say why on the line.

## Python practices

The linter enforces most of this; the rest is judgment.

- **Python 3.11, and only 3.11.** Every dependency ships a wheel there. Don't
  reach for 3.12+ syntax.
- **Types are mandatory.** mypy runs `strict`. Annotate every signature,
  including tests where it is not noise. Prefer `npt.NDArray[np.float64]` and
  the aliases in `contracts.py` over bare `np.ndarray`.
- **`from __future__ import annotations` at the top of every module.**
- **Dataclasses for data that crosses a boundary**, declared in `contracts.py`,
  `frozen=True` where it can be, `eq=False` when a field is a NumPy array (a
  generated `__eq__` raises on truth-testing an elementwise comparison).
- **Raise from the hierarchy.** A deliberate failure is a `CanopyError`
  subclass with a message naming the offending value. Never a bare `Exception`,
  never a silent `except: pass`.
- **Vectorize.** This is a per-frame simulation loop: prefer NumPy array ops to
  Python loops over rays, voxels or drones. Profile before optimizing anything
  that is not obviously hot.
- **`pathlib`, not `os.path`.** Enforced by `PTH`.
- **Docstrings are NumPy-style** (`D` + `pydocstyle` convention = numpy) on
  every public module, class and function. Explain *why* the code is shaped the
  way it is — the existing modules set the bar; match it. Inline comments
  earn their place by explaining a non-obvious constraint, not by restating
  the line below.
- **Units are metres, seconds, radians; the world frame is Z-up** with the
  origin at lot centre. Put units in the name or the docstring when ambiguous.
- **Optional extras stay optional.** `rl`, `detect` and `viewer` are additive.
  Import them lazily inside the function that needs them and raise
  `DependencyMissingError` when absent — a missing extra must never break an
  import of `canopy`.
- **Tests:** `pytest`, mirroring the package layout under `tests/`. Mark
  anything over a couple of seconds `@pytest.mark.slow`. Warnings are errors;
  fix the warning rather than filtering it. Test behaviour at module
  boundaries, not private helpers.

## Working style

- Read before you write. Match the surrounding file's idiom, comment density
  and naming rather than importing a house style from elsewhere.
- Prefer editing an existing module to adding a new one. New top-level modules
  need a reason that fits the pipeline stages above.
- Changing an architecture decision from `spec.md` means writing an ADR in
  `docs/adr/` (`NNNN-kebab-title.md`: Context / Decision / Consequences).
- Don't add dependencies without asking. The dependency graph is deliberate.
- Say what you actually ran and what it printed. "Tests pass" means you ran them.

## Delegation and token budget

See `docs/agent-guide.md` for the full policy. The short version:

- **Delegate by default when a task is a search, a sweep, or a self-contained
  unit of work.** A subagent's file dumps stay in its own context; only its
  conclusion reaches this one.
- **Sonnet is the default model for subagents.** Opus is for the orchestration
  turn and for genuinely hard design or debugging. Searching, reading, writing
  tests, applying mechanical edits and summarizing are Sonnet work — routing
  them to Opus costs roughly an order of magnitude more for the same answer.
- Prefer the narrow agent (`repo-explorer`, `py-implementer`, `test-author`,
  `py-reviewer`) over `general-purpose`; narrower tools mean fewer wasted turns.
- Fan out independent subagents in **one message** so they run concurrently.
