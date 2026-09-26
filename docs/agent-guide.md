# Agent and delegation guide

How Claude Code should use subagents in this repository, and how to keep that
cheap. Read alongside `CLAUDE.md`.

## Why delegate at all

A subagent has its own context window. Everything it reads — the twenty files
it grepped through, the 400-line module it opened to find one function — stays
there and is discarded when it finishes. Only its final report crosses back.

So delegation is not about parallelism first. It is about **what you avoid
carrying**. A search that would splatter 30k tokens of file content into the
main conversation costs a few hundred tokens as a subagent report.

The corollary: delegation is a loss when the answer is already one tool call
away. Spawning an agent to read a file you can name is strictly more expensive
than reading it.

## The rule

| Situation | Do |
| --- | --- |
| You know the file and the symbol | Read it yourself |
| One grep answers it | Grep it yourself |
| "Where is X handled?" across the package | `repo-explorer` (Sonnet) |
| "How do worldgen, sim and perception agree on the manifest?" | `repo-explorer` (Sonnet) |
| A self-contained module or function to write, with the design settled | `py-implementer` (Sonnet) |
| Tests for code that already exists | `test-author` (Sonnet) |
| Review a diff before you hand it over | `py-reviewer` (Sonnet) |
| Architecture, a subtle numerical bug, a decision with tradeoffs | Keep it on Opus, in the main turn |

## Model routing — the point of this document

**Default every subagent to Sonnet.** Pass `model: "sonnet"` on the `Agent`
call, or rely on the `model:` line in the agent definition (all four repo
agents already pin Sonnet).

Sonnet is the right tool for tasks where the work is *finding* or *applying*
rather than *deciding*:

- searching, mapping, and summarizing the codebase
- writing a function against a spec you already wrote
- writing tests for behaviour that is already pinned down
- mechanical refactors: renames, signature changes, import sweeps
- running the toolchain and reporting what broke

Reserve Opus for:

- choosing an architecture, or revisiting a `spec.md` decision
- debugging something where the *hypothesis* is the hard part — a drift in the
  raycast frame, a Hungarian assignment that oscillates
- reviewing a change whose correctness is subtle rather than stylistic

Escalate a specific task to Opus when the Sonnet run comes back confused or
hedging. That is cheaper than routing everything to Opus on the chance one
task needs it.

## Writing a prompt that doesn't waste tokens

A subagent starts blind. The cost of a vague prompt is a subagent that reads
half the repo to orient itself, then reports something you then have to
correct — two full runs for one answer.

Include, every time:

1. **The goal in one sentence**, and what "done" looks like.
2. **Where to start** — the files, directories or symbols you already know are
   relevant. This is the single biggest saving available.
3. **The constraints** — "3.11", "mypy strict", "don't add dependencies",
   "contracts.py types only".
4. **The shape of the answer you want.** "Return the file:line of each call
   site and one line on what it does" beats "investigate". Ask for a bounded
   report, not a narrative.
5. **Explicit scope limits** — read-only, or "edit only `src/canopy/mapping/`".

And keep the fan-out honest: independent agents go in **one message** so they
run concurrently; dependent work is sequential and should just be sequential.
Don't spawn a second agent to check the first agent's work unless the work is
risky — read its report instead.

## What not to do

- Don't delegate and then redo the search yourself. Wait for the report.
- Don't spawn an agent per file for a change that touches six files. One agent,
  one clear brief.
- Don't use `general-purpose` when a narrow agent fits. Broad tool access is
  an invitation to wander.
- Don't ask a subagent to make a product or architecture decision. Ask it to
  gather what the decision needs, then decide in the main turn.
- Don't let a subagent be the only thing that verified a change. Run the
  toolchain yourself before reporting done.
