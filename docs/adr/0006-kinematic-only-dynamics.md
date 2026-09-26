# 6. Kinematic-only dynamics; the PyBullet track is removed

Date: 2026-09-25
Status: accepted

## Context

The PyBullet track (ADR 0003, ADR 0004) needed WSL2 on Windows, a `git
subtree`-vendored fork, and a reference governor: `DSLPIDControl` is tuned for
the Crazyflie 2.X's 27 g airframe, so handing it a waypoint metres away
saturated the attitude command, and the governor that fixed that capped
physics flight at ~0.6 m/s against kinematic's `sim.v_max` = 3.0 m/s. It was
also a second motion-model code path that every consumer of `Dynamics` had to
keep in sync with `KinematicDynamics`.

None of this was ever on the critical path. The spec's own priority order puts
the mapping demo and classical planning first and calls flight-dynamics
fidelity a non-goal; `--dynamics pybullet` existed to show the sim could drive
a physics engine, not because the planner, mapper or site solver needed it.
ADR 0005 already replaced the PyBullet GUI with `canopy-view` as the way to
watch a flight, which removed the last reason to keep the track around just to
have something to look at.

## Decision

Remove the PyBullet backend entirely. `KinematicDynamics` in
`canopy.sim.dynamics` is the only motion model; there is no `Dynamics`
protocol to implement against, no `make_dynamics` factory, and no
`--dynamics` flag to choose one.

This supersedes ADR 0003 and ADR 0004, and the spec's "Key architecture
decisions" bullet "Default motion model is kinematic. PyBullet physics is an
optional flag." — the model is now kinematic, full stop.

## Consequences

- One `uv sync` on any platform is the whole install; no WSL2, no second
  `.venv-wsl`, no `scripts/bootstrap-wsl.sh`.
- `third_party/gym-pybullet-drones` is deleted. It is recoverable from git
  history at the `git subtree` merge commit `70bd6a9` if a future ADR brings
  physics back.
- The Python 3.11 pin (ADR 0002) is no longer forced by PyBullet's lack of a
  newer wheel, but it stands for now; revisiting it is a separate decision.
- The spec's stretch-goal list drops "physics mode" — there is no flag left to
  build it behind.
- If physics is wanted again — for RL training realism, say — it comes back
  through a new ADR, not a revert of this one.
