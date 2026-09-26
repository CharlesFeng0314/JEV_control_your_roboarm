# Contributing

Thanks for testing JEV Control Your Roboarm.

The most valuable contribution is not necessarily new code. A reproducible failure case, a comparison against another decision backend, or a new robot/simulator adapter can be equally useful.

## Ways to contribute

### 1. Break the agent

Try a natural-language manipulation task that should be expressible with the existing skills but causes the loop to fail, become ambiguous, or make a poor decision.

Please include:

- exact user instruction,
- simulator or robot setup,
- commit SHA,
- visible objects / scene description,
- selected JEV action(s),
- expected behavior,
- actual behavior,
- sanitized run log if available.

### 2. Add a robot or simulator driver

A driver should clearly advertise the actions it actually supports. Unsupported actions must be removed from the JEV choice set before a decision call.

Please keep privileged simulator ground truth out of the JEV decision context unless the contribution is explicitly marked as an oracle/debug baseline.

### 3. Add a perception backend

Perception contributions should make clear which information comes from sensors, which comes from learned models, and which comes from simulator-only validation.

### 4. Add a decision baseline

LLM, rule-based, or other decision backends are welcome. For fair comparison, keep the following fixed whenever possible:

- perception output,
- world-state representation,
- legal action catalog,
- robot driver,
- task/environment seed.

### 5. Publish benchmark results

Please report enough information for another person to reproduce the result. See `docs/BENCHMARK_PLAN.md` and the benchmark-result issue template.

## Architecture rules

Contributions should preserve these boundaries unless the PR explicitly proposes an alternative architecture:

1. Sensors/perception provide facts.
2. The program builds the legal choice set.
3. JEV selects among those choices.
4. Ordinary code validates hard constraints.
5. The driver executes motion/control.
6. The scene is observed again after execution.

JEV should not silently receive simulator ground truth, arbitrary hidden coordinates, or actions the driver cannot execute.

## Pull requests

A good PR description should answer:

- What problem does this change solve?
- Does it alter the information available to JEV?
- Does it change the legal action space?
- How was it tested?
- Does it introduce simulator-only privileged information?

Small, focused PRs are preferred.
