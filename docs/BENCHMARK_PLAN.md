# Benchmark Plan

## Status

This benchmark has **not been run yet**.

This document defines the intended comparison before results are collected so that the evaluation criteria are not retrofitted after seeing outcomes.

## Primary research question

Can JEV preserve useful zero-shot natural-language task execution over a fixed robot skill set while reducing decision latency and API cost relative to an autoregressive LLM decision backend?

## Comparison principle

The decision backend should be the main independent variable.

Keep fixed whenever possible:

- simulator / robot,
- scene seed,
- perception pipeline,
- world-state representation,
- legal action catalog,
- bounded action parameters,
- deterministic validation,
- robot driver and motion execution,
- maximum turn budget.

### Backend A: JEV

Receives the natural-language goal, current sensor-grounded state, recent results, and dynamically generated typed choices.

### Backend B: LLM

Receives semantically equivalent information and the same legal action catalog. It should be constrained to choose from the same actions and parameters rather than being allowed to invent extra capabilities.

### Optional Backend C: deterministic rules

Useful for narrow known tasks as a lower-cost reference, but not a substitute for the JEV-vs-LLM comparison on novel language/task composition.

## Task families

### 1. Language paraphrase

Same physical task, different unseen wording.

Examples:

- “put the red cup in the blue box”
- “move the crimson drinking vessel into the blue container”
- “place the thing I drink from inside the blue box”

### 2. Unseen task composition

Use known skills in a new combination that was not hard-coded as an exact task sequence.

### 3. Distractor objects

Add irrelevant objects to test target selection and semantic grounding.

### 4. Ambiguity

Use instructions that should trigger re-observation or user handoff rather than confident guessing.

### 5. Recovery

Perturb the scene after an action, move an object, occlude a target, or force a failed grasp and measure whether the agent recovers.

## Metrics

Report at minimum:

- task success rate,
- total task completion time,
- decision latency per call (p50 / p95),
- number of decision calls per task,
- decision-backend API cost per task,
- invalid / rejected action rate,
- user-handoff rate,
- recovery success rate,
- turn-budget exhaustion rate.

When possible, also separate failures into:

- perception failure,
- target-grounding failure,
- semantic decision failure,
- deterministic validation rejection,
- motion/driver failure,
- verification failure.

## Experimental reporting

For every result set, record:

- repository commit SHA,
- simulator / robot version,
- decision backend and model/version,
- API pricing date/source when reporting cost,
- hardware used for local components,
- number of episodes,
- scene/task seeds,
- task list or task-generation rule,
- success definition,
- timeout / turn budget.

## Fairness notes

A faster model should not receive a simpler state, and a more capable model should not receive extra actions unavailable to the other backend.

If one backend requires a different representation, document the difference explicitly.

Do not count simulator ground-truth access as sensor-grounded performance. Oracle-ground-truth runs may be useful, but they should be reported separately as oracle baselines.

## First minimal benchmark

A practical first experiment can use a small set of tabletop tasks and 20–30 episodes per backend before scaling up.

The first goal is not to establish a final leaderboard. It is to learn whether the architecture has enough signal to justify a larger evaluation.
