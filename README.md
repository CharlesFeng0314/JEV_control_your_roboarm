**English** | [中文](README.zh.md)

# JEV Control Your Roboarm

**Natural-language robot manipulation without an autoregressive LLM in the execution loop.**

This project explores a simple question:

> Can a robot execute new natural-language manipulation tasks by combining sensor-grounded perception, Jev typed decisions, reusable robot skills, and deterministic motion/control — without asking an LLM what to do at every step?

The current implementation runs a closed loop:

```text
Natural-language goal
        ↓
Wrist RGB-D perception
        ↓
Scene snapshot + memory
        ↓
Dynamically generated valid choices
        ↓
JEV: choose what to do next
        ↓
Bounded robot action / skill
        ↓
Deterministic validation + robot driver
        ↓
Observe again
        ↺
```

JEV does **not** send motor commands, invent arbitrary coordinates, or receive simulator ground truth. It chooses among actions and parameters that the program has already verified are available to the current robot driver.

> **Project status:** experimental. The control loop and Isaac Sim integration are implemented. A public demo video and a controlled JEV-vs-LLM benchmark have **not** been published yet; both are planned next.

## What this project is testing

The hypothesis is not that JEV should replace motion planning or low-level control.

The hypothesis is that JEV may be useful as a **zero-shot semantic decision layer** between perception and deterministic robot skills:

```text
Sensors tell us what is there.
Code checks hard constraints.
JEV makes the fuzzy semantic decision.
The robot stack executes the motion.
```

The intended comparison is eventually:

```text
same perception
same scene state
same available actions
same robot driver

JEV decision backend  vs  LLM decision backend
```

Then measure task success, decision latency, total task time, cost, invalid-action rate, and recovery behavior.

See [docs/BENCHMARK_PLAN.md](docs/BENCHMARK_PLAN.md).

## What “zero-shot” means here

In this repository, **zero-shot does not mean the robot has never been trained on anything**.

It means a new natural-language task can be attempted **without task-specific robot training or a hand-written state machine for that exact instruction**, as long as the task can be composed from skills the robot already has.

For example, if the robot already supports `pick`, `place`, `observe`, and bounded hand motions, a previously unseen instruction such as:

> “Put the sugar box on the red block.”

can be decomposed online from the current scene and the available action catalog.

JEV does not invent a completely new physical skill that is absent from the driver.

## The boundary between JEV and the robot

A useful way to read the architecture is:

- **Perception:** What is visible, where is it, and how confident are we?
- **JEV:** Given the user goal and current state, what should happen next?
- **Ordinary code:** Is that choice legal, sufficiently grounded, and internally consistent?
- **Robot driver:** How is the selected action physically executed?

This separation is intentional. JEV receives closed, dynamically generated choices instead of an unrestricted text-generation interface.

## Current action catalog

| JEV choice | What the program/driver may do next |
| --- | --- |
| Search | Move only the wrist camera through configured safe views and look for a named object |
| Observe | Re-observe with the wrist camera and localize an object |
| Pick | Grasp an object that has already been seen and checked |
| Place | Put the held object at a selected destination |
| Verify | Re-observe and check whether the intended transfer happened |
| Nudge the hand | Translate or rotate the hand by one bounded step, such as 3 mm or 5 degrees |
| Adjust the gripper | Hold, open, close, or change opening by one bounded step |
| Move a tool | Hold or follow a straight segment, arc, or circle, such as stirring |
| Finish | Request termination after later observation has checked the goal |
| Ask you | Stop and return control when the goal or scene remains ambiguous |

The available choices are filtered from the **live driver capabilities** before JEV is called.

## Sensor grounding

The task camera is a wrist RGB-D camera. The current implementation uses depth clustering to separate tabletop objects, removes gripper pixels, and uses CLIP to assign object labels, colors, and grasp descriptions.

The decision loop maintains scene memory across calls because JEV calls themselves are stateless. It tracks object identity, labels, confidence, visibility, stale state, pose history, observation count, and recent action results.

A spectator camera, simulator ground truth, and validation-only privileged state are intentionally kept out of JEV's decision context.

For the full prompt construction and verification flow, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Project status

Implemented:

- [x] Natural-language task input
- [x] Wrist RGB-D perception path
- [x] Scene snapshot and persistent scene memory
- [x] Dynamic JEV choices derived from scene state and driver capabilities
- [x] Bounded motion/gripper/tool parameters
- [x] Deterministic rejection of unsupported or insufficiently grounded actions
- [x] Re-observation after actions
- [x] Finish gating and turn budget
- [x] Isaac Sim RPC driver path
- [x] Per-run logging of prompts, JEV answers, actions, and observations

Next:

- [ ] Public demo video / GIF
- [ ] JEV vs LLM controlled benchmark
- [ ] Replay/mock mode for people without Isaac Sim
- [ ] More task families and perturbation/recovery cases
- [ ] Physical robot validation

## Run

Python 3.11 or newer.

```powershell
python -m pip install -r requirements.txt
copy .env.example .env
```

Put your own `AI_GATEWAY_API_KEY` in `.env`.

The current runnable path expects the Isaac RPC driver to be available. Open the app and type a goal:

```powershell
python -m scripts.jev_robot.app --driver-factory scripts.jev_robot.drivers.isaac_rpc:create_driver --scene-id tabletop_household
```

Or pass the instruction directly:

```powershell
python -m scripts.jev_robot.app --driver-factory scripts.jev_robot.drivers.isaac_rpc:create_driver --scene-id tabletop_household --goal "put the sugar box on the red block"
```

The driver connects to `127.0.0.1:47631` unless `JEV_ISAAC_RPC_HOST` and `JEV_ISAAC_RPC_PORT` specify otherwise.

Each run stores the prompt, JEV answers, selected action, and wrist observation under:

```text
data/jev_robot/runs/
```

## Break it

This repository is more useful if other people try tasks that fail.

If you find a failure, please open an issue with:

- the natural-language instruction,
- the scene / robot setup,
- what the robot perceived,
- what JEV selected,
- what you expected instead,
- and a sanitized run log if possible.

Especially useful contributions include:

- new simulator or physical-arm drivers,
- new perception backends,
- LLM baseline adapters,
- replay/mock execution,
- benchmark results,
- adversarial or ambiguous instructions,
- failure cases that expose the boundary of typed decision control.

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Research question, not a benchmark claim

This repository currently demonstrates an architecture and implementation. It does **not** yet claim that JEV is faster, cheaper, safer, or more successful than an LLM for robotic manipulation in a controlled benchmark.

That comparison is the next experiment.

## License

MIT. See [LICENSE](LICENSE).
