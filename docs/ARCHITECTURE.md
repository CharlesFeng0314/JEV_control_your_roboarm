# Architecture

This document describes the current JEV control loop in more detail.

## 1. Perceive

The task camera is the wrist RGB-D camera.

Depth is clustered into separate tabletop objects, pixels belonging to the gripper are removed, and CLIP assigns a label, color, and grasp description to each cluster. This produces the current scene snapshot: what is visible, where it is, and how confident the label is.

JEV calls are stateless, so the program maintains scene memory across turns. A new observation is merged into that memory instead of replacing it.

The memory stores, among other fields:

- object id,
- label evidence and winning label,
- latest pose,
- pose history,
- observation count,
- current visibility,
- stale state,
- recent action notes.

An object that remains unseen for later observations may become stale but is not silently deleted.

A spectator camera, simulator ground truth, and validation-only privileged flags are intentionally excluded from the JEV decision context.

## 2. Build the decision context

The decision context is rebuilt each turn.

It contains a `GIVEN THAT` record including:

- the user's natural-language goal,
- the control contract,
- live driver capabilities,
- the current legal action catalog,
- the current wrist-camera snapshot,
- scene memory,
- recent action results.

The same facts are then converted into separate typed questions with closed answer sets.

### Next action

Answers are generated from the actions the current driver can actually execute, plus termination / handoff choices.

JEV is instructed to use only the user goal, sensor-grounded facts, and recent results, and not to invent objects or poses.

### Safe to continue / information sufficient

These questions help gate physical actions when the current evidence is incomplete or would require invented coordinates.

### Search target

Candidates come from the vision vocabulary currently exposed by the driver/perception stack. Selecting a label does not assert that the object is already present.

### Target and destination

Candidates are built from visible objects and remembered objects. The text includes the attributes needed for semantic choice, such as label, pose, observation count, visibility, stale state, and confidence.

Explicit “not visible” and “ambiguous” choices are included so the model does not need to fabricate a target.

### Motion size

Hand, gripper, and tool motions use named bounded steps such as `fine` or `small`. The choice text states the physical meaning of each step.

JEV never types an arbitrary free-form distance. Ordinary code maps the selected name back to a numeric parameter.

## 3. Control

The selected action must still exist in the current action catalog.

The driver then executes the corresponding physical behavior. Depending on the action, this may include:

- moving the wrist camera,
- taking another observation,
- picking,
- placing,
- bounded hand motion,
- gripper adjustment,
- tool motion.

The action result is recorded into recent history and scene memory.

## 4. Re-observe and check

A new wrist observation is taken on the following turn.

The loop does not assume an action succeeded merely because a command was issued.

`Finish` is also only a JEV choice. The program accepts it only when the required deterministic conditions are satisfied, including a successful earlier action and a later wrist observation.

If the finish gate fails, the rejection becomes part of the next decision context and JEV is called again.

If JEV chooses the user-handoff action, the loop stops and returns control to the user.

A turn budget prevents an unbounded loop.

## Design principle

The intended boundary is:

```text
Perception -> facts
Program -> legal choices + hard checks
JEV -> semantic judgment
Driver -> physical execution
```

The architecture deliberately avoids turning JEV into a servo controller or free-form motion generator.
