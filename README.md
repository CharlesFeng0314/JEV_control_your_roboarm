**English** | [中文](README.zh.md)

# JEV Control Your Roboarm

You say what you want in one sentence. JEV controls the robot arm: it looks at what the wrist camera can support, chooses one allowed action, and the arm carries that choice out. After the arm moves, the camera looks again, a new prompt is built, and JEV chooses the next step. This repeats until the goal is checked, JEV asks you, or the turn budget runs out.

JEV does not send motor commands. Each turn it answers a set of closed questions. The program turns those answers into one action and a checked parameter, then the arm driver executes it.

## What JEV can control the arm to do

| JEV's choice | What the arm is then allowed to do |
| --- | --- |
| Search | Move only the wrist camera through the current view, a nearby set of views, or every configured safe view, and look for the named object |
| Observe | Look again with the wrist camera and localize that object |
| Pick | Grasp an object that this loop has already seen and checked |
| Place | Put the held object at a selected destination |
| Verify | Use the wrist camera to check that the transfer actually happened |
| Nudge the hand | Translate or rotate the hand by one bounded step, such as 3 mm or 5 degrees |
| Adjust the gripper | Hold, open, close, or change the opening by one bounded step per finger |
| Move a tool | Hold, or follow a straight segment, an arc, or a circle, such as stirring |
| Finish | Stop only after a later wrist observation has checked the goal |
| Ask you | Stop and hand the decision back when the goal or the scene is still ambiguous |

Which action, and which step size, is JEV's choice. Whether that choice is allowed, and how the arm moves, belongs to the program and the arm driver.

## How JEV did that

Each turn is four steps: perceive, build JEV's prompt, control, then check.

### 1. Perceive

The wrist RGB-D camera is the task camera. Depth is clustered into separate objects on the table, pixels that belong to the gripper are removed, and CLIP assigns a label, a color, and a grasp description to each cluster. That becomes the current scene snapshot: what is visible, where it is, and how confident the label is.

JEV's API does not remember previous calls, so the program keeps a scene memory and replays it every turn. A new look is merged with the old record instead of replacing it. The memory keeps the object id, how many times each label was seen, the winning label, the latest pose, up to 20 past poses, the observation count, whether the object is visible now, and a stale flag. An object that stays unseen for three later looks is marked stale; it is not deleted. A spectator camera, simulator ground truth, and a validation-only flag are refused and never enter this memory.

### 2. Build the prompt

The prompt is rebuilt from scratch on every turn. It is not a raw image and it is not an open-ended paragraph.

First the program packs a `GIVEN THAT` record:

- your sentence, with surrounding whitespace removed
- a control contract: facts come from the arm's own sensors and driver; simulator ground truth is not allowed; JEV selects the next action and its bounded arguments; validation, motion planning, and actuation stay in ordinary code
- the live driver capabilities, including which actions this driver can actually run
- the action list JEV may choose from, each with its description and legal inputs; if the driver advertises a subset, actions it cannot run are removed before JEV is called
- the current wrist snapshot
- the scene-memory view: known objects, confidences, observation counts, visibility, stale flags, unknown regions, and recent action notes
- only the last 8 action results, not the whole history

That record is what the product window shows and what the run log stores.

Then the same record is split into separate typed questions. Each question has a short instruction and a closed list of answers generated from the facts above:

- **Next action.** The answers are the remaining action descriptions, plus finish and ask-you. The instruction says to use only the user goal, current sensor facts, and recent results, and not to invent objects or poses.
- **Safe to continue** and **information sufficient.** Two yes/no scores. They ask whether the step can be attempted without invented coordinates or ground truth, and whether the wrist evidence is enough for a physical action.
- **Search target.** Answers are the vision vocabulary the driver currently advertises. Choosing a label here does not mean the object is in the scene. If the driver advertises none, the only answer is to ask you instead of inventing a class.
- **Target and destination.** One answer is built for each object visible right now, and one for each remembered object that is not in the current view. The text includes the label, attributes, pose, observation count, and whether it is stale, and it says to look again before touching a stale or currently invisible object. Two extra answers are always present: the thing is not visible, or the evidence does not pick one object confidently.
- **Motion size.** Hand, gripper, and tool questions use named steps (`fine`, `small`, …) whose text already states the physical size, such as 3 mm or 5 degrees. JEV cannot type an arbitrary distance.

After JEV answers, ordinary code maps those names back to numbers (`fine` translation becomes 0.003 m) and drops a target or destination that JEV marked not visible or ambiguous.

### 3. Control

The chosen action must be one of the actions still in the catalog. The driver then runs it: search and observe move the wrist camera and read it again; pick, place, nudge, gripper, and tool motion go through the arm interface. A new wrist snapshot is taken at the start of the next turn, and the action result is written into both the prompt history and the scene memory.

### 4. Check

JEV may choose finish only as an answer. The program accepts that finish only when three things are already true: at least two JEV decisions have happened, some earlier action succeeded, and a wrist snapshot was taken after that success. Otherwise finish is rejected, the rejection is stored as an action result, and JEV is asked again. If JEV chooses ask-you, the loop stops for you. If none of this happens within 20 new turns, the run stops on the turn budget.

## Run

Python 3.11 or newer. The repository includes a tabletop scene for trying the arm.

```powershell
python -m pip install -r requirements.txt
copy .env.example .env
```

Put your own `AI_GATEWAY_API_KEY` in `.env`, then open the window and type the goal:

```powershell
python -m scripts.jev_robot.app --driver-factory scripts.jev_robot.drivers.isaac_rpc:create_driver --scene-id tabletop_household
```

Or pass the sentence directly:

```powershell
python -m scripts.jev_robot.app --driver-factory scripts.jev_robot.drivers.isaac_rpc:create_driver --scene-id tabletop_household --goal "put the sugar box on the red block"
```

The arm driver connects to `127.0.0.1:47631` unless `JEV_ISAAC_RPC_HOST` and `JEV_ISAAC_RPC_PORT` say otherwise. Each run records the prompt, JEV's answers, the action, and the wrist observation under `data/jev_robot/runs/`.
