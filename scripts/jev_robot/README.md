# JEV robot product

Product entry is `python -m scripts.jev_robot.app`. Actions live in `scripts/manipulation/actions/`. Wrist RGB-D perception lives in `scripts/manipulation/perception/`.

```powershell
python -m scripts.jev_robot.app --driver-factory scripts.jev_robot.drivers.isaac_rpc:create_driver --scene-id tabletop_household
```
