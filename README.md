# jevcontrol_your_roboarm

自然语言目标进入决策循环。JEV 从 action 清单里选择，对应模块做参数校验，再交给驱动执行，并把新的观察送进下一轮。

只改这个仓库里的文件。公开 GitHub 也是从这里推上去的，没有第二份代码目录。

密钥、本机路径、benchmark、仿真宿主和试跑参数不会进 Git。

## 运行

需要 Python 3.11 或更新版本。

```powershell
python -m pip install -r requirements.txt
copy .env.example .env
```

在 `.env` 里填写你自己的 `AI_GATEWAY_API_KEY`，然后：

```powershell
python -m scripts.jev_robot.app --driver-factory scripts.jev_robot.drivers.isaac_rpc:create_driver --scene-id tabletop_household
```

不打开界面：

```powershell
python -m scripts.jev_robot.app --driver-factory scripts.jev_robot.drivers.isaac_rpc:create_driver --scene-id tabletop_household --goal "描述你想让机器人完成的事"
```

Isaac RPC 默认连接 `127.0.0.1:47631`，可用 `JEV_ISAAC_RPC_HOST` 和 `JEV_ISAAC_RPC_PORT` 修改。仓库不包含仿真宿主。

## Actions

| Action | 模块 |
| --- | --- |
| `search_object` | `scripts/manipulation/actions/search_object.py` |
| `observe_object` | `scripts/manipulation/actions/observe_object.py` |
| `pick_object` | `scripts/manipulation/actions/pick_object.py` |
| `place_object` | `scripts/manipulation/actions/place_object.py` |
| `verify_transfer` | `scripts/manipulation/actions/verify_transfer.py` |
| `adjust_end_effector` | `scripts/manipulation/actions/adjust_end_effector.py` |
| `adjust_gripper` | `scripts/manipulation/actions/adjust_gripper.py` |
| `execute_tool_motion` | `scripts/manipulation/actions/execute_tool_motion.py` |

## 感知

产品用的是腕部 RGB-D：

- `scripts/manipulation/perception/rgbd_geometry.py`
- `scripts/manipulation/perception/wrist_semantics.py`
- `scripts/manipulation/perception/wrist_rgbd.json`

本地如果还有试跑配置，会优先用那份。没有的话就用上面的 `wrist_rgbd.json`。CLIP 权重不进仓库；本地有 `weights/clip/ViT-B-32.pt` 就用它，否则用模型名 `ViT-B/32` 下载。

## 场景

`scenes/manipulation/tabletop_household_franka_wrist_rgbd_v1.usd` 和 `scenes/simple_room.usd` 是试验场景。

## 记录

运行记录写在 `data/jev_robot/`，这个目录不提交。
