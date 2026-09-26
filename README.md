# JEV Control Your Roboarm

你用一句话说出要做的事。JEV 看着机械臂现在能看见什么、能做什么，然后决定下一步。程序检查这个决定，让机械臂去做，做完再看一次，再交给 JEV，直到这件事做完，或者 JEV 停下来问你。

JEV 不直接拧电机。它每次只选一个 **action**：机械臂被允许做的一件事，以及这件事允许范围内的参数，比如往哪边、动多少、看到什么为止。

## 机械臂能做的事

| 你看到的选择 | 机械臂实际在做什么 |
| --- | --- |
| 寻找物体 | 只移动腕部相机，在当前画面、近处或更大范围里找你说的东西 |
| 观察物体 | 再用腕部相机看一次，确认它在哪 |
| 抓起物体 | 抓住已经看到、并且确认过的东西 |
| 放下物体 | 把手里的东西放到选定的位置 |
| 检查结果 | 用腕部相机确认东西是不是到了该去的地方 |
| 微调末端 | 让手沿一个方向移动或转动一小段，例如几毫米或几度 |
| 调整夹爪 | 张开、合上，或每次只开合一小段 |
| 工具运动 | 拿着工具走直线、弧线，或转圈，例如在杯子里搅拌 |

选哪一件、参数取哪一档，由 JEV 决定。动作能不能做、怎么执行，由程序和机械臂驱动完成。

腕部相机看到的画面会先被整理成场景事实：桌面上有什么、大概在哪、有多确定。JEV 根据这些事实做选择，不会拿到仿真里的标准答案。

## 跑起来

需要 Python 3.11 或更新版本。仓库里有一张桌面场景，用来试这只机械臂。

```powershell
python -m pip install -r requirements.txt
copy .env.example .env
```

在 `.env` 里填入你自己的 `AI_GATEWAY_API_KEY`，然后打开界面，输入你想让机械臂做的事：

```powershell
python -m scripts.jev_robot.app --driver-factory scripts.jev_robot.drivers.isaac_rpc:create_driver --scene-id tabletop_household
```

也可以直接给一句话：

```powershell
python -m scripts.jev_robot.app --driver-factory scripts.jev_robot.drivers.isaac_rpc:create_driver --scene-id tabletop_household --goal "把糖盒放到红色方块上"
```

机械臂驱动默认连接本机 `127.0.0.1:47631`。地址和端口可以用 `JEV_ISAAC_RPC_HOST`、`JEV_ISAAC_RPC_PORT` 改。

每次运行的选择、动作结果和腕部观察会记在 `data/jev_robot/runs/`。
