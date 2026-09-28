[English](README.md) | **中文**

# JEV Control Your Roboarm

**让机械臂执行自然语言任务，但不把自回归 LLM 放在高频执行闭环里。**

这个项目想验证一个很简单的问题：

> 能不能把“传感器感知 + JEV typed decision + 可复用机器人技能 + 确定性运动控制”组合起来，让机器人面对新的自然语言操作任务时，不需要每一步都去问一个 LLM？

当前实现的闭环是：

```text
用户自然语言任务
        ↓
腕部 RGB-D 感知
        ↓
当前场景 + 场景记忆
        ↓
程序动态生成当前合法 choices
        ↓
JEV：决定下一步做什么
        ↓
受限制的机器人动作 / skill
        ↓
确定性校验 + 机器人驱动
        ↓
再次观察
        ↺
```

JEV **不直接发电机指令，不生成任意坐标，也拿不到仿真 ground truth**。它只能在程序已经确认当前机器人具备的动作和参数范围里做选择。

> **项目状态：实验阶段。** 当前闭环和 Isaac Sim 接入已经实现，但还没有公开 demo 视频，也还没有完成严格的 JEV-vs-LLM 对照 benchmark。两者都是下一步工作。

## 这个项目到底在验证什么

这里的假设不是“JEV 应该替代 MoveIt、运动规划或底层控制”。

真正想测试的是：JEV 能不能成为位于感知与确定性机器人技能之间的 **zero-shot 语义决策层**。

```text
传感器告诉我们发生了什么。
普通代码检查硬约束和事实。
JEV 决定“现在该做什么”。
机器人栈负责“具体怎么动”。
```

之后真正公平的比较应该是：

```text
相同感知
相同 world state
相同可用动作
相同机器人驱动

JEV 决策后端  vs  LLM 决策后端
```

然后测任务成功率、决策延迟、总任务时间、API 成本、非法动作率、失败恢复能力等。

详细计划见 [docs/BENCHMARK_PLAN.md](docs/BENCHMARK_PLAN.md)。

## 这里说的 zero-shot 是什么

这里的 **zero-shot 并不是说视觉模型、JEV 或机器人什么都没训练过**。

这里指的是：只要新任务能够由机器人已有的 skill 组合出来，就不需要为了这条新指令重新做 task-specific robot training，也不需要提前给这条 exact instruction 写一个状态机。

例如机器人已经会：

- `pick`
- `place`
- `observe`
- 有边界的手部微调

那么第一次收到：

> “把糖盒放到红色方块上。”

系统可以根据当前场景和当前 action catalog 在线完成组合。

但如果 driver 根本不存在某个物理技能，JEV 不会凭空发明它。

## JEV 和机器人栈的边界

可以把整个系统理解成四层：

- **感知：** 看到了什么？在哪里？置信度多少？
- **JEV：** 根据用户目标和当前状态，下一步应该做什么？
- **普通代码：** 这个选择是否合法、信息是否足够、约束是否满足？
- **机器人驱动：** 选定的动作物理上怎么执行？

这种分层是刻意设计的。JEV 不是开放式生成任意动作字符串，而是在程序动态生成的一组 closed choices 里做语义判断。

## 当前 action catalog

| JEV 的选择 | 程序 / driver 接下来允许做的事 |
| --- | --- |
| 寻找 | 只移动腕部相机，在配置好的安全视角里找指定物体 |
| 观察 | 再看一次并重新定位目标 |
| 抓取 | 抓取已经被当前闭环看见并检查过的物体 |
| 放置 | 按 JEV 选择的 on、inside、next_to、方向关系或有限偏移放置手中物体 |
| 核对 | 重新观察目标和目的地，验证所选空间关系 |
| 微调手 | 只沿一个方向平移/旋转一档，例如 3 mm 或 5° |
| 调整夹爪 | 一次闭合到接触以抓起，或一次完全张开以释放；不再做手指微步进 |
| 工具运动 | 保持、走直线、圆弧或圆周，例如搅拌 |
| 结束 | 在后续观察已经核对目标后请求结束 |
| 问你 | 目标或场景仍有歧义时停止并把决定交还给用户 |

在调用 JEV 之前，程序会根据 **当前 driver 实际支持的能力**过滤 choice。

## 感知和 grounding

任务相机是腕部 RGB-D。当前实现通过深度将桌面物体分簇，去掉属于夹爪自身的像素，再用 CLIP 为每个 cluster 生成标签、颜色和抓取描述。

由于 JEV 每次 API 调用本身不保存历史，程序自己维护 scene memory，包括：物体 ID、标签、置信度、当前是否可见、是否 stale、位姿历史、观察次数，以及最近动作结果。

旁观相机、仿真真值和仅用于 validation 的 privileged state 被明确禁止进入 JEV 的决策上下文。

完整 prompt 构造、scene memory 和 finish gate 见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 当前进度

已经实现：

- [x] 自然语言任务输入
- [x] 腕部 RGB-D 感知链路
- [x] 当前 scene snapshot + 持久 scene memory
- [x] 根据场景和 driver capability 动态生成 JEV choices
- [x] 有边界的末端 / 工具动作参数，以及二值抓起 / 放开控制
- [x] 用普通代码拒绝 unsupported 或信息不足的动作
- [x] 动作后重新观察
- [x] finish gate 和 turn budget
- [x] Isaac Sim RPC driver 路径
- [x] 动态 pick/place RPC、JEV 空间关系选择和腕部视觉验证
- [x] 产品化本地 Web 界面按“题目 → choices → JEV’s choice”展示，完整 JSON 仅保留在运行归档
- [x] 每次运行记录 prompt、JEV 回答、动作和腕部观察

下一步：

- [ ] 公开 demo 视频 / GIF
- [ ] JEV vs LLM 严格对照 benchmark
- [ ] 不需要 Isaac Sim 的 replay/mock 模式
- [ ] 更多任务类型和扰动 / recovery case
- [ ] 真机械臂验证

## 运行

需要 Python 3.11 或更新版本。

```powershell
python -m pip install -r requirements.txt
copy .env.example .env
```

在 `.env` 中填写自己的 `AI_GATEWAY_API_KEY`。

当前产品由两个常驻进程组成。请打开两个 PowerShell 窗口，并都先进入项目目录。

**终端 A — 启动 Isaac Sim 与机械臂 RPC 服务：**

```powershell
cd H:\robo
H:\robo\.conda-isaacsim\python.exe -u scripts\manipulation\simulation\isaac_scene_host.py --config scripts\manipulation\config\tabletop_household_franka_wrist_rgbd_v1.json --output data\manipulation\scene_hosts\tabletop_franka_live --port 47631
```

这一个命令会同时打开 Isaac Sim GUI，并在 `127.0.0.1:47631` 提供场景、物理、机械臂状态和腕部 RGB-D 服务，不需要再启动第三个服务。保持这个终端和 Isaac 窗口常驻。

**终端 B — 启动 JEV Web 控制台：**

```powershell
cd H:\robo
H:\robo\.conda\python.exe -m scripts.jev_robot.app --driver-factory scripts.jev_robot.drivers.isaac_rpc:create_driver --scene-id tabletop_franka_live
```

浏览器会打开本地控制台。输入任务并点击 **Start fresh run**，每次都会从 turn 1 开始。机械臂仍会在内部使用腕部 RGB-D 做感知，但 Web 控制台不再传输或显示相机预览。

也可以不打开 Web 页面，直接运行一句任务：

```powershell
H:\robo\.conda\python.exe -m scripts.jev_robot.app --driver-factory scripts.jev_robot.drivers.isaac_rpc:create_driver --scene-id tabletop_franka_live --goal "把糖盒放到红色方块上"
```

默认连接 `127.0.0.1:47631`；真机或远端服务可用 `JEV_ISAAC_RPC_HOST` 和 `JEV_ISAAC_RPC_PORT` 修改。停止时先在终端 B 按 `Ctrl+C`，再关闭终端 A / Isaac Sim。

每次运行的 prompt、JEV 回答、选择动作和腕部观察记录在：

```text
data/jev_robot/runs/
```

## 欢迎把它玩坏

如果你找到系统完成不了、理解错或者恢复失败的任务，请直接开 issue。

最好附上：

- 自然语言任务原文
- scene / robot 环境
- 系统感知到了什么
- JEV 选择了什么
- 你预期它做什么
- 如果方便，附一份去掉敏感信息的 run log

尤其欢迎：

- 新 simulator / 真机械臂 driver
- 新感知 backend
- LLM baseline adapter
- replay/mock 模式
- benchmark 结果
- 有歧义或具有挑战性的自然语言指令
- 能揭示 typed decision 边界的失败案例

贡献说明见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 这是研究问题，不是 benchmark 结论

当前仓库展示的是一套架构和实现。现在 **还不能声称** JEV 在机器人操作任务上已经被严格证明比 LLM 更快、更便宜、更安全或成功率更高。

下一步就是把这件事测出来。

## License

MIT，见 [LICENSE](LICENSE)。
