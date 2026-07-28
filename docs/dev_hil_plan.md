# A10 真机强化学习平台搭建 (dev/hil)

> 目标:在 A10 真机上搭建 **π0.5 + 残差 HIL RL** 平台,验证"VLA BC 预训练 + HIL 残差 RL 微调 > 纯 VLA"的核心 claim,后续做论文新颖性。

## 一、背景与现状(已核实)

### 已有资产
- **π0.5 微调 checkpoint**:`openpi/checkpoints/pi05_a10_finetune/Reach_5_9_1/130000`
  - 配置 `pi05_a10_finetune`,LoRA(gemma_2b_lora + gemma_300m_lora)
  - 任务 "Reach the yellow lemon",数据集 `dataset/dataset_5_9`(LeRobot 格式)
- **推理服务**:`uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_a10_finetune --policy.dir=...`(websocket :8000)
- **部署桥接**:`openpi/third_party/A10_new/client/run_bridge.py`,async 双缓冲 10Hz,发 `SET_JOINTS_BATCH`
- **A10 TCP 控制器**:`third_party/A10_new`,端口 8080
- **XLeVR 遥操作**:已跑通,EE delta/target 模式
- **LeRobot HIL-SERL 基础设施**:SAC / actor-learner(gRPC)/ processor / ReplayBuffer / reward classifier 全部内置

### 关键事实
| 项 | 值 |
|----|----|
| 动作格式 | 7D = `[joint_1..6, gripper]`,训练时前 6 维转 delta、gripper 绝对;推理输出经 `AbsoluteActions` 转回**绝对关节目标** |
| action_horizon | 10 |
| 观测 | `observation/state`(7D) + `observation/images/right`(224² CHW) + `observation/images/top`(224² CHW) + `prompt` |
| 夹爪 | 100mm 行程,0–100mm,绝对位置 |
| GPU | 5090 (24GB) |

## 二、架构

```
π0.5 WebsocketPolicyServer (5090, 独立进程)
   infer({state, images/right, images/top, prompt}) → actions [10, 7] (绝对关节)
        ▲  WebsocketClientPolicy + ActionChunkBroker (chunk=10)
        │  每步释放 a_vla[t] (7D 绝对关节)
        ▼
ResidualSACHead (本地小网络: resnet10 encoder + actor/critic MLP)
   输入: 相机(128²) + 关节状态(7) + a_vla[t](7)
   输出: Δa (7D 关节修正)
        ▼
action = a_vla[t] + α·Δa   (α 起步 0.1)
        ▼
A10RobotEnv → A10TCPClient.send_joint_targets(绝对关节 7D)  [SET_JOINTS]
        ▼
reward (classifier / 手动) + XLeVR 干预 (override action)
        ▼ transitions
Learner (5090): SAC 残差头更新 + ReplayBuffer
  offline ← dataset_5_9 ; online ← actor
```

**维度全链路对齐**:π0.5 输出 7D 绝对关节 → 残差 7D → SET_JOINTS 7D,无需维度转换。

## 三、分阶段实施

### Phase 0 — π0.5 推理服务在环验证
- 用 `WebsocketClientPolicy` + `ActionChunkBroker` 写最小 actor 客户端
- 固定 prompt,每 10 步拉一次 chunk,以 10Hz 取 `a_vla[t]` 送 A10
- **验收**:π0.5 + A10 开环跑 50 步,无掉帧,动作合理
- **止损**:若推理跟不上 10Hz,降级到 5Hz 或换 diffusion policy

### Phase 1 — A10RobotEnv 适配层
- 新建 `src/lerobot/rl/gym_manipulator_a10.py`
- `A10RobotEnv(gym.Env)`:不依赖 `robot.bus`,直接 `get_observation`/`send_action`
- 动作空间 7D 绝对关节(与 π0.5 对齐),reset 用 `SET_JOINTS` 平滑插值
- **验收**:env 单独跑随机动作 100 步稳定;π0.5 闭环跑通一个 episode

### Phase 2 — XLeVR 干预适配
- 给 `XLeVRTeleop` 加 `get_teleop_events()`(复用 VREventHandler)
  - trigger → is_intervention;grip → success;thumbstick → rerecord
- 让 `get_action()` 返回 EE delta dict
- **验收**:π0.5 闭环,人按 VR trigger 能接管,松开回 π0.5

### Phase 3 — ResidualSAC 残差策略 + offline 预热
- 新建 `src/lerobot/policies/residual_sac/`
  - `ResidualSACConfig`:base_policy_server host/port、α、网络
  - `ResidualSACPolicy`:持有 π0.5 客户端(不内嵌)、小 ResNet encoder、actor/critic head
  - `select_action`:拉 `a_vla` + 本地 Δa → `a_vla + α·Δa`
- ReplayBuffer offline ← `dataset_5_9`(`ReplayBuffer.from_lerobot_dataset`)
- critic Q(s, a) 中 a = combined action,梯度只流过 Δa
- **验收**:offline critic loss 下降,actor 残差不发散

### Phase 4 — 在线 HIL 训练
- actor rollout:π0.5 server 出 chunk → 残差加 Δa → env → transition → learner
- 人用 XLeVR 干预,干预帧用 teleop_action,仍进 buffer
- wandb:episodic reward、intervention rate、critic loss
- **验收**:intervention rate 下降;成功率 > 纯 π0.5

### Phase 5 — 评估与论文实验
- baseline:纯 π0.5
- 对比:π0.5 + 残差(无 HIL) / π0.5 + 残差 + HIL
- 指标:成功率、cycle time、intervention rate 曲线
- **验收**:π0.5 + HIL 残差 > 纯 π0.5,可复现曲线

## 四、文件改动清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `docs/dev_hil_plan.md` | 新建 | 本文档 |
| `src/lerobot/rl/gym_manipulator_a10.py` | 新建 | `A10RobotEnv` |
| `src/lerobot/teleoperators/xlevr/teleop_xlevr.py` | 改 | 加 `get_teleop_events()` |
| `src/lerobot/policies/residual_sac/` | 新建 | 残差 SAC 策略 |
| `src/lerobot/rl/actor.py` | 改 | dispatch a10 + 残差头 + π0.5 客户端 |
| `src/lerobot/rl/learner.py` | 改 | 支持残差 SAC 更新 |
| `src/lerobot/configs/train_config_residual_pi05_a10.json` | 新建 | 训练配置 |

## 五、风险与止损

| 风险 | 缓解 |
|------|------|
| π0.5 推理跟不上 10Hz | 降级 5Hz 或换 diffusion policy |
| 残差 SAC 发散 | α 设小 (0.01) 或先只训 critic |
| 任务太难 | 先在 Reach 任务跑通,再迁移抓取 |
| 真机炸机 | 关节限位 clamp;reset_pose 安全姿态;e-stop |

### Phase 1 — 代码完成,待真机验证
- 新建 `src/lerobot/rl/gym_manipulator_a10.py`:`A10RobotEnv`(关节空间,不依赖 `robot.bus`)
- 动作 7D [-1,1] 归一化 → env 内反归一化为绝对关节送 `SET_JOINTS`
- `make_a10_robot_env` / `make_a10_processors`(最小管线,无 IK)
- `gym_manipulator.py` 加 A10 分发(`make_robot_env`/`make_processors` 按 `robot.type` 路由,不破坏 SO100)
- 单测 `tests/rl/test_a10_robot_env.py` 通过(mock 机器人)
- **待真机验证**:env 连真机 reset/step 100 步稳定;π0.5 闭环跑通一个 episode
- **Phase 2 TODO**:`make_a10_processors` 的 action_steps 为空,需补 `JointInterventionProcessorStep`(干预时用 teleop_action 覆盖 7D 关节动作)

### Phase 2 — 事件适配完成,干预动作覆盖留 Phase 4
- `XLeVRTeleop` 新增 `get_teleop_events()`,映射 HIL-SERL 事件:
  - `IS_INTERVENTION` = 右手 squeeze 激活
  - `SUCCESS` = grip 按钮按下
  - `TERMINATE_EPISODE` = 左手摇杆右 (exit_early)
  - `RERECORD_EPISODE` = 左手摇杆左
- 单测 `tests/rl/test_xlevr_events.py` 通过
- **设计决策**:XLeVR 输出 EE delta,但 A10 残差 RL 动作空间是 7D 关节。干预时动作覆盖逻辑放 Phase 4 actor 接线阶段实现(干预时 actor 直接把 XLeVR EE delta 送 `SET_EE_DELTA`,并记录回读关节位置作为 buffer 中的 action),不在 processor 层做 IK 转换
- **待真机验证**:π0.5 闭环,人按 VR squeeze 能接管

### Phase 3 — 残差策略推理路径完成,训练集成留 Phase 4
- 新建 `src/lerobot/policies/residual_sac/`:
  - `ResidualSACConfig`:继承 `SACConfig` 复用全部 SAC 超参,新增 π0.5 连接 + α 残差缩放 + 关节限位
  - `ResidualSACPolicy`:继承 `SACPolicy`,覆盖 `select_action`(叠加 a_vla + α·Δa)与 `reset`(清 chunk buffer)
  - `_ChunkBuffer`:内联 chunk 逐步释放(不依赖 openpi `tree`)
  - `init_base_client()`:actor 启动时连 π0.5 websocket 服务
- 在 `policies/factory.py` 注册 `residual_sac`
- 单测 `tests/policies/test_residual_sac_policy.py` 通过:
  - 归一化(绝对关节→[-1,1])✅
  - 退化模式 select_action(a_vla=0,combined=α·Δa∈[-0.1,0.1])✅
  - reset ✅
- **Phase 4 TODO(训练集成)**:
  - `forward()` actor loss 改为最大化 `Q(s, a_vla_norm + α·Δa)`,a_vla detach
  - learner.py 需在 batch 中携带 `a_vla_norm`(从 transition 的 complementary_data 取)
  - critic loss 不变(buffer 存 combined action)
  - actor.py 接线:每步调 `policy.init_base_client()` + `select_action(batch, base_obs)`
- **待真机验证**:π0.5 服务在环时 select_action 返回合理 combined action

### Phase 4 — 接线规格完成,实现待真机集成

训练配置 `src/lerobot/configs/train_config_residual_pi05_a10.json` 已建。

#### actor.py 需要的修改 (4 处)

1. **import A10 + residual_sac**:`from lerobot.robots import a10_follower` (行 ~68 附近)
2. **policy 初始化后调 base client**:`policy = make_policy(...)` 后,若 `isinstance(policy, ResidualSACPolicy)`,`policy.init_base_client()`
3. **select_action 传 base_obs**:`act_with_policy` 内,每步构建 `base_obs`(state 7D + images 224² CHW + prompt)传给 `policy.select_action(batch, base_obs=base_obs)`
4. **干预分支**:检测 `teleop_events[IS_INTERVENTION]` 时:
   - 不调 `env.step`,改为 `robot.client.send_ee_delta(xlevr_ee_delta_7d)` 直接驱动
   - 回读关节 `robot.get_observation()` 作为 transition 的 action(归一化到 [-1,1])
   - 仍把 transition 推入 transitions_queue 进 buffer

#### learner.py 需要的修改 (1 处)

- `update_policy` 内 actor loss 计算:若 policy 是 ResidualSACPolicy,
  - 从 batch 取 `a_vla_norm`(complementary_data,detach)
  - actor 输出 Δa,构造 `combined = a_vla_norm + α·Δa`
  - actor loss = `-Q(s, combined).mean() + α_ent·log_prob(Δa)`
  - critic loss 不变(用 buffer 中的 combined action)

#### 启动命令 (真机集成时)

```bash
# 终端1: π0.5 服务 (openpi 仓库)
cd ~/Allen/openpi && uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_a10_finetune \
  --policy.dir=checkpoints/pi05_a10_finetune/Reach_5_9_1/130000

# 终端2: learner
python -m lerobot.rl.learner --config_path src/lerobot/configs/train_config_residual_pi05_a10.json

# 终端3: actor
python -m lerobot.rl.actor --config_path src/lerobot/configs/train_config_residual_pi05_a10.json
```

**待真机验证**:三进程跑通,wandb 显示 reward 上升 + intervention rate 下降。

## 六、进度记录

### Phase 0 — 代码完成,待真机验证
- 新建 `examples/dev_hil/phase0_pi05_openloop.py`
- 内联 `ChunkBuffer`(等价 openpi ActionChunkBroker,避免引入 `tree` 依赖)
- 图像预处理 / 动作映射 / ChunkBuffer 逻辑均通过单元验证
- 依赖:lerobot conda env + openpi-client(通过 sys.path 注入,不污染 lerobot 依赖)
- **待用户在真机上运行验证**:
  1. 在 GPU 机器起 π0.5 服务(见脚本头注释)
  2. `python examples/dev_hil/phase0_pi05_openloop.py --policy-host <GPU_IP> --policy-port 8000 --robot-host 192.168.1.12 --robot-port 8080 --steps 50 --hz 10`
- **验收标准**:超时比例 <10%,动作合理,无掉帧

