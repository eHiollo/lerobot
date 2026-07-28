# VR 遥操作升级：原点增量闭环 + 控制环/录制频率解耦 + 72Hz

> 分支：两端均在 `dev/vr_dev`
> 日期：2026-07-28
> 关联提交：
> - A10_new `91b1737` vr: 新增 GET_EE_STATE / SET_EE_TARGET 支持原点增量闭环
> - lerobot  `97d9800` vr: 原点增量闭环 + 控制环/录制频率解耦 + 72Hz

## 1. 背景与目标

原 VR 遥操作存在三个问题：

1. **频率被锁死**：`teleoperate.py` 默认 30Hz，`lerobot_record` 录制时控制环与数据集共用 `fps`，VR 设备可达 72Hz 的能力被浪费。
2. **帧间增量漂移**：`xlevr_to_ee_delta` 每帧发 `vr_curr - vr_prev`，A10 端 `SET_EE_DELTA` 累加进 `target_pm`。VR 静止时噪声被持续累加，机器人缓慢爬行。
3. **WebSocket vs UDP**：实测在局域网下 WebSocket 不是瓶颈，**保持现状不改**。

目标：
- 控制环顶到 72Hz，录制仍按 30Hz 子采样。
- 实现"按下 squeeze 以原点为参考的增量"（零漂移）。

## 2. 关键发现（决策依据）

读 A10_new 源码后确认：

- A10 RT 控制环 `k_dt = 0.002` → **500Hz**，接受 72Hz 指令无压力。
- `SET_EE_DELTA` 是**累加语义**：`target_pm = target_pm ∘ delta_pm`，内部 `SE3Follower` 闭环追踪 `target_pm`。
- A10 只通过 `GET_FOLLOWER_STATE` 回关节角，**不回 EE 位姿**；RT 内部已算 FK（`T_base_to_ee`）但未暴露到 TCP。
- 仓库无 URDF/DH（运动模型在机器人上的 `kaanh.xml`，aris 私有格式），Python 端无法自行 FK。

**结论**：在累加协议下，"原点增量"与"帧间增量"数学等价（`incremental_t = vr_t - vr_{t-1}`）。要实现真·零漂移原点增量，必须改 A10 C++，加两个指令：
- `GET_EE_STATE`：返回当前 EE 位姿 `pe=[x,y,z,rx,ry,rz]`。
- `SET_EE_TARGET`：直接替换 `target_pm`（不累加）。

Python 端按下 squeeze 时用 `GET_EE_STATE` 抓一次 `robot_origin`，之后每帧发 `SET_EE_TARGET robot_origin + scale·remap(vr_curr ⊖ vr_origin)`。VR 静止 → target 不变 → SE3Follower 收敛后停住 → 零漂移。

## 3. A10_new 改动（C++）

### 3.1 `a10_tcp_server.hpp/cpp`
- 新增 `update_ee_pose(pe)` / `get_ee_pose(out)` + `ee_pose_mutex_`/`current_ee_pe_`。
- 新增 `fetch_ee_target_if_updated` / `clear_ee_target_nrt` / `ee_target_seq` + `ee_target_mutex_`/`target_ee_absolute_`/`ee_target_seq_`。
- `process_line` 新增：
  - `GET_EE_STATE` → `send_ee_state` 回 `{"ee":[x,y,z,rx,ry,rz]}`（未更新过回 `{"ee":null}`）。
  - `SET_EE_TARGET` → 解析 7D `[x,y,z,rx,ry,rz,gripper]`，存入 `target_ee_absolute_`，`ee_target_seq++`。
- `clear_policy_tcp_targets_nrt` 一并清理 ee_target。

### 3.2 `a10_vr_plan.cpp`
- 新增 `pe_to_pm` 辅助（rotvec→R，compose_transform）。
- `Imp` 新增 `consumed_ee_target_seq_`。
- `executeRT` 中 FK 后 `pm2pe` → `update_ee_pose`（发布当前 EE）。
- ee_delta 消费块之后新增 ee_target 消费块：取出绝对目标 → `pe_to_pm` → 替换 `target_pm` → `follower.setTargetPm`。
- `prepareNrt` / `request_vr_teleop_stop` 清 ee_target。

> ⚠️ 编译需 aris/kaanh SDK，需在机器人控制器环境构建部署。

## 4. lerobot 改动（Python）

### 4.1 `a10_client.py`
- `get_ee_state()`：发 `GET_EE_STATE`，回 `{"ee": np.array(6)}` 或 `{"ee": None}`。
- `send_ee_target(actions)`：发 `SET_EE_TARGET {"actions":[7D]}`。

### 4.2 `config_a10_follower.py` + `a10_follower.py`
- 新增 `use_ee_target`（与 `use_ee_delta` 互斥）。
- target 模式下：
  - `get_observation` 额外 `get_ee_state`，注入 `ee.x/y/z/rx/ry/rz`。
  - `send_action` 处理 `ee.target_*` → `send_ee_target`。
  - `observation_features` / `action_features` 扩展。
- 旧 `use_ee_delta` 数据集 schema 不受影响。

### 4.3 `xlevr_processor.py`：`xlevr_to_ee_target`（核心）
- 内置精简 One Euro 滤波器（VR 位姿降噪，可关）。
- 按下 squeeze 上升沿：`robot_origin = obs EE`，`vr_origin = 当前 VR`，`last_target = robot_origin`。
- 持续激活：
  - `d_pos_vr_body = R(vr_origin)^-1 · (vr_curr - vr_origin)` → `remap` → `target_pos = robot_origin_pos + pos_scale · R_origin_robot · d_pos_robot_body`
  - `R_delta = R(vr_origin)^-1 · R(vr_curr)` → `remap` → `target_rot = R_origin_robot · R(angle_scale · rotvec_robot)`
  - 输出 `ee.target_*` + `gripper.pos` + `ee.enabled=True`
- 松开 squeeze：保持 `last_target`（hold），夹爪仍随摇杆更新，`ee.enabled=False`。
- 无 EE 反馈时退化为 hold，避免乱跑。

### 4.4 `config_xlevr.py` + `factory.py`
- 新增 `use_ee_target_mode`、`enable_filter`、`filter_min_cutoff`、`filter_beta`。
- `make_xlevr_a10_processors` 按开关选 `XLeVRTargetEEMapper` / `XLeVRDeltaEEMapper`。

### 4.5 `teleoperate.py`
- 默认 `FPS = 72`。
- 新增 `--ee-target` / `--ee-delta`；target 模式下机器人侧 `use_ee_target=True, use_ee_delta=False`。
- 打印当前模式与频率。

### 4.6 `lerobot_record.py`：频率解耦
- `DatasetRecordConfig` 新增 `control_fps: int | None`（None → 行为不变）。
- `record_loop` 控制环按 `control_fps` 全速跑（obs/action/send 每帧执行），数据集按 `fps` **基于时间戳子采样**保存。
- 两处 `record_loop` 调用均传 `control_fps=cfg.dataset.control_fps`。

## 5. 用法

### 纯遥操作（72Hz，原点增量）
```bash
python examples/xlevr_to_a10/teleoperate.py \
  --robot-host 192.168.1.12 --robot-port 8080 --ee-target
# 退回旧模式：--ee-delta
```

### 录制（控制 72Hz / 数据集 30Hz）
```bash
python src/lerobot/scripts/lerobot_record.py \
  --dataset.repo_id=allen/xxx --dataset.root=... \
  --dataset.fps=30 --dataset.control_fps=72 \
  --robot.type=a10_follower --robot.host=192.168.1.12 --robot.port=8080 \
  --teleop.type=xlevr --teleop.use_ee_target_mode=true \
  ...（其余参数同旧）
```

## 6. 部署清单

- [ ] A10_new `dev/vr_dev` 在机器人控制器环境编译部署（含 GET_EE_STATE/SET_EE_TARGET）。
- [ ] VR plan 启动后 `GET_EE_STATE` 才会返回非 null（EE 由 VR plan 的 FK 周期更新）。
- [ ] lerobot 端 `pip install -e .` 重新安装（新增 config 字段）。
- [ ] 首次 clone lerobot 后：`git submodule update --init --recursive`（拉 A10_new）。

## 7. 风险与回退

- **固件未更新却开了 `--ee-target`**：`get_ee_state` 收到 `{"ee":null}` 或连接异常，处理器退化为 hold（不乱跑），但无法移动；切回 `--ee-delta` 即可。
- **录制频率解耦**：`control_fps=None` 时与旧行为完全一致；设为 72 时若相机/磁盘跟不上，可能掉帧，建议先小规模试录检查 `dataset` 帧率。
- **One Euro 滤波**：默认开，若觉得跟手不够可调大 `filter_beta`，觉得抖可调小 `filter_min_cutoff`，或 `enable_filter=false` 关掉。
- **回退**：两端 `git checkout` 旧 commit 即可；submodule 回退用 `git submodule update`。

## 8. 仍可继续提升（未做）
- `send_action` 异步化（独立 sender 线程 + 丢最旧队列），消除偶发网络抖动卡主环。
- A10 TCP 确认 `TCP_NODELAY`。
- 夹爪摇杆加斜率限制/低通。
- 闭环若 obs EE 有延迟，可加预测补偿。
