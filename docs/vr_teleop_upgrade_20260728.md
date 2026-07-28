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
- 闭环若 obs EE 有延迟，可加预测补偿。
- 夹爪摇杆斜率限制（用户反馈当前无影响，跳过）。

## 9. 第二轮优化（异步发送 + TCP_NODELAY）

在原方案基础上补做两项网络层优化，消除偶发 TCP 抖动对控制环的卡顿：

### 9.1 `a10_client.py`
- `connect` 时对 socket 设 `TCP_NODELAY`（关闭 Nagle，小指令立即发出）。
- 新增可选异步发送：`enable_async_send(send_timeout_ms=100)` 启动独立 daemon sender 线程，
  配 `deque(maxlen=1)` + `threading.Condition` 实现 **drop-oldest**（只保留最新 action，
  丢掉积压旧指令）；并设 `SO_SNDTIMEO`，保证 sender 不会无限期占用 `tx_lock`。
- `send_ee_delta` / `send_ee_target` / `send_action` / `sync_write`：异步模式开启时
  走 `_enqueue_send`（非阻塞入队），否则保持原同步行为（向后兼容）。
- 重连场景：`connect` 在 `_async_send=True` 时也会重新设 `SO_SNDTIMEO`，sender 线程
  复用同一 client 实例，自动衔接新 socket。

### 9.2 `teleoperate.py`
- 新增 `--no-async-send` 开关（默认开启异步发送）；机器人连接成功后调用
  `robot.client.enable_async_send(100)`。

### 9.3 验证
- One Euro 滤波单测：方波噪声 std 0.05 → 滤波后 0.011，收敛正常。
- 异步队列单测：5 个快速入队指令，慢 sender 只发 cmd0 与 cmd4（最新），中间 cmd1-3 被丢弃，符合预期。

### 9.4 行为说明
- 正常网络下发送 <1ms，sender 不积压，行为与同步一致。
- 网络抖动时：主环不再因 `send_action` 阻塞而卡顿（入队即返回）；最坏情况下
  `get_observation` 的 GET 请求可能等待 sender 释放 `tx_lock`，但被 `SO_SNDTIMEO=100ms` 上界限制。
- 丢帧对 `SET_EE_TARGET`（绝对目标）无害：下一帧 target 覆盖；对 `SET_EE_DELTA`（累加）
  会丢一个增量步，72Hz 下影响可忽略。

## 10. 第三轮修复（安全闭环复核）

上线前全局复核发现若干"首帧/松手/无 EE 反馈"边界下的安全隐患，统一修复：

### 10.1 处理器 `XLeVRTargetEEMapper`（`xlevr_processor.py`）
- **签名修正**：`RobotActionProcessorStep.__call__` 只把 `action` 传给 `action()`，原
  `action(self, action, obs)` 签名会导致 `TypeError`。改为 `action(self, action)`，obs 从
  `self._current_transition[TransitionKey.OBSERVATION]` 取。
- **`_reset_origins`（原 `_reset_origin`）**：松手时只清空 `robot_origin`/`vr_origin`，**保留**
  `_last_target_*`，使松手后机器人 hold 在上一次目标，而非回退到 (0,0,0)。
- **首帧空闲 hold**：未按下且 `_last_target_pos is None` 时，用 `obs` 的当前 EE 作为 hold 目标，
  避免首帧发出基坐标系原点 (0,0,0) 的危险目标。
- **滤波上提**：只要 VR 给出位姿就持续滤波（不再只在激活时滤波），保持滤波器温热，消除
  按下/松开瞬间的跳变；`vr_origin` 也用滤波后的位姿建立。
- **惰性 `vr_origin`**：若上升沿时 VR 位姿尚未就绪，在持续激活分支用当前帧补建 `vr_origin`，
  避免整段 hold。

### 10.2 `a10_follower.send_action` 配置一致性守卫
- 动作含 `ee.target_*` 但 `use_ee_target=False`（或含 `ee.delta_*` 但 `use_ee_delta=False`）时
  **抛 `ValueError`**，避免错配时落到关节全零指令（机器人冲向零位）。

### 10.3 `teleoperate.py` 启动探测 + 重连探测
- target 模式连接成功后先 `get_ee_state()`，若 `ee is None` → `SystemExit` 并提示刷固件或改用
  `--ee-delta`，从源头杜绝无 EE 反馈时发危险目标。
- 重连成功后同样探测；未通过则不置 `robot_link_ok`，继续等待，避免重连到无 EE 固件仍发动作。

### 10.4 `a10_client.get_ee_state` 读取超时
- 原复用连接时 300s 超时，VR plan 中途退出会让控制环挂死 5 分钟。改为本次读取临时设 2s 超时，
  `finally` 恢复原值；超时被 `ROBOT_LINK_ERRORS` 捕获 → 触发重连。

### 10.5 安全性结论
- VR plan 运行时：obs EE 恒有效 → `_last_target` 首帧即由 EE 填充 → 永不发出 (0,0,0)。
- VR plan 未运行时：`SET_EE_TARGET` 无消费者（仅 VR plan 消费），即便发出也无效，无害。
- 错配（target 动作 + delta 配置）：`send_action` 直接报错，不发送。

### 10.6 默认路径不回归（原始实机流程保持不变）
- `lerobot_record.record_loop`：`control_fps` 未显式提高时（`None` 或等于 `fps`）走原始
  "每帧都存、每帧都建 frame、按 `1/fps` 节拍" 逻辑，仅在高频控制时才按时间子采样。
- `a10_follower`：`use_ee_target=False`（默认）→ delta 特征/delta 发送/不调 `GET_EE_STATE`，与原始一致。
- `a10_client`：`enable_async_send` 未调用时（`lerobot-record` 路径）→ 同步发送；仅多 `TCP_NODELAY`（降尾延迟，非回归）。
- `factory`：`use_ee_target_mode=False`（默认）→ `XLeVRDeltaEEMapper`，参数与原始一致。
- 结论：不带任何新 flag 跑 `lerobot-record` 录制/推理，行为与改动前完全一致。

## 11. 性能优化（合并状态请求 + 多相机并行）

### 11.1 合并状态请求 `GET_STATE`（省一个 RTT/帧）
- target 模式原本每帧发 `GET_FOLLOWER_STATE`（关节）+ `GET_EE_STATE`（末端）两次同步往返。
- A10 端新增 `GET_STATE`：一次返回 `{"q":[7关节+夹爪], "ee":[6D]|null}`（`a10_tcp_server.hpp/cpp`）。
- Python 端 `a10_client.get_state()` 一次拿 q+ee；`a10_follower.get_observation` 在 target 模式改用
  `get_state()`，delta 模式仍用 `get_observation()`（关节），**默认路径不变**。
- 顺手修 `send_follower_state` 的潜在越界（resize 到 12 却访问 `[12]` → 改为 13）。
- 子模块指针更新至 A10_new `9a10d6e`。

### 11.2 多相机并行抓取
- `a10_follower.get_observation` 原顺序 `for cam in cameras: cam.async_read(...)`，N 路相机延迟累加。
- 改为持久 `ThreadPoolExecutor(max_workers=N)` 并行提交所有相机 `async_read`，再统一 `result()` 收集，
  总延迟 ≈ max(单路) 而非 sum(单路)。池在 `disconnect` 时 `shutdown`。
- 单相机时走同一路径，开销可忽略；无相机时不创建池。

### 11.3 收益与兼容
- 72Hz 控制环每帧预算 ~13.9ms：省一个 LAN RTT（~0.5–1ms）+ 多相机并行（两路省 ~5–15ms），
  显著降低 `get_observation` 占比，给控制环留更多裕度。
- delta 模式 / 无相机 / 不带新 flag：行为与改动前一致（默认路径不回归）。



