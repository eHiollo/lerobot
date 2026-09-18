# XLeVR 遥操算法优化与开发交接

> 状态：保留 A1.1 输入防护和 A1.3 诊断；LeRobot A1.2 运动整形默认旁路，
> 由已验证的 A10 `vr_vel` 500 Hz 闭环统一负责限速、限加速度和顺滑控制
> 基线分支：kaanh
> 当前开发分支：kaanh_vr_op
> 最后更新：2026-09-17

本文档用于管理 XLeVR 到 A10 机械臂的遥操算法优化，同时作为后续开发的设计、配置和测试交接文档。

本文同时记录规划与实际开发；各小节和开发记录明确区分已实现、待验证与待商定内容。

## 1. 优化范围

目前暂定两个方向。

### A. 位置控制

在不重写已验证 A10 `vr_vel` 的前提下，将“相邻帧手柄增量”逐步替换为锚点目标生成。
LeRobot 负责输入防护和坐标映射，A10 负责反馈闭环、运动整形和 IK。

当前保留 A1.1/A1.3；A1.2 代码仅作显式回退。A2 和 B 按后文路径推进。

### B. 平移坐标系切换

计划支持：

- controller_body：手柄局部系，保留当前行为。
- vr_world：VR 固定空间，物理方向不随手腕姿态变化。
- robot_base：机器人基座系，需要 VR 到基座的标定及 EE 姿态反馈。

B 暂不实施。A 不应再次写死坐标变换，应预留 translation mapper 接口，等 B 商定后接入。

## 2. 当前系统基线

~~~text
WebXR/A-Frame 手柄绝对位姿
  -> XLeVR WebSocket/ControlGoal
  -> XLeVRTeleop 输出 xlevr.target_position/orientation_quat
  -> XLeVRDeltaEEMapper 计算相邻帧 body 增量
  -> ee.delta_[x,y,z,rx,ry,rz]
  -> A10Follower
  -> TCP SET_EE_DELTA（30 Hz）
  -> A10 TCP server 缓存最新包
  -> A10 vr_vel 累加 target_pm
  -> 实际关节 FK -> target/actual P 速度闭环
  -> 速度限幅 + 500 Hz acceleration slew
  -> command_pm -> IK -> 电机
~~~

当前平移公式：

~~~text
delta_world = p_current - p_previous
delta_body  = R(q_previous)^-1 * delta_world
delta_ee    = axis_remap * delta_body * scale
~~~

LeRobot到A10的目标更新是增量式的，但执行端不是完全开环：`vr_vel`每2 ms由实际关节
做FK，以`target_pm - actual_fk`生成P速度，再完成限速、slew和IK。当前缺口是目标生成
依赖逐帧累加，而不是由human anchor和robot FK anchor重算。

### 2.1 当前控制职责（权威划分）

| 层级 | 保留职责 | 不负责 |
|---|---|---|
| XLeVR/WebXR | 手柄pose、按键和源时间 | 机器人限速、IK |
| LeRobot bridge/teleop | 单调接收时间、sequence、age、统一使能 | 机器人运动整形 |
| LeRobot mapper | finite/四元数、stale、重复帧、时间回退、跳点、重锚、坐标映射、scale/死区、诊断 | 默认不做低通、速度/加速度限制和起步ramp |
| A10 TCP | 接收并提供最新`SET_EE_DELTA` | VR坐标解释 |
| A10 `vr_vel` | target、实际FK、P控制、线/角速度与加速度管理、idle pull、IK和电机命令 | WebXR输入检查 |
| LeRobot recorder | observation/action/video及JSONL旁路 | 改写控制命令 |

正常实机使用`motion_shaping_mode=robot_controller`，运动整形只发生在A10。旧A1.2仅在
`motion_shaping_mode=lerobot_a1`时启用，用于离线对比或回退，避免30 Hz与500 Hz双重整形。

现有优点：

- 松开后清除参考，重新按下的第一帧为零，不因 VR 绝对位置不同而跳变。
- 使用四元数和 rotvec，没有直接做欧拉角差分。
- 轴变换矩阵属于 SO(3)，平移和旋转使用一致换基。
- 已有位置/角度 scale、死区和可选单帧限幅。

已知缺口：

- human anchor + robot FK anchor目标生成尚未实施。
- LeRobot拿不到A10内部末端FK、tracking error、IK失败和限幅状态。
- max_delta_pos_m 和 max_delta_angle_deg 默认关闭。
- 外部 XLeVR 与 LeRobot 都有 vr_to_robot_scale，可能意外叠乘。
- 实际控制频率由应用/录制循环决定，teleop.control_fps 不是唯一真实来源。

### 2.1 Pipeline 输出与 A10 线协议

必须区分三层数据：

| 层级 | 数据类型 | 是否发给 A10 |
|---|---|---|
| XLeVR 原始 action | Python 字典，包含位置、四元数和按键 | 否 |
| Pipeline 输出 | Python RobotAction 字典 | 不直接发送 |
| A10 TCP 负载 | 7 个浮点数的 JSON 数组 | 是 |

当前 Pipeline 的核心输出为：

~~~python
{
    "ee.enabled": bool,
    "ee.delta_x": float,
    "ee.delta_y": float,
    "ee.delta_z": float,
    "ee.delta_rx": float,
    "ee.delta_ry": float,
    "ee.delta_rz": float,
    "gripper.pos": float,
}
~~~

A10Follower.send_action() 只取六维 EE 增量和夹爪，转为：

~~~text
SET_EE_DELTA {"actions": [dx, dy, dz, rx, ry, rz, gripper]}
~~~

前 3 维按当前客户端契约为 EE-local 位置增量，单位 m；中间 3 维为 EE-local rotvec，单位 rad；最后 1 维为夹爪。ee.enabled=false 不单独发送给 A10，LeRobot 会将前 6 维置零后继续发送同样的 7 维负载。vr.* 调试字段不会进入 TCP。

### 2.2 位置算法修改时的协议兼容原则

A1 和 A2 默认保持上述 7 维 SET_EE_DELTA 外部契约。位置算法可以从相邻帧差分改成滤波、reference governor 或 FK 闭环，但最终 output adapter 仍输出“本控制周期应执行的 EE-local 位姿增量”。在此前提下：

- A10 的接收字段、JSON 结构和解析代码不需要修改。
- 改变的是每帧数值的计算来源，不是线上数据类型。
- A2 若经 TCP 获取 EE pose，只需扩展机器人到 LeRobot 的反馈协议；原 SET_EE_DELTA 下行负载仍可保持。
- 必须用旧 A10 服务端做协议回归，确认长度、顺序、单位和有限数均符合原契约。

以下任一变化都会破坏线协议，必须同步适配 A10 接收端，并引入明确协议版本：

- 把 dx/dy/dz 从每周期位移增量改成速度或绝对位置。
- 把参考系从 EE-local 改成 robot-base 或 VR-world，却继续复用原字段名。
- 改变 m、rad 或夹爪的单位与范围。
- 改变 7 维顺序、长度，或增加 A10 必须解析的字段。
- 让 A10 改为接收绝对 EE pose，例如新命令 SET_EE_POSE。

如需这些改动，应新增命令名或 protocol_version，不能在同一个 SET_EE_DELTA 名称下静默改变语义。在 A0 完成接收端确认前，EE-local、m、rad 仍只是依据当前客户端代码得到的待实机验证契约。

## 3. A：位置控制目标架构

位置控制按运行位置拆成四层：

~~~text
1. LeRobot Input guard
   有效性、时间戳、stale、跳点检查

2. LeRobot Translation mapper（由 B 管理）
   VR 位移 -> 选定的机器人参考系

3. A10 Position reference/controller
   robot anchor、target、实际FK误差、防积压

4. A10 Command governor（500 Hz）
   P速度、线/角速度限幅、线/角加速度slew、IK
~~~

### 3.1 控制状态机

| 状态 | 行为 |
|---|---|
| IDLE | EE 六维输出为零，不保留运动目标 |
| ARMING | 检查新鲜度并捕获VR参考，本帧输出零 |
| ACTIVE | 输出受输入防护的EE-local增量，由A10整形执行 |
| STALE | VR 超时，立即输出零并废弃锚点；恢复后必须重新建锚 |
| FAULT | 非有限数、异常跳变或跟踪误差超限；输出零并记录原因 |

输入层应生成唯一 control_active。位置控制器不应再分别解释 xlevr.enabled、squeeze、grip_active 和 trigger 阈值。

### 3.2 输入契约与防护

每个 VR 样本至少包含：

~~~text
position[3]
orientation_quat[4]
source_timestamp
receive_monotonic_timestamp
sample_sequence
control_active
~~~

检查顺序：

1. 维度和类型正确。
2. 所有分量为有限数。
3. 四元数范数合理，再归一化。
4. 数据 age 不超过 stale_timeout_s。
5. 使用实际 dt 检查人手速度，而非只检查单帧距离。
6. 检查 WebXR 重定位、时间戳回退和中断恢复后的突变。

跳点处理原则：

- 超大跳点不能简单裁成最大动作，否则错误输入仍会驱动机器人。
- 单个异常样本先保持上一目标。
- 连续样本证明是真实快速移动时，仍经过 reference governor 和速度限制。
- WebXR 重定位或连续异常时进入 FAULT/ARMING，不把新坐标与旧锚点相减。

阈值必须由 A10 能力、VR 日志和实机低速试验标定，不直接复制参考项目的 0.5 m 常量。

### 3.3 双锚点、路径与防积压

control_active 上升沿且输入有效时：

~~~text
human_anchor = current_vr_position
robot_anchor = current_robot_ee_position_from_fk
filtered_displacement = 0
reference_position = robot_anchor
previous_command = 0
~~~

持续控制时：

~~~text
human_displacement = current_vr_position - human_anchor
mapped_displacement = translation_mapper(human_displacement, ...)
user_target = robot_anchor + position_scale * mapped_displacement
~~~

user_target 不是一次性远端终点，而是每个 VR 样本生成的新路径点。两个样本之间的真实曲线无法凭空恢复，因此仍要求稳定采样率、时间戳和必要插值。

不能把未受限 user_target 直接交给机器人，否则人手移动过快时会积压很长的追赶路径。必须增加 reference governor：

~~~text
reference_step = clamp_norm(
    user_target - reference_position,
    max_reference_speed_m_s * dt,
)
reference_position += reference_step
~~~

然后用当前 FK 计算闭环误差：

~~~text
error_base = reference_position - current_robot_ee_position
error_ee = R_base_ee.T * error_base
~~~

如果 error_base 范数超过 max_tracking_error_m，不应继续积压目标。候选策略是冻结 reference、输出受限降速指令或进入 FAULT，并要求重新使能。不得让“排队的未执行路程”在松手后继续执行。

### 3.4 当前方案：LeRobot 防护，A10 闭环整形

当前 LeRobot 不需要重复实现 FK 闭环：已验证的 A10 `vr_vel` 使用机器人实际关节 FK，
并在 500 Hz 控制环内完成目标跟踪和运动整形。

#### A1：LeRobot 输入安全层

保持 SET_EE_DELTA 和现有坐标行为，先实现：

- 状态机和单一 control_active。
- 非有限数、stale、跳点和时间戳检查。
- 连接中断和 processor reset 的显式行为。

A1.1 负责阻止非法、过期、重复或跳变输入；A1.3 负责记录诊断。A1.2 的 Python
滤波、速度/加速度限制和起步渐变仅保留为显式回退模式，正常运行默认旁路。

#### A2：可选的跨端反馈增强

反馈路径二选一：

1. 优先扩展 A10 TCP，由控制器直接回传 T_base_ee。
2. 或引入 A10 URDF，明确关节顺序、零位、正方向和单位，复用 RobotKinematics.forward_kinematics。

只有需要 LeRobot 获知 A10 实际跟踪误差时，才考虑扩展 EE pose 反馈，并实现双锚点、
reference governor 和跟踪误差监控；不得与 A10 现有闭环职责重复。

A2 的关键验收：人为丢弃或裁剪一帧命令后，后续帧能根据 FK 渐进纠偏，且不会超限追赶。

### 3.5 历史 A1.2 低通方案（默认旁路）

以下算法仅供 `motion_shaping_mode=lerobot_a1` 回退或离线对比。正常模式由 A10
`vr_vel` 完成顺滑，避免 30 Hz LeRobot 与 500 Hz A10 双重滤波。

优先对累计位移/reference 做二阶低通，而不是只过滤单帧 delta。锚点位移始终可由当前 VR 位置相对初始位置重算，不会把单帧噪声永久积分进去。

可先使用二阶 Butterworth，每轴独立保存状态：

~~~text
fc = min(configured_cutoff_hz, 0.95 * fs / 2)
fs = 1 / measured_dt
~~~

当前控制约 30 Hz，不能照搬 100 Hz 参考系统的 12/20 Hz 参数。纸面起始范围：

~~~text
position_cutoff_hz: 4-6 Hz
~~~

最终值必须用实际 VR 轨迹和机械臂响应调参。重锚、stale、fault 或重连时均重置滤波器。

### 3.6 历史 A1.2 限速与渐变（默认旁路）

以下限制同样只属于 `lerobot_a1` 模式；正常实机模式使用 A10 `vr_vel` 已验证的速度、
加速度和目标跟踪逻辑。

限制应基于实际 dt 和向量范数：

~~~text
delta_max = max_linear_speed_m_s * dt
delta_cmd = clamp_norm(delta_unlimited, delta_max)
~~~

加速度限制：

~~~text
velocity_delta = velocity_cmd - previous_velocity_cmd
velocity_delta = clamp_norm(velocity_delta, max_linear_accel_m_s2 * dt)
velocity_cmd = previous_velocity_cmd + velocity_delta
delta_cmd = velocity_cmd * dt
~~~

启用后的 engage_ramp_s 内乘平滑增益：

~~~text
r = clamp(active_elapsed / engage_ramp_s, 0, 1)
gain = r * r * (3 - 2 * r)
delta_cmd *= gain
~~~

参考系统约 2 s 的渐变对 A10 可能过慢，台架初始试验范围建议 0.3-0.5 s。最大速度、加速度和跟踪误差在确认 A10 能力前不写死，实机从低值开始。

### 3.7 控制频率与路径连续性

- 使用源时间戳或 monotonic 接收时间，不假设恒定 30 Hz。
- 样本间隔超过阈值时停止并重锚，不跨越空白区间插值。
- 若 A10 支持，控制循环可提高到 60 Hz，数据集仍保持 30 Hz。
- 控制与相机读取/视频写入解耦，避免 I/O 卡顿变成路径突变。

## 4. A 的分阶段实施路径

### A0：基线与协议确认

- 记录不连接机械臂的 VR 原始位姿与时间戳。
- 确认 SET_EE_DELTA 的单位、参考系、积分方式、IK 失败、限速和插补策略。
- 确认 GET_FOLLOWER_STATE.q 的顺序、单位、零位和回传频率。
- 测量 VR 源频率、控制频率、延迟、抖动、丢帧和重定位特征。
- 建立旧版输入/输出回放样例。

产出：协议记录、VR 回放样例、旧算法指标和安全边界。

### A1.1：数据契约和状态机（已实现）

- bridge 兼容保留 source timestamp，并增加 monotonic receive time。
- 向 mapper 传递 sample_age_s 和 sample_sequence。
- 建立 IDLE/ARMING/ACTIVE/STALE/FAULT 状态机。
- 统一 squeeze、trigger 和 xlevr.enabled 语义。

本次实现以 `pose_receive_monotonic_s` 作为安全判断的权威时间。浏览器虽然发送
`Date.now()`，但当前仓库外 `XLeRobot/XLeVR/xlevr/inputs/vr_ws_server.py` 尚未把它写入
`ControlGoal.metadata`；bridge 已兼容 `timestamp/source_timestamp`，外部服务完成透传后无需
修改 mapper。按键类 partial goal 只能更新 goal 时间，不能刷新 pose 时间和 pose 序号。

验收：按下、松开、浏览器卡住、WebSocket 断开和恢复均不产生非零突变。

### A1.2：安全整形（保留代码，默认旁路）

- finite、四元数、时间戳和跳点检查归入 A1.1，继续默认启用。
- 低通、速度/加速度限制和起步渐变保留在 `lerobot_a1` 模式。
- 默认 `robot_controller` 模式直接输出受防护的坐标增量，由 A10 `vr_vel` 整形。

验收：默认路径不执行 LeRobot 运动整形；显式回退模式的原测试继续通过。

### A1.3：记录与台架调参（已实现代码，待台架/实机验收）

- mapper 每帧生成 schema version 1 诊断快照，记录原始 VR 位姿/按键、接收时间、age、
  sequence、控制状态、过滤原因、滤波前后速度、限幅标志和最终命令。
- `lerobot_record` 默认将快照写到
  `<dataset_root>/meta/xlevr_diagnostics/session_*.jsonl`，并补充 recording phase、
  episode/control frame index 和 A10 adapter 返回的 `sent_action`。
- JSONL 使用有界后台队列，控制线程只做内存快照和非阻塞入队；队列满时丢弃诊断记录，
  不阻塞机器人控制。退出日志报告 written、dropped 和 writer error。
- 训练 action schema 只保留 `ee.enabled`、6 个 `ee.delta_*` 和
  `gripper.pos`；原始 VR 进入 metadata sidecar，不再伪装成机器人动作。
- 每次 initial reset、recording、reset/re-recording 进入 `record_loop` 时 reset
  teleop/robot processor，禁止跨阶段继承 VR 锚点、滤波器和命令速度。
- `examples/xlevr_to_a10/teleoperate.py` 新增
  `--diagnostics-jsonl`，支持先做 `--vr-only` 日志采集。

过滤原因枚举：

| guard_reason | 含义 |
|---|---|
| idle / legacy_idle | 未请求手臂控制 |
| arming | 首帧或恢复帧只重锚 |
| active / legacy_active | 本帧进入正常映射 |
| duplicate_sample | 控制循环重复读取同一 VR pose，不重复积分 |
| stale_age / sample_gap | 样本当前已过期或相邻 pose 间隔过长 |
| invalid_sample | 位置、四元数、trigger 或时间/序号非法 |
| sequence_rollback / timestamp_regression | 序号或单调接收时间回退 |
| missing_reference | 内部参考缺失，fail closed |
| spike_pending / spike_reanchor | 跳点等待连续确认 / 已在新位置零输出重锚 |

代码验收标准已满足：单帧可还原为何放行、过滤、限幅或禁用；真实 VR、模拟 TCP 和 A10
低速无负载调参仍未执行，不得据此把暂定参数视为实机验收值。

### A2.1：EE 位姿反馈

- 决定由 A10 回传 EE pose，还是由 LeRobot 用 URDF + FK。
- 建立坐标系、单位、时间戳和有效性契约。
- 禁止使用过期 FK 快速纠偏。

验收：多个静态姿态下，LeRobot 与 A10 对 EE 位姿的计算在约定容差内一致。

### A2.2：锚点闭环和防积压

- 捕获 human/robot 双锚点。
- 实现 user target、reference governor、FK error 和 EE-local command。
- 实现 max_tracking_error_m 与 freeze/fault/re-anchor 策略。
- 添加丢指令、裁指令和 IK 失败测试。

验收：开环误差可被反馈纠正；快速手动或机器人堵转时不积压危险目标。

### A3：控制/录制解耦（可选）

- A10 和网络稳定支持时，控制循环提高到 60 Hz 或实测可靠频率。
- 数据集可继续按 30 Hz 取样。
- 明确快速控制帧与数据集帧的对齐方式。

## 5. B：平移坐标系切换（待商定）

### controller_body

优点是不需要基座标定、适合沿工具轴插入抽出、接近 EE-local 接口。缺点是相同物理路径会因手腕姿态不同而生成不同机器人路径。

### vr_world

物理上、前、右不随手柄姿态变化。待决定 VR 原点和 yaw 如何建立、是否使用 headset yaw，以及头显重定位后的重锚策略。

### robot_base

便于闭环目标、工作空间约束和轨迹可视化，但要求已知 R_base_vr、T_base_ee，并确认 A10 命令参考系。

### 预留接口

~~~python
mapped = translation_mapper.map_displacement(
    human_displacement=...,
    controller_orientation=...,
    robot_ee_pose=...,
    calibration=...,
)
~~~

输出参考系必须由类型或字段名明确表达，不能只靠注释区分 base、world 和 EE-local。

## 6. 配置设计

### 6.1 建议配置

| 配置 | 用途 | 初始策略 |
|---|---|---|
| position_control_mode | safe_frame_delta / legacy_frame_delta；后续再加 anchored 模式 | 本次默认 safe，可显式回退 legacy |
| translation_frame | controller_body / vr_world / robot_base | B 尚未实施，当前保持 controller_body |
| stale_timeout_s | VR pose 样本最大 age 和最大间隔 | 暂定 0.25 s，待日志标定 |
| max_vr_speed_m_s | VR 跳点判定 | 暂定 2.0 m/s，待日志标定 |
| spike_recovery_frames | 新位置连续稳定多少帧后重锚 | 暂定 2 帧 |
| motion_shaping_mode | 运动整形责任方 | 默认 robot_controller；lerobot_a1 仅回退 |
| position_cutoff_hz | 位置速度二阶级联低通 | 仅 lerobot_a1，5.0 Hz |
| max_linear_speed_m_s | 末端命令向量速度上限 | 仅 lerobot_a1，0.25 m/s |
| max_linear_accel_m_s2 | 末端命令向量加速度上限 | 仅 lerobot_a1，1.0 m/s² |
| engage_ramp_s | 起步渐变 | 仅 lerobot_a1，0.35 s |
| max_tracking_error_m | 防目标积压 | A2 上线前标定 |
| fault_recovery_mode | 后续可配置 require_reengage / auto_reanchor | 本轮固定为零输出重锚；跳点需稳定帧确认 |
| record_vr_diagnostics | 保存 JSONL sidecar | 默认 true；可用 CLI 关闭 |
| diagnostics_queue_size | 诊断后台队列容量 | 默认 2048；满时丢诊断而不阻塞控制 |
| diagnostics_flush_every | 后台写盘 flush 间隔 | 默认 30 条 |

配置原则：

- 只保留一个权威 VR 位置比例，外部 XLeVR 默认传原始米制坐标。
- 区分 VR 源采样率、机器人控制率和数据集采样率。
- 参数名携带单位，如 _m_s、_m_s2、_hz 和 _s。
- 不把机器人 IP、本机绝对路径、repo id 和实验时长做成库内强制默认。
- 对旧数据集记录算法/配置版本，禁止无元数据混合新老动作定义。

### 6.2 交接前必须确认

- A10 是否把 SET_EE_DELTA 解释为 EE-local？
- dx/dy/dz 是每帧位移、速度命令，还是固定周期增量？
- A10 是否已有工作空间、速度/加速度限制、插补和 IK 失败反馈？
- A10 能否回传 EE pose 与命令接受/拒绝状态？
- GET_FOLLOWER_STATE.q 是度还是弧度，夹爪是否为第 7 维？
- 允许的末端线速度、加速度和跟踪误差是多少？
- trigger 是精调还是夹爪？外部 XLeVR 注释与 LeRobot 语义不完全一致。
- 原始 VR 位姿是否进入数据集，以及归入 observation、metadata 还是外部日志？

## 7. 项目文件与修改清单

### 7.1 LeRobot 内的 VR 核心

| 路径 | 作用 | A 预计动作 |
|---|---|---|
| src/lerobot/teleoperators/xlevr/teleop_xlevr.py | 读取 ControlGoal，生成 xlevr.* | 传递时间/序号，统一使能 |
| src/lerobot/teleoperators/xlevr/vr_monitor_bridge.py | 服务桥接、goal 合并、stale 统计 | 保留时间戳并暴露新鲜度 |
| src/lerobot/teleoperators/xlevr/xlevr_processor.py | VR 位姿到 EE delta | 状态机、防护、滤波、限速、渐变、锚点 |
| src/lerobot/teleoperators/xlevr/quaternion_utils.py | 四元数、body delta、轴变换 | A 仅加数学工具；坐标改动归 B |
| src/lerobot/teleoperators/xlevr/config_xlevr.py | 遥操配置 | 增加位置配置，整理 scale/fps |
| src/lerobot/teleoperators/xlevr/diagnostics.py | A1.3 JSONL 旁路 | 非阻塞入队、严格 JSON 转换、写入统计 |
| src/lerobot/teleoperators/xlevr/factory.py | 构造 pipeline | 传入新配置并按模式构造 |
| src/lerobot/teleoperators/xlevr/vr_events.py | 左摇杆转录制事件 | 通常不改，但需回归 |
| src/lerobot/teleoperators/xlevr/README.md | 现有用法和坐标说明 | 实施后更新新老模式 |
| src/lerobot/teleoperators/xlevr/__init__.py | 对外导出 | 仅新增公共类时修改 |

### 7.2 机器人、录制和示例

| 路径 | 作用 | 预计动作 |
|---|---|---|
| src/lerobot/robots/a10_follower/a10_client.py | A10 TCP | A2 可能增加 EE pose/命令状态 |
| src/lerobot/robots/a10_follower/a10_follower.py | Robot 适配 | A2 可能在 observation 暴露 EE pose |
| src/lerobot/robots/a10_follower/config_a10_follower.py | A10 配置 | 可能增加反馈配置并清理硬编码 |
| src/lerobot/model/kinematics.py | 通用 URDF FK/IK | 优先复用 |
| src/lerobot/scripts/lerobot_record.py | 正式录制 | 诊断数据、processor reset、频率解耦 |
| src/lerobot/scripts/lerobot_teleoperate.py | 通用遥操入口 | 新 pipeline 配置和控制率 |
| examples/xlevr_to_a10/teleoperate.py | 专用遥操及 --vr-only | 新配置、故障注入和状态打印 |
| examples/xlevr_to_a10/test_action_print.py | 打印映射结果 | 展示滤波、限幅和 fault |

### 7.3 测试

| 路径 | 动作 |
|---|---|
| tests/teleoperators/test_xlevr_axis_remap.py | 保留；B 实施时扩展 |
| tests/teleoperators/test_xlevr_position_control.py | 已新增，覆盖 A1 状态机和安全整形 |
| tests/teleoperators/test_xlevr_bridge_freshness.py | 已新增，覆盖时间戳、序号和 partial goal freshness |
| tests/teleoperators/test_xlevr_diagnostics.py | 已新增，覆盖 JSONL drain、严格 JSON 和快照复制 |
| tests/robots/test_a10_ee_delta.py | 计划新增，测试禁用清零、单位和 EE pose 协议 |

必须覆盖：

- 首次按下和持续静止输出零。
- 松开清理锚点、滤波和速度状态，重新按下无跳变。
- NaN、Inf、零四元数、时间戳回退进入安全输出。
- stale 输出零，恢复后不跳变。
- 单帧跳点被拒绝，真实快速运动受 governor 限制。
- 对角线运动不突破向量上限。
- 滤波器重锚后无旧状态尾巴。
- 不同帧率和抖动 dt 下速度上限一致。
- A2 中命令丢失或被裁剪后能纠偏且不追赶过冲。

### 7.4 仓库外部 XLeVR

当前默认依赖：

~~~text
/home/robot/VLA/XLeRobot/XLeVR
~~~

| 路径 | 作用 | 预计动作 |
|---|---|---|
| web-ui/vr_app.js | 采集 A-Frame 位姿和按键 | 可能增加稳定序号和 WebXR 时间戳 |
| xlevr/inputs/vr_ws_server.py | 解码并生成 ControlGoal | 保留时间、取消重复 scale、清理旧语义 |
| xlevr/config.py、config.yaml | 网络、TLS、VR scale | 默认传原始米制位姿 |

kaanh_vr_op 只能管理 LeRobot 内修改，不会包含外部目录。开发前须选择：

1. 在 XLeRobot/XLeVR 建立对应分支并记录 commit id；或
2. 将必要接入层纳入本项目或子模块；或
3. 外部依赖不改，由 LeRobot bridge 补充 monotonic 时间和安全检查。

不应依靠复制某台机器上未版本化的 XLeVR 目录复现系统。

## 8. 分支与开发交接

### 8.1 分支创建记录

2026-09-14 已从 `kaanh` 创建并切换到 `kaanh_vr_op`，创建时的 HEAD 为
`fffffae5fd7337cfd59c9fd432f65b65e6641b81`（`遥操作完全正常可用`）。

分支创建时 working tree 已有未提交内容，因此新分支继承了这些本地改动，而不是从完全
clean 的工作区开始。本次没有替用户提交、丢弃或覆盖既有改动，也没有创建 Git commit。
审核时应按 11.6 的文件归属清单区分本次 A1 修改与继承修改。

### 8.2 建议提交拆分

1. docs: add XLeVR position-control design and handoff
2. test: add replay fixtures for legacy XLeVR mapping
3. feat: propagate XLeVR timestamps and freshness
4. feat: add XLeVR control state machine and input guards
5. feat: add rate-aware position filter and motion limits
6. feat: add XLeVR diagnostics to recording
7. feat: expose A10 end-effector pose（若选择 TCP 路线）
8. feat: add anchored FK position control
9. feat: add selectable XLeVR translation frames（B 商定后）

不要在一个提交中同时修改数据集 schema、A10 TCP、坐标映射和底层限速。

### 8.3 合并请求交接内容

- 基线 commit 和外部 XLeVR commit。
- 机器人控制器、VR 设备和浏览器版本。
- 控制率、VR 样本率和数据集 fps。
- 新参数表与调参依据。
- legacy 回归、离线回放、模拟 TCP 和实机结果。
- 安全环境、急停方式和未解决风险。
- 数据集 schema 是否变化及兼容方法。

## 9. 决策记录

| 编号 | 问题 | 当前结论 | 状态 |
|---|---|---|---|
| D-001 | 是否直接替换旧位置逻辑 | 默认 safe_frame_delta，保留 legacy_frame_delta 回退和对比 | 已实现 |
| D-002 | 是否引入 stale/跳点/有限数防护 | 必须引入 | 已定 |
| D-003 | 运动整形由谁负责 | 默认由 A10 `vr_vel` 负责；LeRobot A1.2 仅显式回退 | 已定 |
| D-004 | A2 使用 TCP EE pose 还是 URDF FK | 未决定 | 待商定 |
| D-005 | 默认平移坐标系 | 暂保持 controller_body | B 待商定 |
| D-006 | 控制率是否与录制 fps 解耦 | 有价值，先测 A10 吞吐 | 待商定 |
| D-007 | 外部 XLeVR 如何版本化 | 本次不改外部目录，由 bridge 提供接收时间；源时间仍待透传 | 部分完成 |
| D-008 | 原始 VR 位姿是否进数据集 | A1.3 放入 dataset metadata JSONL sidecar，不进入训练 action | 已实现 |
| D-009 | A 优化是否修改 A10 下行协议 | A1 保持 7 维 SET_EE_DELTA；语义变化时必须版本化并同步适配 A10 | A1 已确认 |

## 10. 第一个开发迭代的边界

`kaanh_vr_op` 当前边界：

- 保持 `controller_body` 映射、姿态算法和 7 维 `SET_EE_DELTA` 协议不变。
- A1.1 输入防护默认启用。
- A1.3 JSONL 诊断和录制阶段 reset 保留。
- A1.2 运动整形代码保留，但默认旁路；仅 `lerobot_a1` 显式启用。
- A10 `vr_vel` 独占正常模式下的 FK、目标跟踪、限速、加速度约束和顺滑。
- 本轮不实施 A2 跨端反馈和 B 坐标系切换。

线协议与训练 action schema 均不因本次职责调整而变化。

## 11. 2026-09-14 A1.1/A1.2 历史开发记录

本节保留当时实现记录。当前以第 2、4、9、10、14 节为准：A1.1 保留，A1.2 默认旁路，
A1.3 保留。

### 11.1 本次结果和范围

- 开发分支：`kaanh_vr_op`。
- 基线 HEAD：`fffffae5fd7337cfd59c9fd432f65b65e6641b81`。
- 已实现：pose 单调接收时间/序号、统一使能、状态机、输入防护、跳点恢复重锚、
  二阶位置速度低通、向量速度限制、向量加速度限制、smoothstep 起步渐变和 processor reset。
- 保持不变：controller_body → EE-local 轴映射、姿态增量算法、m/rad 单位、夹爪含义、
  A10 `SET_EE_DELTA` 命令名及 7 维顺序。
- 此记录完成后又实施了 A1.3；A2 FK/锚点闭环、B 平移坐标系切换和 A10 协议扩展仍未实施。
- 未修改仓库外 `/home/robot/VLA/XLeRobot`；其浏览器源时间戳透传仍是后续工作。

### 11.2 状态机和恢复语义

| 状态 | 进入条件 | 本帧手臂输出 | 恢复方式 |
|---|---|---|---|
| IDLE | `xlevr.enabled=false` 或 reset | disabled + 六维零增量 | 下一有效使能帧进入 ARMING |
| ARMING | 首次使能或跳点确认后重锚 | disabled + 六维零增量 | 收到下一新鲜 pose 后进入 ACTIVE |
| ACTIVE | 连续、有效且新鲜的 pose | 输出受整形的增量 | 松开、stale 或 fault 后退出 |
| STALE | age 超阈值或相邻 pose 间隔超阈值 | disabled + 六维零增量 | 新鲜 pose 先重新 ARMING，不跨空白补轨迹 |
| FAULT | 非有限数、无效四元数、时间回退或待确认跳点 | disabled + 六维零增量 | 普通输入错误需新鲜帧重锚；跳点需连续稳定帧确认 |

单帧跳点不会生成锚点到远端之间的未知轨迹。实现会先停用手臂并保存候选位置；候选轨迹连续
稳定达到 `spike_recovery_frames` 后，只在新位置重锚且仍输出零，下一帧才恢复 ACTIVE。因此
“远距离空白段”被明确丢弃，不会插值，也不会让机器人追赶这段距离。

### 11.3 A1.2 位置命令计算顺序

对每个新的 pose 样本执行：

1. 检查 position/quaternion/trigger/time/age/sequence 的有限性和合法性。
2. 用 pose sequence 去重，避免控制循环对同一 VR 样本积分两次。
3. 用 monotonic receive time 的实际差值计算 `dt`，而不是假设固定 30 Hz。
4. 以 VR 位移范数除以 `dt` 检测 tracking jump。
5. 沿用旧算法，将世界位移旋转到上一帧 controller body，再使用原 `axis_remap` 映射到 EE-local。
6. `desired_velocity = mapped_delta / dt`，经过两个一阶极点级联形成二阶低通。
7. 对速度使用三维向量范数限幅，保持对角线方向，不再用逐轴裁剪形成“方盒”上限。
8. 施加 smoothstep 起步增益，再按 `max_linear_accel_m_s2 * dt` 限制速度向量变化量。
9. `delta_cmd = limited_velocity * dt`，最后应用现有位置死区。

本轮没有 FK，因而低通对象是映射后的 EE-local 位置速度；A2 引入机器人 EE 反馈后，再把滤波
迁移到累计 reference/目标位置并评估 Butterworth，以避免把不同 EE 姿态下的局部速度状态混用。

### 11.4 配置、默认值和回退

新增配置全部位于 `XLeVRTeleopConfig`，并由 `factory.py` 显式传入 mapper：

| 参数 | 本次默认值 | 备注 |
|---|---:|---|
| position_control_mode | safe_frame_delta | 使用 A1 安全路径 |
| stale_timeout_s | 0.25 s | 同时约束当前 age 和相邻 pose 间隔 |
| max_vr_speed_m_s | 2.0 m/s | 输入跳点阈值，不是机器人速度上限 |
| spike_recovery_frames | 2 | 连续稳定后只重锚，不补空白轨迹 |
| motion_shaping_mode | robot_controller | 默认由 A10 整形；`lerobot_a1` 才启用下列参数 |
| position_cutoff_hz | 5.0 Hz | 仅 `lerobot_a1`：两极点级联低通 |
| max_linear_speed_m_s | 0.25 m/s | 仅 `lerobot_a1`：平移速度上限 |
| max_linear_accel_m_s2 | 1.0 m/s² | 仅 `lerobot_a1`：平移加速度上限 |
| engage_ramp_s | 0.35 s | 仅 `lerobot_a1`：smoothstep 起步渐变 |

这些是用于离线和台架起步的暂定值，不代表 A10 厂商安全额定值。连接实机前必须从低速、
无负载环境验证。命令行回退示例：

~~~bash
--teleop.position_control_mode=legacy_frame_delta
~~~

legacy 模式继续接受旧输入，不要求 sample time/age/sequence，并保留旧逐帧位置算法和逐轴
`max_delta_pos_m` 行为，便于同一日志对照。

### 11.5 Pipeline 与 A10 兼容结论

| 边界 | 本次变化 | 是否要求 A10 适配 |
|---|---|---|
| bridge → teleop | goal metadata 增加接收时间和序号 | 否 |
| teleop → mapper | 增加 4 个 `xlevr.*` 时序字段；mapper 内部消费 | 否 |
| mapper → robot | 仍输出 `ee.enabled`、6 个 `ee.delta_*` 和 `gripper.pos` | 否 |
| A10 TCP | 仍发送 `SET_EE_DELTA.actions=[dx,dy,dz,rx,ry,rz,gripper]` | 否 |

因此 A10 收到的数据类型、字段数量、顺序和单位没有变化，不需要同步改接收端。变化只在于
六维增量的生成过程更安全、平滑；disabled 帧仍由现有 A10 adapter 下发六维零增量。内部
teleop 原始 action schema 增加了时序字段，但这些字段在 mapper 中被 pop，不进入 A10 线协议。

### 11.6 修改文件与审核归属

本次 A1 实现涉及：

| 文件 | 修改内容 |
|---|---|
| `src/lerobot/teleoperators/xlevr/vr_monitor_bridge.py` | 原子记录 goal/pose monotonic 时间和独立序号；partial goal 不刷新 pose freshness |
| `src/lerobot/teleoperators/xlevr/teleop_xlevr.py` | 计算 sample age，传递时序字段，统一 squeeze/trigger enabled，补齐 idle schema |
| `src/lerobot/teleoperators/xlevr/xlevr_processor.py` | safe/legacy 双路径、状态机、防护、滤波、向量限速/加速度、渐变和 reset |
| `src/lerobot/teleoperators/xlevr/quaternion_utils.py` | 拒绝 NaN/Inf、零范数和畸形四元数 |
| `src/lerobot/teleoperators/xlevr/config_xlevr.py` | 新增 A1 配置与暂定默认值 |
| `src/lerobot/teleoperators/xlevr/factory.py` | 将 A1 配置注入 mapper |
| `src/lerobot/teleoperators/xlevr/README.md` | 补充 safe/legacy 用法、跳点重锚语义和实机标定警告 |
| `tests/teleoperators/test_xlevr_position_control.py` | 新增状态机、限幅、跳点、重锚和 legacy 测试 |
| `tests/teleoperators/test_xlevr_bridge_freshness.py` | 新增 pose freshness 与 partial goal 测试 |
| `docs/source/xlevr_teleoperation_optimization.md` | 同步技术细节、决策和本开发记录 |
| `docs/source/_toctree.yml` | 将本文加入项目文档导航 |

分支创建前已经存在、不是本次 A1 新增的工作区修改：`CLI.txt`、
`src/lerobot/datasets/video_utils.py`、`src/lerobot/robots/a10_follower/config_a10_follower.py`、
`src/lerobot/scripts/lerobot_record.py`、`src/lerobot/teleoperators/xlevr/vr_events.py`，以及未跟踪的
`a10_vr_9_11_3.zip` 和参考文档 `轮臂双臂增量遥操方法与参数.md`。审核或提交时不要把这些文件
误归入 A1.1/A1.2。

A1.3 后续确实修改了 `lerobot_record.py`，但仅追加 processor reset、诊断接线和实际下发

### 11.7 验证记录和未完成验证

- `python -m py_compile`：本次 Python 源码和新增测试均通过。
- `git diff --check`：通过。
- 不落盘的最小依赖 harness：bridge partial-goal freshness、首次 ARMING、方向映射、重复帧、
  向量速度限制、松开/重使能重锚和零四元数 fail-closed 均通过。
- 新增正式 pytest 用例，但当前系统 Python 缺少 `torch` 和 `draccus`，仓库 pytest collection
  无法完成；没有为通过测试而临时安装或修改项目依赖。
- 尚未执行真实 VR 浏览器、`--vr-only` 长时日志回放、模拟 A10 TCP 或实机低速测试。
- 当前参数不得视为实机验收完成；完成上述验证并记录 VR 频率/抖动和 A10 限制后再合并。

审核建议先限定本次文件：

~~~bash
git diff -- src/lerobot/teleoperators/xlevr \
  tests/teleoperators/test_xlevr_position_control.py \
  tests/teleoperators/test_xlevr_bridge_freshness.py \
  docs/source/xlevr_teleoperation_optimization.md \
  docs/source/_toctree.yml
~~~

## 12. 2026-09-14 A1.3 开发记录

### 12.1 实现结果

A1.3 代码实现已完成，目标是让每个控制帧既能复盘，又不让诊断写盘阻塞控制线程：

- mapper 在处理原始 action 前保存 VR 位姿、四元数、按键、source/receive time、age 和 sequence。
- 所有 safe/legacy 返回路径均填写 `guard_reason`、状态和最终 EE/夹爪命令。
- ACTIVE 帧额外保存 body/mapped delta、滤波前后速度、速度限幅、ramp、加速度限幅和最终速度。
- 正式录制和 `--vr-only` 均可写严格 JSONL；NaN/Inf 转为 `null`。
- 后台队列容量固定可配，`put_nowait` 失败只增加 dropped 计数，不等待磁盘。
- 录制阶段边界统一 reset pipeline，避免 reset 阶段的锚点和滤波速度进入下一 episode。
- 数据集 action 优先使用 `Robot.send_action()` 返回值，符合 Robot 接口“实际发送动作”的契约。

### 12.2 数据边界与兼容性

| 层级 | A1.3 后的内容 | 用途 |
|---|---|---|
| teleop 原始 action | `xlevr.*` 位姿、按键、时间和序号 | mapper 内部输入 |
| metadata JSONL | 原始 VR + 判定过程 + 最终/实际发送命令 | 调试、回放、调参 |
| 训练 action | `ee.enabled`、6 个 `ee.delta_*`、`gripper.pos` | 数据集训练目标 |
| A10 TCP | `SET_EE_DELTA.actions` 7 个数 | 机器人执行 |

A10 接收端无需适配：命令名、7 维顺序、EE-local 参考系和 m/rad 单位完全未变。A1.3
只清理了 LeRobot 数据集内部 action schema，并增加旁路日志。

这是有意的数据集 schema 变化：A1.3 前的 XLeVR 数据集 action 还含
`vr.trigger/thumbstick/button_*`，A1.3 后不再包含。旧数据集若用 `--resume=true`
出现 feature mismatch，应新建数据集，或先做明确的数据迁移；不得静默混合两种 action 维度。
legacy_frame_delta 只回退位置生成算法，不回退旧数据集 schema。

### 12.3 JSONL 文件与关键字段

正式录制默认路径：

~~~text
<dataset_root>/meta/xlevr_diagnostics/session_YYYYmmdd_HHMMSS.jsonl
~~~

可用 `--teleop.record_vr_diagnostics=false` 关闭；默认
`diagnostics_queue_size=2048`、`diagnostics_flush_every=30`。

关键字段分组：

- 对齐：`diagnostics_record_sequence`、`mapper_frame_index`、`recording_phase`、
  `dataset_episode_index`、`control_frame_index`。
- 原始输入：`raw_position_vr_m`、`raw_orientation_quat_xyzw`、`raw_trigger`、
  `raw_thumbstick`、`raw_buttons`。
- 时序：`sample_sequence`、`sample_receive_time_s`、`sample_age_s`、
  `source_timestamp`、`sample_dt_s`。
- 判定：`control_state`、`guard_reason`、`vr_input_speed_m_s`。
- 整形：`unfiltered_velocity_m_s`、`filtered_velocity_m_s`、
  `speed_limited_velocity_m_s`、`speed_limited`、`ramp_gain`、
  `acceleration_limited`、`commanded_velocity_m_s`。
- 输出：`final_position_delta_m`、`final_rotation_delta_rad`、`gripper_command`、
  `sent_action`。

`diagnostics_record_sequence` 出现缺号表示后台队列曾丢帧；进程退出日志同时给出
written/dropped/error，台架验收要求 dropped 为 0。

### 12.4 本次修改文件

| 文件 | A1.3 修改 |
|---|---|
| `src/lerobot/teleoperators/xlevr/diagnostics.py` | 新增有界后台 JSONL writer、严格 JSON 转换和 mapper 查找 |
| `src/lerobot/teleoperators/xlevr/xlevr_processor.py` | 逐帧诊断快照、过滤原因、中间量；移除输出中的 `vr.*` |
| `src/lerobot/teleoperators/xlevr/config_xlevr.py` | 新增诊断开关、queue 和 flush 配置 |
| `src/lerobot/teleoperators/xlevr/__init__.py` | 导出诊断接口 |
| `src/lerobot/scripts/lerobot_record.py` | 阶段 reset、JSONL 接线、记录实际 sent action |
| `examples/xlevr_to_a10/teleoperate.py` | 新增 `--diagnostics-jsonl`，记录 vr-only/teleoperate |
| `examples/xlevr_to_a10/test_action_print.py` | 从 raw action 读取调试手柄值，适配新 action schema |
| `tests/teleoperators/test_xlevr_position_control.py` | 新增原因、中间量、最终值及 action schema 测试 |
| `tests/teleoperators/test_xlevr_diagnostics.py` | 新增 writer drain、NaN/Inf、序号和快照复制测试 |
| `src/lerobot/teleoperators/xlevr/README.md` | 增加日志路径、schema 与 vr-only 用法 |
| `docs/source/xlevr_teleoperation_optimization.md` | 本设计和开发/验收记录 |

未修改 A10 client/follower 线协议，也未修改仓库外 `/home/robot/VLA/XLeRobot`。

### 12.5 验收结果

- `python -m py_compile`：A1.3 源码、示例和测试通过。
- `git diff --check`：通过。
- 最小依赖离线 harness：14/14 通过，其中 mapper/状态机/action schema 12 项，
  JSONL writer/诊断提取 2 项。
- 已验证原因包括 arming、active、duplicate_sample、stale_age、invalid_sample 和
  spike_pending；原有 release/re-anchor、sequence rollback、速度/加速度限制和 legacy
  回归也通过。
- JSONL 验证了后台 drain、连续 record sequence、NaN/Inf 写为 JSON `null`、close 后拒绝写入。
- 正式 pytest 仍受当前系统 Python 缺少 `torch` 和 `draccus` 阻塞，无法完成 collection；
  本次没有安装依赖或修改环境来掩盖该问题。
- 尚未完成：真实 VR 浏览器长时采集、模拟 A10 TCP、A10 低速无负载实机测试。

因此当前结论是“A1.3 代码和离线算法可用”，不是“整套 VR/A10 已实机验收”。

### 12.6 下一步台架路径

1. 无机器人采集 5 至 10 分钟真实 VR 日志：

   ~~~bash
   python examples/xlevr_to_a10/teleoperate.py --vr-only \
     --diagnostics-jsonl /tmp/xlevr_vr_only.jsonl
   ~~~

2. 统计 sample rate/age/dt、各 `guard_reason` 次数、最大 VR 输入速度、限幅占比、
   writer dropped 数；确认静止噪声和正常快速动作不会频繁触发 spike。
3. 接模拟 TCP 服务验证每个 mapper 帧对应的 7 维 payload，断开/重连期间无非零补发。
4. 实机仅在低速、无负载、可立即急停条件下开始；先验证松开、stale、跳点、重连和
   episode 切换均为零输出，再逐步标定 cutoff、速度、加速度和 ramp。
5. 台架记录必须写明 VR/浏览器版本、控制 fps、A10 固件、参数值以及 JSONL 路径。

审核 A1.3 可先限定：

~~~bash
git diff -- src/lerobot/teleoperators/xlevr \
  src/lerobot/scripts/lerobot_record.py \
  examples/xlevr_to_a10 \
  tests/teleoperators/test_xlevr_position_control.py \
  tests/teleoperators/test_xlevr_diagnostics.py \
  docs/source/xlevr_teleoperation_optimization.md
~~~

## 13. 参考

- 项目内部参考：轮臂双臂增量遥操方法与参数.md
- 当前 XLeVR 说明：src/lerobot/teleoperators/xlevr/README.md
- 当前坐标映射测试：tests/teleoperators/test_xlevr_axis_remap.py

## 14. 2026-09-17 控制职责调整

- 结论：保留 A1.1 输入防护和 A1.3 诊断；A1.2 默认旁路，A10 `vr_vel` 负责运动整形。
- 代码：新增 `motion_shaping_mode`；默认 `robot_controller`，回退值为 `lerobot_a1`。
- 诊断：新增 `motion_shaping_mode`、`motion_shaping_owner` 和 `shaping_bypassed`。
- 兼容：下行仍为 7 维 `SET_EE_DELTA`，A10 接收端与训练 action schema 无需修改。
- 验证：XLeVR 状态机、诊断、A1.2 回退及坐标映射共 22 项测试通过；`git diff --check` 通过。

## 15. A2 锚点闭环（A2.0/A2.1/A2.2 代码已完成）

本节与前面的 A1 开发记录分开。当前已完成 LeRobot 影子计算和 A10 影子接收；现有控制模式与旧协议保持不变。

### 15.1 控制职责

| 模块 | 职责 |
|---|---|
| LeRobot | A1.1 输入防护、人手锚点、相对位姿、坐标映射和 A1.3 诊断 |
| A10 | 机器人 FK 锚点、目标/reference、跟踪误差、限速、slew、IK 和电机命令 |

LeRobot 不实现 A10 FK/IK；A10 不解释 WebXR 原始坐标；A1.2 继续默认旁路。

### 15.2 最小方案

- LeRobot 新增 `anchored_pose` 模式，按下时记录 human anchor，之后输出相对锚点的绝对六维 offset。
- A2.1 仅写诊断；A2.2 已通过 sideband 发送 `SET_EE_ANCHOR`，并在写入数据集前移除该内部字段。
- A10 在新 `anchor_id` 首帧用实际 FK 保存 robot anchor，并计算 `user_target_pm = robot_anchor_pm * offset_pm`。
- A10 新增 `user_target_pm` 和 `robot_anchor_pm`；复用 `target_pm` 作为受限 reference，原 `command_pm`、P 控制、限速、slew 和 IK 不变。
- 松开、stale、跳点或 fault 时废弃双锚点，不执行尚未完成的追赶路径。

#### A2.0 契约（已确定，A2.2 才上线）

计划中的最小逻辑消息为：

```json
{
  "type": "SET_EE_ANCHOR",
  "active": true,
  "session_id": "进程内随机 UUID",
  "anchor_id": 12,
  "sample_sequence": 1836,
  "offset": [0.01, 0.00, -0.02, 0.00, 0.05, 0.00],
  "gripper": 0.0
}
```

- `offset[0:3]`：相对人手锚点、经现有 `axis_remap` 映射到 EE-local 的绝对平移，单位 m。
- `offset[3:6]`：相对人手锚点的旋转向量，经相同轴映射，单位 rad；不是逐帧姿态增量。
- `session_id + anchor_id` 唯一标识一次锚点；进程重启生成新 `session_id`，每次有效重锚递增 `anchor_id`。
- `sample_sequence` 是生成该 offset 的 VR 样本序号；重复帧不推进目标，序号回退使锚点失效。
- `active=false`、松开、stale、无效样本、时间回退或跳点会使当前锚点失效；重新按下或跳点恢复后从零 offset 重锚。
- 锚点建立时冻结位置/角度 scale，避免中途切换精细模式导致整段绝对 offset 突变。
- 绝对 offset 不使用逐帧 `max_delta_*` 截断；A10 后续由 reference governor、工作空间和原速度/加速度链限制实际运动。

A2.1 影子计算公式：

```text
translation_ee = axis_remap(R(anchor_quat)^T * (position_now - position_anchor)
                            * vr_to_robot_scale * frozen_pos_scale)
rotation_ee    = axis_remap(log(inv(anchor_quat) * quat_now) * frozen_angle_scale)
```

### 15.3 实施顺序

1. **A2.0 契约（已完成）**：已确定协议字段、单位、坐标系、重锚和失效语义。
2. **A2.1 影子计算（已完成）**：LeRobot 计算并记录 anchor offset，仍发送旧命令。
3. **A2.2 影子联调（代码已完成）**：LeRobot 发送新协议，A10 记录 robot/user target，不驱动电机；真实 TCP/实机日志待验证。
4. **A2.3 A10 闭环**：加入 reference governor 和 tracking error 冻结/故障策略，复用原控制链。
5. **A2.4 低速台架**：低速、无负载、可急停条件下验收，再决定是否切换默认模式。

### 15.4 主要修改文件

- LeRobot：`xlevr_processor.py`、`config_a10_follower.py`、`a10_follower.py`、`a10_client.py`及对应测试。
- A10：`a10_anchor_protocol.hpp/.cpp`、`a10_tcp_server.hpp/.cpp`、`a10_vr_vel_plan.cpp`及对应测试。
- 不修改 XLeRobot WebXR、A10 FK/IK 模型、旧 `SET_EE_DELTA` 和夹爪控制。

### 15.5 验收重点

- 首次按下零运动，松开/重按无跳变。
- 重复、丢失或覆盖中间包后，下一绝对 offset 能恢复正确目标。
- VR 跳点、stale、序号回退不产生追赶轨迹。
- tracking error 超限时 reference 停止前进，松开后不继续执行积压路径。
- `safe_frame_delta` 回归结果不变；新模式验收前不设为默认、不与旧模式数据静默混录。

### 15.6 A2.0/A2.1 开发记录（2026-09-17）

- 范围：仅修改 LeRobot；`A10_new/kaanh` 未修改。当前 action 和 TCP 下行仍是原 `SET_EE_DELTA`。
- 实现：新增 `compute_anchor_shadow`（默认开启），复用 A1.1 有效样本、轴映射和失效逻辑，锚点 offset 仅进入 A1.3 JSONL 诊断，字段前缀为 `anchor_shadow_*`。
- 文件：`config_xlevr.py`、`factory.py`、`xlevr_processor.py`、`test_xlevr_anchor_shadow.py` 和本文档。
- 验证：锚点绝对偏移、动作输出不变、重复帧不推进、松开重锚、scale 冻结、姿态轴映射和跳点重锚均已覆盖；A1/A2.1 定向回归 `28 passed`，整个 `tests/teleoperators` 回归 `41 passed`。

A2.1 实机验证仍建议录制 JSONL，对比 `anchor_shadow_offset_6d`，并检查松开、重复帧、stale 和跳点后的归零重锚。

### 15.7 A2.2 开发记录（2026-09-18）

- 分支：LeRobot 和 A10_new 均为 `kaanh_vr_op`，原 `kaanh` 未修改。
- 发送：mapper 附加内部 anchor sideband；A10 follower 先发 `SET_EE_ANCHOR`、再走原 `SET_EE_DELTA`，并在返回 action 前移除 sideband，数据集 schema 不变。
- 接收：A10 严格解析新协议并用 mailbox 传给 `vr_vel`；新锚点保存当前 FK 为 `robot_anchor_pm`，计算 `user_target_pm = robot_anchor_pm * offset_pm`。
- 边界：仅保存并周期打印影子目标；不写 `target_pm`、`command_pm`，不调用 IK/电机或夹爪，原 `SET_EE_DELTA` 路径不变。
- 测试：LeRobot 定向回归 `49 passed`；A10 协议单测通过，TCP 源文件语法检查通过。
- 限制：本机缺少 `kaanhbotConfig.cmake`，A10 完整工程 CMake 未能配置；真实 TCP/实机影子日志尚未联调。

A2.2 剩余工作仅为安全影子联调：确认 `robot_xyz/user_xyz` 日志正确，且机器人运动仍只由旧 `SET_EE_DELTA` 驱动；本轮不进入 A2.3。
