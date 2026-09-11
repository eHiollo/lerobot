# XLeVR 遥操作（LeRobot ↔ A10）

本目录把 **XLeVR（WebXR VR 手柄）** 接到 LeRobot，再经 TCP 发给 A10 机械臂控制器。

外部依赖：[XLeRobot/XLeVR](https://github.com/huggingface/XLeRobot)（默认路径见 `config_xlevr.py` 的 `xlevr_path`）。

---

## 数据流

```
VR 浏览器 (WebXR)
    → wss WebSocket → XLeVR/vr_ws_server.py
    → ControlGoal 队列
    → vr_monitor_bridge.py
    → teleop_xlevr.py              # xlevr.* 原始字段
    → xlevr_processor.py           # body 系四元数增量 → ee.delta_* + gripper
    → A10Follower.send_action
    → TCP: SET_EE_DELTA {...}
```

示例：`examples/xlevr_to_a10/teleoperate.py`、`test_action_print.py`。

---

## 网络服务

| 服务 | 默认端口 | 说明 |
|------|----------|------|
| HTTPS（VR 网页） | **8443** | `https://<本机IP>:8443` |
| WebSocket | **8442** | XLeVR 内部使用 |

**同一台机器只能跑一个 XLeVR 实例**（一个 `teleoperate`）。端口占用时结束旧进程后再启动。

---

## 坐标系与控制

### VR 手柄 body 系

- **+X** 右，**+Y** 上，**+Z** 朝后（往前推 ≈ −Z）
- 平移：`Δp_body = R_prev⁻¹ (p_curr − p_prev)`
- 旋转：`R_Δ = R_prev⁻¹ R_curr` → rotvec (rad)

### 机器人 `SET_EE_DELTA` 系

增量在 **末端（EE）局部系**，不是世界/基座系。同款臂、基座怎么转都不改 `axis_remap`。

EE：**+X** 上，**+Y** 右，**+Z** 前。

`axis_remap`（平移与 rotvec 共用）：`上=+VR_Y, 右=+VR_X, 前=-VR_Z`

### TCP 7 维

```text
SET_EE_DELTA {"actions": [dx, dy, dz, rx, ry, rz, gripper]}
```

| 下标 | 含义 | 单位 |
|------|------|------|
| 0–2 | 位置增量 | m |
| 3–5 | 旋转向量（非欧拉） | rad |
| 6 | 夹爪（右手摇杆 x） | ~[-1, 1] |

`ee.enabled == false` 时前 6 维为 0，夹爪仍发送。

### 操作

- **侧键 squeeze**：启用机械臂（粗调，`pos_scale=0.9`, `angle_scale=1.3`）
- **前扳机 trigger ≥ 0.5**：无需 squeeze，scale ×0.5 精调
- **摇杆 x**：夹爪

默认控制 / 录制频率：**30 Hz**。

---

## 常用命令

```bash
cd /home/allen/Allen/lerobot

python examples/xlevr_to_a10/teleoperate.py --robot-host 192.168.1.12 --robot-port 8080
python examples/xlevr_to_a10/test_action_print.py
```

正式数据集录制用 `lerobot_record`（`--dataset.fps=30`，`--teleop.type=xlevr`）。

---

## 本目录文件

| 文件 | 作用 |
|------|------|
| `config_xlevr.py` | 缩放、死区、轴映射、精调扳机 |
| `teleop_xlevr.py` | `Teleoperator`：连 XLeVR、读手柄 |
| `vr_monitor_bridge.py` | HTTPS/WS、合并 ControlGoal |
| `xlevr_processor.py` | body 系 VR → `ee.delta_*` |
| `quaternion_utils.py` | 四元数 / rotvec 工具 |
| `factory.py` | `make_xlevr_a10_processors()` |
| `vr_events.py` | 左手摇杆 → 录制事件 |
