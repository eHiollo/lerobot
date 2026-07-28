# AsyncVLA: 真机 Test-Time Compute via 异步推理与人类引导验证

> **核心 Claim**: 在真机 VLA 部署中，利用异步推理流水线的 GPU 空闲窗口做并行 action 采样 + 轻量 verifier 选优 + 自适应计算预算，实现「多花算力换成功率」的 test-time compute scaling，同时保持实时控制频率。
>
> **分支**: 待定（建议 `dev/async-ttc`，横跨 lerobot + openpi）
> **日期**: 2026-07-28
> **目标投稿**: CoRL 2027 / RSS 2027 / ICRA 2027

---

## 一、背景与动机

### 1.1 LLM 的 test-time compute 启示

2025–2026 年 LLM 最重要的范式转变之一是 **test-time compute scaling**（o1/o3, DeepSeek R1, s1）：推理时不再单次贪心解码，而是通过多步推理、多次采样、self-verification 换取显著性能提升。核心发现是「推理算力可以替代/补充模型规模」。

**机器人 VLA 领域尚未系统回答同一个问题**：VLA 推理时多花算力，能否换来真机任务成功率的实质提升？如果能，代价（延迟、采样成本、安全）是多少？

### 1.2 2026 VLA landscape（已调研）

| 方向 | 代表工作 | 环境 | 缺口 |
|------|---------|------|------|
| TTC 架构改造 | RD-VLA（latent recurrent）, VLA-ATTC（cognitive clutch）, ElegantVLA（phase-adaptive） | **仿真 LIBERO** | 需改 base model，真机 0 篇 |
| TTC 采样验证 | RoboMonkey（sample+verify） | 仿真 | 1.5Hz，真机不可行 |
| Online adaptation | Agentic-VLA（reward synthesis + LGE + memory） | 仿真 | 真机部署是作者明确 future work |
| Fleet post-training | SOP（AgiBot） | 真机 | 需机器人机群，单机不适用 |
| HiL dexterous | HandITL, DexHiL, BORA, HiL-ResRL | 真机 | 全聚焦灵巧手，单臂工业任务空白 |
| Continual VLA | ContinualVLA | 真机 | 仅 empirical study，无方法 |

**关键空白**：
1. **真机 TTC 完全空白** —— 仿真 TTC 已卷，真机 0 篇（三座大山：采样成本高、无地面真值 verifier、实时性约束）。
2. **单臂工业任务的在线适应** —— HiL 工作全在灵巧手。
3. **异步推理 × TTC 的联合** —— 没人把推理流水线本身当作 TTC 的 enabler。

### 1.3 我们的独特位置

我们的既有基础设施恰好把真机 TTC 的三座大山翻过去：

| 真机 TTC 难点 | 我们的既有资产 | 如何解决 |
|-------------|--------------|---------|
| 多次采样成本高 | **异步双缓冲流水线**（openpi dev/async-inference） | 执行 chunk_k 期间 GPU 本就空闲，并行采样 chunk_{k+1} 候选「免费」 |
| 无地面真值 verifier | **VR 高频遥操作 + HIL 框架**（lerobot dev/vr_dev, dev/hil） | 人类 VR 轻纠偏作为偏好信号，训练轻量 Relative Action Critic |
| 实时性约束 | **72Hz 控制环 + 异步发送 + TCP_NODELAY** | 自适应计算预算：简单动作 N=1，复杂动作 N=8，保控制频率 |

**这就是论文的核心叙事**：不是把 TTC 硬搬到真机，而是论证「异步推理流水线天然是真机 TTC 的 enabler」，并用真机实验给出第一张 TTC 权衡曲线（计算 vs 成功率 vs 延迟）。

---

## 二、核心 Insight

> **异步推理流水线的 GPU 空闲窗口，是真机 test-time compute 的免费采样预算。**

同步部署中，机器人执行 chunk 期间 GPU 闲置；异步部署中，这段窗口被用来推理下一个 chunk。我们进一步论证：这段窗口的算力足以并行采样 **N 个候选 chunk**，加上一个**轻量 verifier** 选优，就能在不增加任何 wall-clock 延迟的前提下，把单次贪心推理升级为 best-of-N 推理。

而 verifier 的训练信号，来自 **HIL 人类纠偏**——人在 VR 中对失败候选的介入，天然构成「候选 A 优于候选 B」的偏好对，无需额外标注。

---

## 三、方法：三件套

### 3.1 Async Parallel Sampling（异步并行采样）

**关键观察**：对 ACT / Diffusion Policy / π0 类 VLA，N 路采样 ≠ N 倍成本。

- **Amortized sampling**：vision-language backbone（ViT + LLM）对同一观测只 forward 一次，action head（diffusion denoising / ACT decoder）采样 N 次。backbone 占计算大头，head 采样成本接近可忽略。VLA-ATTC 用了同一 trick。
- **流水线重叠**：稳态下 `T_exec(chunk_k)` 期间完成 `backbone(obs_k) + head_sample(N)`。只要 `T_backbone + T_head×N < T_exec`，采样完全隐藏。

**实现**：
```
稳态每轮:
  chunk_k = buffer.pop()                # 上一轮 verifier 选出的最优
  push chunk_k 到机器人 (T_exec)
  ── GPU 重叠窗口 ──
  obs_k = get_follower_state()          # 新鲜观测(A10 已改 batch 期间刷新)
  feats = backbone(obs_k)               # 1 次,大头
  candidates = [head_sample(feats) for _ in range(N)]  # N 次,小头
  best = verifier.select(candidates, obs_k)
  buffer.append(best)
  wait_policy_idle()
```

### 3.2 Lightweight Verifier（轻量验证器）

两级设计，从无需训练到逐步学习：

**Level 1 — Geometric Consistency（零训练，v1 baseline）**
- N 个候选 chunk 的 action 序列两两计算距离（关节空间 L2 + gripper 一致性）
- 选 **medoid**（与其他候选总距离最小者）
- 假设：正确动作在多采样中聚簇，错误动作是离群点
- 类似 RoboMonkey 的 majority voting，但在关节空间连续度量

**Level 2 — Relative Action Critic（学习式，v2 核心创新）**
- 训练数据：HIL 部署中，人类 VR 介入纠正失败候选 → 构成偏好对 `(a_human ≻ a_vla)`
- 模型：轻量 critic，`C(obs, a_i, a_j) → P(a_i ≻ a_j)`，相对判断而非绝对打分（借鉴 VLA-ATTC 的 RAC，降低训练难度）
- 推理：N 个候选 tournament 两两比较，选出最优
- 与 VLA-ATTC 区分：它用自动构造的仿真偏好对，我们用**真机 HIL 偏好对**——这是真机场景的自然产物

**Level 3 — Uncertainty-Gated HIL（主动求助）**
- 候选间分歧度（几何一致性的逆）超阈值 → 主动请求人类 VR 介入
- 区别于传统 IIL（人主动接管）：VLA **主动求助**，是真机部署的新交互模式
- 求助片段自动成为 Level 2 的训练数据，闭环

### 3.3 Adaptive Compute Budget（自适应计算预算）

- **分歧度驱动**：候选 action 分布熵 / 几何分歧度低 → N=1（简单动作不多花算力）；高 → N=8（复杂动作多采样）
- **频率保障**：backbone 每轮必跑（保观测新鲜），head 采样数 N 动态调整，确保 `T_total < T_exec` 硬约束
- 与 ElegantVLA 区分：它用 RL 学 scheduler 改 backbone 计算层级，我们固定 backbone、只调采样数，不动 base model

---

## 四、系统架构与已有资产映射

```
┌─────────────────────────────────────────────────────────┐
│  GPU 端 (5090)                                           │
│  openpi serve_policy.py (websocket :8000)               │
│  ├─ backbone(obs) × 1        ← 复用现有                  │
│  ├─ head_sample(feats) × N   ← 新增:批量采样接口          │
│  └─ verifier.select(cands)   ← 新增:Level1/2/3           │
└──────────────▲──────────────────────────────────────────┘
               │ websocket msgpack
┌──────────────┴──────────────────────────────────────────┐
│  机器人端桥接 (openpi/third_party/A10_new/client)          │
│  run_bridge.py --mode async --sample-n N                 │
│  ├─ 双缓冲流水线             ← 复用 dev/async-inference    │
│  ├─ ThreadedCamera          ← 复用                      │
│  └─ HIL 触发 → VR 求助       ← 新增                     │
└──────────────▲──────────────────────────────────────────┘
               │ TCP JSON (:8080)
┌──────────────┴──────────────────────────────────────────┐
│  A10 控制器 (third_party/A10_new C++)                     │
│  SET_JOINTS_BATCH / GET_STATE / STOP_POLICY  ← 复用       │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│  人类端 (lerobot)                                        │
│  XLeVR 遥操作 (72Hz, EE target 模式)        ← 复用 dev/vr_dev │
│  ├─ 被动接管 (传统 IIL)                                    │
│  └─ 主动求助响应 (Level 3)    ← 新增                      │
└─────────────────────────────────────────────────────────┘
```

**改动量评估**：
| 组件 | 改动 | 位置 |
|------|------|------|
| 批量采样接口 | 新增 | openpi `serve_policy.py` |
| Verifier (L1/L2/L3) | 新增 | openpi `verifier/` |
| Bridge 采样数参数 | 改 | `run_bridge.py` |
| HIL 求助协议 | 新增 | bridge ↔ XLeVR |
| Critic 训练管线 | 新增 | lerobot `policies/relative_critic/` |
| 其余全部 | **复用** | dev/async-inference + dev/vr_dev |

---

## 五、实验设计

### 5.1 任务

真机 A10（高精度，可胜任精密对准类任务），3 类难度递进任务（单臂工业场景，区别于灵巧手工作）：
1. **Reach**（简单，已有数据集 `dataset_5_9`）：验证 baseline 不退化
2. **Pick-and-Place**（中等）：主对比任务
3. **精密插入/对准**（困难）：TTC 收益最大场景，动作精度敏感，A10 精度可支撑

每任务 50 trials × N 条件，报成功率 + std。

**全部真机实验，不做仿真先行**：避免引入 real2sim 资产建模与 sim2real 迁移的额外变量；止损机制改用真机小规模 pilot（见里程碑 M2）。

### 5.2 Baseline 对比

| 条件 | 说明 |
|------|------|
| B0 sync | 同步单采样（原始部署） |
| B1 async | 异步单采样（dev/async-inference 现状） |
| B2 async + L1 | + 几何一致性 medoid（N=4 固定） |
| B3 async + L2 | + Relative Action Critic |
| B4 async + L3 | + 主动求助（完整 AsyncVLA） |
| B5 async + adaptive | 自适应 N（1~8） |

### 5.3 关键指标

1. **任务成功率**（主指标）
2. **TTC 权衡曲线**：x=平均计算预算（backbone 当量），y=成功率，多条曲线（固定 N vs adaptive N）—— **本文核心图**
3. **控制频率**：验证 TTC 不降实时性（对比 RoboMonkey 1.5Hz）
4. **人类干预率**：Level 3 主动求助次数 / 被动接管次数
5. **采样隐藏率**：GPU 窗口内完成 N 路采样的比例

### 5.4 Ablation

- N ∈ {1, 2, 4, 8}：采样数边际收益
- Verifier 设计：medoid / RAC / oracle（ ground-truth 成功）
- 自适应触发阈值
- 观测新鲜度：TTC 对 obs 陈旧的敏感度

### 5.5 预期结果

- Pick-and-Place：B1 ~70% → B4 ~90%
- 精密插入：B1 ~30% → B4 ~70%（TTC 收益最大）
- 控制频率：B4 ≥ 15Hz（vs RoboMonkey 1.5Hz）
- adaptive N 在同等成功率下计算省 40%+

---

## 六、与现有工作的区分（Positioning）

| 工作 | 环境 | 改 base model | Verifier | 我们的差异 |
|------|------|--------------|----------|-----------|
| RD-VLA | 仿真 | ✅ recurrent 架构 | 无（latent 收敛） | 我们不动 base model + 真机 |
| VLA-ATTC | 仿真 | ❌ | RAC（自动偏好对） | 我们用真机 HIL 偏好对 + 异步采样隐藏 |
| ElegantVLA | 仿真 | ✅ scheduler 改计算层级 | 无 | 我们固定 backbone 只调采样数 |
| RoboMonkey | 仿真 | ❌ | VLM verifier | 我们 15Hz+ vs 它 1.5Hz，真机可行 |
| Agentic-VLA | 仿真 | ✅ 训练侧 | reward synthesis | 我们推理侧 TTC，正交可结合 |
| BORA / HiL-ResRL | 真机灵巧手 | ✅ residual RL | 无 | 我们单臂 + 推理时 TTC，不训练 policy |

**一句话定位**：*第一个在真机上系统研究 VLA test-time compute 的工作，论证异步推理流水线是其天然 enabler，并给出真机 TTC 权衡曲线。*

---

## 七、风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| Verifier 不够好，TTC 退化为取均值 | 新颖性受损 | 从 L1 几何一致性起步（零训练保底），L2 用真机 HIL 数据逐步升级；oracle ablation 证明上限 |
| N 路采样超过 GPU 窗口 | 实时性下降 | amortized sampling（head 采样便宜）；adaptive N 硬约束 `T_total < T_exec` |
| HIL 偏好对数据量不足 | L2 critic 训不好 | L3 主动求助定向采集高价值偏好对；数据增广 |
| 与 VLA-ATTC 区分度被质疑 | 审稿风险 | 强调真机独有贡献（权衡曲线、HIL verifier、异步隐藏采样）；真机实验是它没有的 |
| 真机采样安全 | 硬件风险 | 候选 chunk 先经关节限位 clamp；分歧度极高时默认 STOP_POLICY |
| 任务太简单，TTC 无收益 | 实验失败 | 精密插入类精度敏感任务必做；M2 真机 pilot（单任务 20 trials）快速验证收益再全量投入 |

---

## 八、里程碑与时间线（约 3 个月，全真机）

| 阶段 | 内容 | 产出 | 时间 |
|------|------|------|------|
| M1 | openpi 批量采样接口 + L1 verifier | B2 跑通 | 2 周 |
| M2 | **真机 pilot 止损**：Pick-and-Place 上 B1 vs B2，20 trials | TTC 收益初步信号 | 1 周 |
| M3 | 真机 B0/B1/B2 对比 + N 扫描（Pick-and-Place + 精密插入） | 第一张真机权衡曲线 | 3 周 |
| M4 | HIL 偏好对采集 + L2 critic 训练 | B3 跑通 | 4 周 |
| M5 | L3 主动求助 + adaptive N | B4/B5 完整系统 | 3 周 |
| M6 | 三任务全量实验 + ablation | 完整实验数据 | 3 周 |
| M7 | 论文撰写 | 投稿 | 3 周 |

**M2 是关键止损点**：若 pilot 中 B2 相对 B1 无显著收益（<5% 成功率提升），说明几何一致性 verifier 对目标任务无效，需直接进入 M4 学习式 critic，或评估转向备选方向。

---

## 九、开放问题（待调研）

1. **Verifier 上限**：真机上 RAC 能逼近 oracle 多少？需要多少 HIL 偏好对？
2. **观测陈旧 × TTC**：多采样是否放大对 obs 新鲜度的敏感？与 receding horizon 的关系？
3. **TTC × RL 结合**：TTC 选出的最优 chunk 执行结果，能否反哺 policy 更新（离线 RL / DAgger）？与 dev/hil 残差 RL 线的关系：**两线当前并行独立开发**（dev/hil 验证通过后才合并主分支），AsyncVLA 聚焦推理时计算、dev/hil 聚焦训练时适应，叙事正交；合并主分支前需统一故事（例如「TTC 负责部署期即时纠错，残差 RL 负责长期参数更新」）。
4. **跨任务迁移**：一个任务上训的 RAC，能否迁移到新任务？

---

## 十、备选方向（若主线受阻）

- **备选 B**：HIL-Driven Real-World Online Adaptation（Agentic-VLA 真机版）——把 reward synthesis + language-guided exploration 搬到真机，HIL 做 safe exploration
- **备选 C**：Uncertainty-Aware VLA Deployment with Active Human Handover——聚焦 Level 3 主动求助，做成交互模式创新

---

## 进度记录

(每个里程碑完成后在此追加)
