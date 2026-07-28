# AsyncVLA 工具建设过程记录（M1 后续：D/A/B/C/E）

> 日期：2026-07-28 深夜
> 范围：verifier 测试固化(D) / 候选记录设施(A) / adaptive N(B) / L2 critic 框架(C) / related work 库(E)
> 分支：`dev/async-ttc`（三仓同名）
> 前置：`async_vla_ttc_m1_log.md`（M1 批量采样 + L1 verifier）

## 一、各项产出与验证

### D. verifier 回归测试（A10_new `3f60a9d`）

- 新增 `client/tests/test_verifier.py`，7 场景：聚簇选择 / 夹爪加权 / N=1 直通 / divergence 跨 N 口径 / 全同候选 / 权重长度校验 / 非法 ndim。
- 兼容 pytest 与直接运行（本机无 pytest，直接运行方式已通过）。
- **问题记录**：初版断言「离群距离 > 5×best」过严——理论上界为 3 倍（聚簇内距离→0 时），修为 2 倍。

### A. 候选实验记录设施（A10_new `c93d0a6`）

- `client/candidate_logger.py`：jsonl 记录每轮候选全量 + best_index + divergence，线程安全，写失败不影响主流程。数据量约 3-5KB/行（N=4,T=10,D=7），pilot 规模可接受。
- `client/analyze_candidates.py`：三类分析——
  1. divergence 分布（p50/p90 直接作为 adaptive N 阈值标定依据）；
  2. **各维度距离贡献占比（不加权口径）**——校准夹爪权重 0.03 的直接证据；
  3. verifier best_index 分布（>90% 偏向单一候选时告警，防服务端采样退化）。
- bridge 接入 `--log-candidates <dir>`，stop() 时关闭。
- 验证：mock 50 轮端到端跑通，三类报告输出正确。

### B. adaptive N 机制（A10_new `fbae6bc`）

- `--adaptive-n`：divergence EMA 驱动 {1,2,4,8} 档位，默认阈值 0.05/0.15/0.30（待 pilot 校准）。
- **设计决策 1（jit 约束）**：N 变化触发服务端 batch shape 变化 → jit 重编译，故用离散档位 + 降档滞后 5 轮，不用连续 N。
- **设计决策 2（升快降慢）**：测试发现 EMA 平滑会拖慢升档（单次 spike 只升到中间档），违背「复杂动作立即多采样」意图——修正为**升档判定用 max(ema, 瞬时值)**，降档仍走 EMA+滞后。
- **设计决策 3（N=1 信号缺失）**：N=1 走单次采样无 verifier → 无 divergence → 永远升不回来。加**周期探测**：N=1 时每 20 轮插一轮 4 候选。
- **测试修正记录**：初版测试期望「逐级降档」错误——EMA 在滞后期间持续走低，触发时直接跳到当前目标档（这正是期望行为）；探测轮 spike 升档问题引出决策 2 修正。最终 8 项状态机测试全过（AST 级，无 websockets 依赖）。

### S. openpi submodule 指针（openpi `2dd942b`）

- 跟进子仓 D/A/B 三提交至 `fbae6bc`。

### C. L2 Relative Action Critic 框架（lerobot `27d606d`）

- `src/lerobot/policies/relative_critic/`：独立模块（不挂 lerobot policy 注册体系，避免过度工程）。
- 网络：state MLP + action chunk MLP + pair head → P(a_i≻a_j) logit；锦标赛 select 与 L1 同接口便于 bridge 替换。
- 数据：jsonl 偏好对格式约定 + 近正远负合成对冷启动（RoboMonkey Stage-1 思路）；**真机 HIL 对格式待 dev/hil 介入事件格式确定后联调**（当前最大待定项）。
- 训练：AdamW + BCE，val_acc 最优存 best.pt。
- 验证（conda lerobot, torch 2.7.1）：前向/锦标赛 shape、合成数据收敛（loss 0.704→0.000）、泛化胜率 20/20。
- **测试修正记录**：初版「训后 select 选近扰动」断言因小数据过拟合 + 常数偏移候选超出训练分布而不稳，改为直接测训练目标泛化（20 对 win_score 胜率）。

### E. related work 库（lerobot `8f38923`）

- `docs/async_vla_related_work.md`：14 篇对标按 TTC / 在线适应 / HIL / 持续学习四类，每篇一句话定位 + 差异。
- `docs/async_vla_refs.bib`：对应 BibTeX。**注意：arXiv id 来自检索摘要，作者/年份信息论文撰写阶段需经 Scholar 核对补全**。

## 二、需要后期 check 的点（新增）

1. **adaptive N 阈值默认值是拍的**（0.05/0.15/0.30）：pilot 时先开 `--log-candidates` 固定 N=4 跑，用 `analyze_candidates.py` 的 p50/p90 校准后再开 `--adaptive-n`。
2. **adaptive N 切档触发 jit 重编译**：档位切换当轮有编译停顿，pilot 观察日志中升降档频率是否合理（若过于频繁说明阈值太近或 EMA alpha 太大）。
3. **探测轮数据混入候选记录**：jsonl 中 `sample_n` 字段可区分（探测轮=4，常规 N=1 档位无记录），分析时注意过滤。
4. **critic 数据格式是约定而非联调结果**：HIL 介入事件（dev/hil `get_teleop_events`）落地后需对齐字段。
5. **BibTeX 条目信息未核实**：投稿前必须 Scholar 逐条核对。

## 三、第二轮：critic 数据管线 + bridge 集成（G1/G2/G3）

### G1. 合成偏好对生成脚本（lerobot，本次）

- `generate_synthetic_pairs.py`：直读 parquet（`LeRobotDataset` 版本检查拒绝旧格式数据集，绕过），锚 chunk 取 `state[t:t+T]` 绝对关节序列。
- **重要发现（数据集）**：`dataset_5_9` 的 parquet `action` 列是**常量**（逐维 std=0，采集侧未写入真实 action）；openpi 训练以 `use_state_as_action_targets=True` 从 `observation.state` 构造动作目标，故本脚本同约定。后续新采数据集若 action 列正常，注意语义对齐。
- **分维噪声**：`make_synthetic_pair` 加 `dim_noise_scale`——关节 rad 尺度 (0.02/0.2) 与夹爪 mm 尺度差 100 倍，不加缩放则夹爪扰动无意义。
- 动作空间对齐：bridge 端 candidates 经 `AbsoluteActions` 输出变换后为**绝对关节空间**，合成对同样在绝对空间构造（训练时模型学的是 delta，critic 看到的是绝对，两者不冲突——critic 是独立模块）。

### G2. 端到端训练验证（本次）

- `dataset_5_9`（30 episodes, Reach）→ `--stride 2 --pairs-per-step 3` → **7524 对**。
- `train_relative_critic` 默认 config 跑满 20k steps（CPU 38s），**best_val_acc=1.000**。合成对近/远扰动可分性极强，满分属预期——只证明管线正确，不代表真实候选判别力（需 HIL 标注数据）。

### G3. bridge 双 verifier 集成（A10_new，本次）

- `verifier.py`：拆出 `pairwise_dist_matrix` / `compute_divergence` 独立函数。
- 新增 `critic_verifier.py`：**自包含网络定义**（`_build_net` 按 checkpoint config 重建，state_dict 键名与 lerobot 侧兼容，E2E load 验证通过）——bridge 不依赖 lerobot 包，仅需 torch（lazy import，未装时报错提示回退 medoid）。
- bridge `--verifier medoid|critic` + `--critic-checkpoint` + `--critic-device`（默认 cpu）。
- **设计决策（divergence 口径统一）**：critic 无天然分歧度，adaptive N 与日志的 divergence **始终用几何口径**（两实现可比）；critic 的锦标赛胜场数存日志 `per_cand_score` 字段（原 `per_cand_mean` 改名，含义随 `verifier` 字段切换）。
- 验证：mock 硬件行为级测试——medoid 路径 sample_n 透传、critic 路径选中近专家候选、缺 checkpoint/未知 verifier 两个 guard 报错；critic E2E（真 checkpoint）：噪声越大胜场越少（wins=[3,2,1,0]），单调性正确；verifier 7 项 pytest 重构后全过。

### 需要后期 check 的点（第二轮新增）

6. **机器人 PC 装 torch**：critic 模式需 `pip install torch`（CPU 版即可，MLP 推理 ms 级）；未装时 medoid 不受影响。
7. **Reach 数据夹爪恒 0**：`dataset_5_9` 全程夹爪不动（state 第 7 维 std=0），critic 的夹爪判别力为零——Pick-and-Place 数据（含夹爪变化）训练后才有意义；pilot 前用新任务数据重训。
8. **critic 合成对只编码「接近专家=好」先验**：真实候选间的细粒度优劣（如同等接近但碰撞风险不同）需 HIL Stage-2 数据；M4 联调时先用合成 critic 跑通全链路再替换。

## 四、commit 时间线（dev/async-ttc）

| 仓库 | 提交 | 内容 |
|------|------|------|
| A10_new | `3f60a9d` | D: verifier 回归测试 |
| A10_new | `c93d0a6` | A: 候选记录设施 + 分析脚本 |
| A10_new | `fbae6bc` | B: adaptive N 机制 |
| openpi | `2dd942b` | S: submodule 指针跟进 |
| lerobot | `27d606d` | C: relative critic 框架 |
| lerobot | `8f38923` | E: related work 库 |
| lerobot | (第一轮) | P: 过程文档 |
| lerobot | (本次) | G1: 合成对生成脚本 + 分维噪声 |
| A10_new | (本次) | G3: bridge 双 verifier 集成 + divergence 拆解 |
| openpi | (本次) | S2: submodule 指针跟进 |
| lerobot | (本次) | G4: CLI.txt critic 用法 + 过程文档更新 |
