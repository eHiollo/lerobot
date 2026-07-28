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

## 三、commit 时间线（dev/async-ttc）

| 仓库 | 提交 | 内容 |
|------|------|------|
| A10_new | `3f60a9d` | D: verifier 回归测试 |
| A10_new | `c93d0a6` | A: 候选记录设施 + 分析脚本 |
| A10_new | `fbae6bc` | B: adaptive N 机制 |
| openpi | `2dd942b` | S: submodule 指针跟进 |
| lerobot | `27d606d` | C: relative critic 框架 |
| lerobot | `8f38923` | E: related work 库 |
| lerobot | (本次) | P: 本过程文档 |
