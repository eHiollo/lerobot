# AsyncVLA M1 搭建过程记录

> 日期：2026-07-28
> 范围：M1 —— openpi 批量采样接口 + L1 verifier（几何一致性 medoid）+ bridge 集成
> 分支：`dev/async-ttc`（openpi 主仓 / A10_new 子仓 / lerobot 三仓同名）
> 方案文档：`docs/async_vla_ttc_proposal.md`

## 一、改动清单

### openpi 主仓（4 文件）

| 文件 | 改动 | 说明 |
|------|------|------|
| `src/openpi/policies/policy.py` | 改 | `Policy.infer` 新增 `sample_n` 关键字参数；>1 时把观测 broadcast 到 batch=N（JAX 用 `jnp.repeat`，PyTorch 用 `x.repeat`），模型一次前向输出 (N,T,D) 候选；输出保留 batch 维并附 `sample_n` 字段。`PolicyRecorder.infer` 改为 `**kwargs` 透传 |
| `src/openpi/policies/a10_policy.py` | 改 | `A10Outputs` 兼容 ndim==2/3，切片改为 `actions[..., :7]` |
| `src/openpi/serving/websocket_policy_server.py` | 改 | handler 从 obs 中 `pop("sample_n")`（不进入 transforms），>1 时调用 `infer(obs, sample_n=N)` |
| `packages/openpi-client/.../websocket_client_policy.py` | 改 | 官方客户端 `infer(obs, sample_n=1)`，>1 时注入 obs，协议两端对齐 |

### A10_new 子仓 client/（3 文件）

| 文件 | 改动 | 说明 |
|------|------|------|
| `client/verifier.py` | 新建 | L1 `GeometricMedoidVerifier`：候选两两加权 L2 距离（对 horizon 取均值）→ 选 medoid；输出 `VerificationResult{best, best_index, divergence, per_candidate_mean_dist}`。divergence 预留给后续 adaptive N / HIL 触发 |
| `client/bridge.py` | 改 | `BridgeConfig` 新增 `sample_n` / `verifier_gripper_weight`（默认 0.03）；新增 `_extract_action_candidates` 保留 N 维；`_infer_chunk` 双路径（sample_n=1 走原路）；CLI 新增 `--sample-n` / `--verifier-gripper-weight` |
| `client/openpi_ws_client.py` | 改 | `infer(observation, sample_n=1)` |

### 关键设计决策

1. **不改模型**：`pi0.sample_actions` 的 batch 维由 `observation.state.shape[0]` 决定，batch=N 时自动采 N 个独立 noise。prefix（vision-language backbone）在 batch=N 下 GPU 并行前向——wall-clock 接近单次，即「amortized sampling」的务实形态。真正的 prefix KV-cache 复用（prefix 算 1 次 + suffix 采 N 次）是后续性能优化点，M1 未做。
2. **完全向后兼容**：`sample_n=1` 时协议、形状、行为与原来逐字节一致（batch 维 squeeze 逻辑保持不变，只是挪了位置）。
3. **sample_n 走 obs 通道**：websocket 协议无头字段，复用 obs dict 注入、服务端 pop，老客户端零影响。
4. **verifier 在 bridge 端本地运行**：N 个候选经网络传回后本地选优，无需第二次 GPU 往返；O(N²·T·D) 计算量可忽略（8²×10×7≈4.5K flops）。

## 二、验证情况

- `py_compile`：openpi 4 文件 + client 4 文件全部通过。
- verifier 单元自测（numpy，无硬件）3 场景通过：
  1. 3 聚簇 + 1 离群 → 正确选出聚簇内候选；
  2. 夹爪 +50mm 偏差场景：不加权时距离被夹爪主导，加权（0.03）后正确忽略夹爪偏差选聚簇；
  3. N=1 直通。
- **未做真机/GPU 测试**（本机无硬件），JAX jit 对 batch=1 与 batch=N 会各自编译一次，首轮 N 采样会有编译停顿，属预期。

## 三、遇到的问题与解决

| 问题 | 解决 |
|------|------|
| `A10Outputs` 原实现 `ndim!=2` 直接报错，批量输出 (N,T,D) 会被拒 | 改为接受 ndim∈{2,3} + `actions[..., :7]` |
| `AbsoluteActions` transform 是否兼容 batch 维？ | 读源码确认用 ellipsis 索引（`actions[..., :dims] +=`），天然兼容，无需改 |
| `BasePolicy` 接口只有 `infer(obs)`，直接加参数会破坏第三方实现 | 服务端仅在 `sample_n>1` 时传参；`sample_n=1` 走原调用。第三方 policy 被请求批量采样会 TypeError 并返回错误文本——可接受，暂未加 fallback |
| `PolicyRecorder` 包装后 `infer(obs)` 单参数，sample_n 会 TypeError | 改为 `**kwargs` 透传 |

## 四、需要后期 check 的点（真机/GPU 验证时）

1. **首轮 jit 编译停顿**：batch=N 首次推理触发重新编译，pilot 时先空跑几轮预热，别计入时延数据。
2. **batch=N 的实际 wall-clock**：需实测 N∈{2,4,8} 时 infer 耗时相对 N=1 的增长曲线，确认「amortized」假设在 5090 上成立（prefix batch=N 并行 vs 真复用 KV cache 的差距要量化）。若增长接近线性，需提前做 prefix 复用优化。
3. **夹爪权重 0.03 是否合适**：基于「夹爪 0–100mm vs 关节 rad 量级」拍的值，pilot 时看 verifier 日志中 divergence 的维度构成再校准。
4. **绝对关节空间距离语义**：verifier 在「绝对关节目标」空间算距离（推理输出经 AbsoluteActions 转换后），聚簇假设在关节空间的合理性需真机数据确认。
5. **`sample_n` 响应字段**：服务端在 sample_n>1 时返回里塞了 `sample_n` 字段，bridge 端未消费，仅调试可见。
6. **hunger 风险**：N 增大使 T_infer 变长，async 模式下若 T_infer > T_exec 会出现 buffer 饥饿（机器人短暂停顿），pilot 时观察 buffer 深度日志。

## 五、下一步（M2 pilot）

```bash
# GPU 端
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_a10_finetune \
  --policy.dir=checkpoints/pi05_a10_finetune/Reach_5_9_1/130000

# 机器人端桥接：B2 (async + 4候选 + medoid)
python3 run_bridge.py \
  --policy-host <GPU_IP> --policy-port 8000 \
  --robot-host 127.0.0.1 --robot-port 8080 \
  --mode async --camera-mode threaded \
  --sample-n 4 --hz 10

# 对照 B1: --sample-n 1（原行为）
```

Pilot 方案：Pick-and-Place 任务，B1 vs B2 各 20 trials，记录成功率 + verifier divergence 日志。
