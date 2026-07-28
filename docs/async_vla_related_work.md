# AsyncVLA Related Work 库

> 调研日期：2026-07-28
> 用途：论文 related work / positioning 直接引用；每篇一句话定位 + 与 AsyncVLA 的差异。
> BibTeX 见同目录 `async_vla_refs.bib`。

## 一、Test-Time Compute for VLA（核心对标方向）

| 工作 | 一句话定位 | 与 AsyncVLA 差异 |
|------|-----------|-----------------|
| RD-VLA (arXiv 2602.07845) | latent recurrent 迭代推理实现隐式 TTC，恒定内存 | 改 base 架构 + 仿真；我们不动 base model + 真机 |
| VLA-ATTC (arXiv 2605.01194) | cognitive clutch 触发 + Relative Action Critic 相对打分 | 仿真 + 自动构造偏好对；我们真机 + HIL 人类偏好对 + 异步流水线隐藏采样 |
| ElegantVLA (arXiv 2605.29438) | phase-adaptive scheduler 分配 backbone/action-head 计算 | RL 学 scheduler 改计算层级；我们固定 backbone 只调采样数 |
| RoboMonkey (CoRL 2025) | sample N + VLM verifier 选优，1.5Hz | 频率真机不可行；我们异步双缓冲保 15Hz+ |
| CoVer-VLA | 层级 verification 采样选优 | 同上，仿真 + 重 verifier |

## 二、Online Adaptation / Post-Training（相邻方向）

| 工作 | 一句话定位 | 与 AsyncVLA 差异 |
|------|-----------|-----------------|
| Agentic-VLA (arXiv 2605.22896) | 训练侧 agentic 框架：reward synthesis + language-guided exploration + experience memory | 仿真训练侧（真机部署是其 future work）；我们推理侧 TTC，正交可结合 |
| SOP (arXiv 2601.03044, AgiBot) | 机器人机群在线分布式 post-training，云端集中更新 | 需机群；我们单机 |
| TT-VLA | 推理时 RL：progress reward 在线适配 | 需 reward 模型；我们推理时选优不更新参数 |
| ROAD-VLA | advantage-guided self-distillation 稳定在线适配 | 训练侧 |
| Success Memory | 非参数成功轨迹检索引导采样 | 需轨迹库；我们轻量 verifier |

## 三、HIL Post-Training（交互信号来源）

| 工作 | 一句话定位 | 与 AsyncVLA 差异 |
|------|-----------|-----------------|
| BORA (arXiv 2605.30226) | offline critic + 冻结 VLA + HIL 残差 chunk 适配 | 灵巧手 + 训练侧残差 RL；我们单臂 + 推理时选优 |
| HiL-ResRL (arXiv 2606.22860) | model-agnostic 残差 RL 适配器 + HIL 引导探索 | 更新参数（RL）；我们推理时 TTC 不更新 |
| DexHiL (arXiv 2603.09121) | 灵巧手 VLA 的 HIL post-training + 介入感知数据采样 | 灵巧手；我们单臂工业任务 |
| HandITL (arXiv 2605.15157) | 双臂灵巧手无缝介入（retargeting 消除 gesture jump） | 解决介入连续性；我们用介入数据训 verifier 而非直接 SFT |

## 四、Continual Learning（实验设计参考）

| 工作 | 一句话定位 | 与 AsyncVLA 差异 |
|------|-----------|-----------------|
| ContinualVLA (arXiv 2605.26820) | 真机持续学习 empirical study：灾难性遗忘 + replay 因素 | 仅 study 无方法；可参考其真机实验设计 |

## 五、定位一句话

> AsyncVLA = 第一个真机 VLA TTC 系统：异步推理流水线把 GPU 空闲窗口变成免费采样预算（vs 仿真 TTC 全改架构或降频率），HIL 人类偏好对训练轻量 RAC（vs 仿真自动构造），自适应预算保实时（vs RoboMonkey 1.5Hz）。
