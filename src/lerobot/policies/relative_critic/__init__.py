"""AsyncVLA Level-2 verifier: Relative Action Critic (RAC) 训练框架。

定位：L1（几何 medoid）零训练但无任务语义；RAC 用真机 HIL 偏好对
学习「哪个候选更可能成功」，回答的是相对问题 P(a_i ≻ a_j) 而非绝对
打分——借鉴 VLA-ATTC 的 RAC 设计降低学习难度，但训练信号来自
真机人类介入而非仿真自动构造（AsyncVLA 论文核心差异点之一）。

当前状态：v1 框架（纯 proprioception + action 几何，无图像）。
偏好对数据格式约定见 dataset_preference.py；HIL 介入事件接入
（dev/hil 线）确定后在此联调。
"""
