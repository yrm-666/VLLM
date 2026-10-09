# Query-Aware Adaptive Frame Selection for Molmo2

这个仓库承载已经确定的主项目：在不训练 Molmo2、不改模型结构的前提下，根据问题从视频中选择更少但更有信息量的帧，降低视频推理成本。

当前阶段只做时间维帧选择。暂不做 SFT、RL、空间 token 剪枝或模型微调。

2026-10-09：4090D 已跑通真实推理与 MVBench scene-transition 20 题开发 pilot。
Uniform-8 达到 18/20；当前结果未证明 query-aware 优于均匀采样。
新增 `query_aware_v2` 时间分区覆盖约束，尚待远程 GPU 验证；运行步骤见
[V2 回归与独立验证手册](docs/V2_RUNBOOK.md)。不修改旧结果，也不重选项目方向。

## 已完成的本地阶段

本地先完成可独立验证的算法核心：

1. 从完整视频均匀采样最多 `M` 个候选帧；
2. 冻结的图文双编码器在 GPU 服务器上计算问题—帧相关性；
3. 贪心选择 `K` 帧，同时兼顾相关性、视觉多样性和时间覆盖；
4. 将最终帧按时间顺序交给 Molmo2，避免打乱视频叙事顺序。

选择目标为：

```text
score(i | S) = alpha * relevance(i)
             + beta  * visual_diversity(i, S)
             + gamma * temporal_coverage(i, S)
```

首轮工程默认值：`M=128`、`K=32`、`alpha=0.60`、`beta=0.25`、`gamma=0.15`。这些只是待验证起点，不是调过测试集的结论。

## 目录

```text
src/molmo2_frame_selector/
  config.py       # 可复现实验配置与参数校验
  sampling.py     # 均匀候选帧索引
  selector.py     # 与模型框架无关的贪心选择核心
tests/             # 纯标准库单元测试
```

核心算法测试不需要 GPU、PyTorch 或模型权重；生产入口会在服务器上按需加载 SigLIP2 和 Molmo2。当前已经完成：

- 官方式 `uniform_last_frame` 候选索引（保留 round 规则，尚未严格对齐 native evaluator）；
- 冻结 SigLIP2 双编码器适配器；
- relevance/diversity/coverage 贪心选择；
- 原始帧索引和时间戳保留；
- SQLite 图像特征缓存；
- Molmo2 HF 预解码帧接入，显式关闭二次采样；
- Official-Original、Uniform、三组消融的统一实验入口；
- JSONL 指标记录和结果汇总；
- CPU 假编码器/假模型端到端测试。

## 本地验证

无需安装依赖：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

也可以用预计算特征走完整的命令行路径。示例配置中的 `K=32`，而示例只有 6 帧，所以 6 帧都会保留；这也同时验证了短视频不复制帧的规则：

```powershell
$env:PYTHONPATH = "src"
python -m molmo2_frame_selector `
  --input examples/synthetic_features.json `
  --config configs/selector_k32.json
```

输入中的 `query` 只用于日志。真正的 query-aware 相关性由后续 SigLIP2 适配器生成，并写入 `relevance_scores`；选择器本身不绑定任何模型实现。

完整设计与执行文档：

- [官方代码调用链与接入点](docs/CODE_PATH.md)
- [4090/4090D 执行手册](docs/GPU_RUNBOOK.md)
- [第一阶段实验协议](docs/EXPERIMENT_PROTOCOL.md)
- [当前完成度与 GPU 边界](docs/STATUS.md)
- [V2 回归与独立验证](docs/V2_RUNBOOK.md)

## 迁移原则

代码通过 Git 迁移到 GPU 服务器；模型权重和数据集在服务器直接下载，不提交到仓库。正式环境建议 Python 3.11、单卡 24 GB、BF16、batch size 1，先跑 `K=32`，再实测 `K=64`。
