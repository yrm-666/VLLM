# 第一阶段实验协议

## 要回答的问题

在相同帧预算 `K=32` 下，query-aware selection 是否比 uniform sampling 保留更多视频 QA 性能？相对官方原始采样，它用多少帧、视觉 token、时间和显存换取多少准确率？

## 固定条件

- Molmo2-4B，同一 checkpoint；
- BF16、SDPA、batch size 1；
- 相同 prompt、max new tokens、解码策略；
- 相同视频和问题顺序；
- 多选题 query 包含全部选项，但绝不包含答案；
- Query-Aware 与 Uniform 在主比较中都用 `K=32`；
- 候选池默认 `M=128`、最多 2 fps；
- 最终入模帧必须按原始时间排序。

## 实验组

| 名称 | 帧策略 | 目的 |
|---|---|---|
| Official-Original | 官方式上限 384、2 fps | 参考上界，不称作所有原始帧 |
| Uniform-64 | 官方式均匀 64 | 帧预算曲线 |
| Uniform-32 | 官方式均匀 32 | 主 baseline |
| Relevance-32 | 仅问题相关性 | 消融 |
| Relevance+Diversity-32 | 相关性与视觉去重 | 消融 |
| QueryAware-32 | 再加入时间覆盖 | 完整方法 |

默认权重 `0.60 / 0.25 / 0.15` 是工程起点。不得在测试集反复调参后把结果写成零训练先验；若调参，必须划出 validation subset 并记录搜索范围。

## 每条样本记录

- prediction、gold、是否正确；
- candidate/selected frame 数；
- selected original indices；
- estimated 与 processor actual visual tokens；
- selector、processor、TTFT、generation、端到端时间；
- peak allocated VRAM；
- cache hit/miss；
- task/category。

TTFT 当前用 `max_new_tokens=1` 的完整 generation 近似测量，包含 prefill 和首 token decode；不要把它标成纯 prefill latency。若后续加 CUDA event/hook 单独测 forward，再新增 `prefill_seconds` 字段。

## 测量规则

- 正确性实验允许复用帧特征缓存；
- 同时报告冷缓存和暖缓存 selector 时间；
- 性能计时前至少 warm-up 2 条；
- CUDA 计时前后 synchronize；
- 每种配置至少重复 3 次，报告 median；
- 主表报告总体和每个 task；
- 额外报告 `official_original_frames > 32` 子集，避免大量短视频天然不超过 32 帧掩盖效率差异；
- 所有 OOM、解码失败、无法解析答案都纳入失败统计。

## 阶段门槛

先在 MVBench 20～50 条配对 pilot 上验证：

1. 五组都能跑完；
2. selected indices/timestamps 正确；
3. Uniform-32 和 QueryAware-32 的实际 visual tokens 一致；
4. QueryAware 不是只集中在局部时间；
5. 缓存前后选择结果完全一致。

满足后再扩大 MVBench；NeXTQA 用于时间/因果能力复核；Video-MME 放在最后。第一阶段结果未稳定前不进入 spatial token pruning。

