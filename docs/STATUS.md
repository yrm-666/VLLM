# 项目状态与上卡边界

## 本地已完成

- [x] 复核 Molmo2 官方最新 README、HF 模型卡和视频处理源码；
- [x] 确认 HF 视频 decode/sample/processor/generate 调用链；
- [x] 确认最小侵入接入方式和原始时间戳保留方式；
- [x] 官方式 `uniform_last_frame` 候选采样；
- [x] Query + 全部多选项构造，不接触 ground truth；
- [x] 冻结 SigLIP2 双编码器适配器；
- [x] relevance/diversity/temporal coverage 选择器；
- [x] short-video、tie-break、时间排序和输入校验；
- [x] SQLite 帧特征缓存；
- [x] 预选帧接 Molmo2 HF processor，关闭二次 sampling；
- [x] Official-Original、Uniform-32/64 和三组消融；
- [x] 单视频与 JSONL manifest runner；
- [x] MVBench 官方 annotation 转 manifest；
- [x] accuracy、frame、token、latency、TTFT、VRAM 日志；
- [x] 中断续跑、结果汇总和跨模式比较；
- [x] 纯 CPU 全链路替身测试；
- [x] 4090/4090D 环境检查和执行手册。

## 远程 GPU 已确认（2026-10-09，用户运行日志）

- [x] Ubuntu 20.04、4090D 24 GB、Python 3.11、PyTorch 2.8 cu128 环境；
- [x] Molmo2-4B 和 SigLIP2 下载、真实视频推理、离线模式、缓存命中；
- [x] MVBench scene-transition 首 20 题共 9 组配置；
- [x] Uniform-32/16/8 均 18/20；Query-aware-16 为 17/20，-8/-32 为 18/20；
- [x] 时间覆盖审计：V1 K=16/8 分别有 9/17 题未覆盖四个帧序号分区。

共 180 次推理仍只有 20 个独立样本。没有证据证明 query-aware 带来精度或延迟优势。
Uniform-8 的视觉 token 为 648，41 帧参考为 3321；这些是开发集结果，不是完整基准成绩。

## 本轮 V2 与评估修补

- [x] 独立 `query_aware_v2` 模式，非空时间分区覆盖约束；保留 V1 权重；
- [x] 增量最近距离计算与 V1 逐步评分回归测试；
- [x] 单视频/批量统一计时，加载单列，冷/热特征缓存、预热与输出 token；
- [x] 时间区间解码、原始帧索引保留、转换偏移、重复 ID 检查；
- [x] 失败样本记录、配置续跑保护、配对样本检查；
- [x] 独立视频验证集准备脚本与数据 revision 来源记录；
- [ ] V2 真实 GPU 回归、区间任务/新下载脚本远程验证；
- [ ] 新 20 题独立视频验证以及跨任务实验。

## 尚未完成或确认

- [ ] K=64 显存及 TTFT 实测；
- [ ] 根据 pilot 决定是否值得接官方 native evaluator；
- [ ] 形成经过独立验证、足以支持简历主张的算法收益。

本机没有 NumPy、PyTorch、Transformers、视频解码器或 CUDA，因此不能诚实地把后半部分标记为完成。当前代码已经把这些依赖全部隔离为延迟导入；GPU 端第一次执行如果暴露第三方 ABI 或 remote-code 版本问题，应在服务器修复并把修复提交回本仓库。
