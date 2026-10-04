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

## 必须上 GPU 才能完成或确认

- [ ] 安装与服务器驱动匹配的 CUDA PyTorch；
- [ ] 下载 `allenai/Molmo2-4B` 和 `google/siglip2-base-patch16-224`；
- [ ] 用真实 Decord/Numpy frame array 验证 HF remote-code processor；
- [ ] 实测 4090D 上 BF16 K=32/64 的显存和 TTFT；
- [ ] 下载 MVBench 可用 task，跑配对 pilot；
- [ ] 根据 pilot 决定是否值得接官方 native evaluator；
- [ ] 产生任何可写入简历的真实结果数字。

本机没有 NumPy、PyTorch、Transformers、视频解码器或 CUDA，因此不能诚实地把后半部分标记为完成。当前代码已经把这些依赖全部隔离为延迟导入；GPU 端第一次执行如果暴露第三方 ABI 或 remote-code 版本问题，应在服务器修复并把修复提交回本仓库。

