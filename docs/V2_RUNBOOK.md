# V2：覆盖约束与可复现实验

## 改动边界

没有训练或修改 Molmo2/SigLIP2 的网络。`query_aware` 保留原评分和确定性 tie-break；
`query_aware_v2` 先为每个非空时间分区选一帧，再按同一评分补满 K。默认四分区，
K 小于 4 时分区数降为 K。分区依据完整视频/标注区间的时间边界，不依据候选帧序号。
保证覆盖的是有候选帧的分区，不保证选首尾帧，也不保证准确率提高。

增量维护候选帧到已选集合的最近距离，将距离比较从 O(M K² D) 降为 O(M K D)。
CPU 回归测试核对了旧 V1 的每一步评分和顺序；GPU 延迟收益尚未验证。

新增日志：四分区覆盖、首末选帧跨度、输出 token 数、错误类型、解析后的模型 revision、
manifest SHA256。模型加载单独计时；单视频和批量入口使用相同的逐样本计时路径。
带标注起止时间的任务仅解码 `[start,end)`，保留原始视频帧编号和完整视频元数据。
这套区间约定仍需与官方 native evaluator 配对核验。

现有 `official_original` 是本项目的官方式采样参考，不是严格 native 复现：
保留旧 `round` 索引规则，官方某些路径使用整数截断。日志明确记为 `legacy-round-v1`。
尚未接入 native evaluator，不能宣传达到官方成绩。

## 1. 将代码同步到 Ubuntu

先在 Windows 项目目录检查改动，提交并推送自己的仓库。不要上传模型、数据或缓存。
然后在远程 Ubuntu 执行：

```bash
conda activate efficient-molmo2
cd ~/projects/VLLM
git status --short
git pull --ff-only
python -m pip install --no-deps -e .
python -m unittest discover -s tests -q
```

若远程有未提交改动或 pull 失败，先保存改动并处理；不要用 reset --hard。
当前环境依赖已装好，无需升级系统、驱动或重建 Conda 环境。

## 2. 先用原 20 题回归，三个模式同预算

在 Ubuntu 中整段复制。三个输出文件均为新文件，旧结果不覆盖。
先检查 K=8，再把 `k=8` 改为 `k=16` 重跑。

```bash
k=8
for mode in uniform query_aware query_aware_v2; do
  HF_HUB_OFFLINE=1 HF_ENDPOINT=https://huggingface.co \
  python scripts/run_manifest.py \
    --manifest manifests/mvbench-scene-pilot20.jsonl \
    --output "outputs/scene-dev20-schema2-${mode}-k${k}-cold-r1.jsonl" \
    --mode "$mode" --num-candidates 128 --num-selected "$k" \
    --max-fps 2 --selector-batch-size 8 --dtype bfloat16 \
    --max-new-tokens 64 --cache-state cold --warmup 2 || break
done
```

`cold` 指禁用图像特征缓存，不是冷磁盘、未加载模型或清空 CUDA 缓存；
预热样本不计入结果，测量时仍重新编码图像。

```bash
python scripts/compare_results.py \
  outputs/scene-dev20-schema2-uniform-k8-cold-r1.jsonl \
  outputs/scene-dev20-schema2-query_aware-k8-cold-r1.jsonl \
  outputs/scene-dev20-schema2-query_aware_v2-k8-cold-r1.jsonl
```

优先检查 `failed_examples=0`、三个模式样本一致、V2 的 `temporal_bins_covered`。
候选池有四分区且 K≥4 时，V2 应覆盖四分区。旧人工审计按帧序号分区，
新指标按时间分区，分母也包含末帧时长；数值不要求与旧审计逐位相同。

## 3. 再固定独立视频的 20 题验证

先冻结算法与参数，再下载验证集；不要看验证结果后反复调权重。
脚本按标注顺序从第 21 条起选，排除开发集视频和重复视频，保留原标注 ID。
下载只写入指定外置盘数据目录，manifest 写入代码目录。脚本需要联网，勿加离线变量。

```bash
HF_HUB_OFFLINE=0 HF_ENDPOINT=https://huggingface.co \
python scripts/prepare_scene_validation.py \
  --data-root "$MOLMO_DATA_DIR/mvbench-v2" \
  --development-manifest manifests/mvbench-scene-pilot20.jsonl \
  --output manifests/mvbench-scene-validation20-v2.jsonl \
  --offset 20 --count 20
```

把上面的运行循环改成新 manifest 和 `scene-validation20` 输出前缀，分别跑 K=8/16。
脚本保存数据集两个分支的 commit、样本索引及 SHA256；下载路径和网络仍需远程验证。
这仍是单任务小样本验证，不能替代完整、多任务 MVBench 评测。

## 公平计时与异常处理

- 正式延迟对比每种模式重复至少三次，分别保存 r1/r2/r3；同预算、同视频、同模型 revision。
- 热缓存另跑 `--cache-state warm`：测量前预跑全部待测样本。会额外运行一遍，
  热缓存收益依赖同一视频重复请求，不能与首次请求性能混称。
- `reuse` 沿用缓存，不保证冷热；新编码器指纹包含 revision 与处理器设置，旧缓存保留但首次会重算。
- `.run.json` 记录加载与预热时间；逐样本 `end_to_end_seconds` 不含加载。
  峰值显存是 PyTorch allocated，不是 nvidia-smi 的整卡占用。
- 未开启 `--measure-ttft` 时 TTFT 为 null；开启后是额外一次单 token generate 的近似值，
  不是流式首 token 时间，也不是纯 prefill，额外成本会进入端到端计时。
- 出错先写 JSONL，再退出。`--continue-on-error` 可继续；最终退出码仍为 1，
  失败题记错，成功样本的耗时均值不混入失败占位值。
- `--resume` 不重试已记录的失败题，且必须配置、manifest 与 revision 一致。
  修复失败后使用新输出重跑。旧 schema1 结果不能直接续写 schema2。
- 不把旧单视频含下载的耗时与新逐样本耗时比较。比较脚本校验样本集合和共有关键配置，
  每次只比较一个 K；K=8 与 K=16 分开比较。
