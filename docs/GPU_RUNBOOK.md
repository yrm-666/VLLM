# 4090 / 4090D 上卡执行手册

已有环境和首轮 pilot 的用户，请直接看 [V2 回归手册](V2_RUNBOOK.md)，无需重新安装。

## 推荐机器

- 单卡 RTX 4090 或 RTX 4090D，24 GB；
- 16 vCPU、64 GB 内存；
- 至少 200 GB NVMe，推荐 300 GB；
- Ubuntu；
- Python 3.11；
- BF16、batch size 1、SDPA；
- 第一轮 `M=128, K=32`。

4090D 与 4090 显存同为 24 GB，能承担 Molmo2-4B BF16 的首轮实验。4090D 计算较慢，但项目瓶颈更可能是长上下文 prefill 和视频处理。`Official-Original` 的 384 帧可能因为上下文和 rotary 路径风险失败；这项失败也必须如实记录，不能伪装成“Full 全部原始帧”。

## 安装

```bash
conda create -n efficient-molmo2 python=3.11 -y
conda activate efficient-molmo2

# 按服务器驱动，从 pytorch.org 选择对应 CUDA 命令，同时安装 torch/torchvision。
# 不要盲目复制另一个镜像的 CUDA wheel。

pip install -r requirements-gpu.txt
pip install -e .
python scripts/check_gpu_environment.py
```

官方模型卡当前固定 `transformers==4.57.1`。Molmo2 官方仓库要求 Python >=3.11，并建议 `torchcodec` 单独安装；本项目首轮用 `decord2`，减少 torch/torchcodec ABI 组合问题。

不要在本地上传模型和数据。代码经 Git 迁移后，在服务器设置缓存盘：

```bash
export HF_HOME=/data/huggingface
export MOLMO_DATA_DIR=/data/molmo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 最小下载顺序

1. `allenai/Molmo2-4B`；
2. `google/siglip2-base-patch16-224`；
3. 一条本地测试视频；
4. MVBench 小样本；
5. 确认流程稳定后再下载完整 MVBench；
6. NeXTQA、Video-MME 延后。

不要一开始下载 8B、O-7B、三个 benchmark 和所有官方 checkpoints。

## 上卡后第一组命令

单视频 smoke：

```bash
python scripts/run_video_qa.py \
  --video /data/demo.mp4 \
  --question "Which object does the person pick up?" \
  --option "A book" --option "A cup" --option "A phone" --option "A ball" \
  --mode query_aware --num-candidates 128 --num-selected 32 \
  --measure-ttft
```

JSONL 小样本：

```bash
python scripts/run_manifest.py \
  --manifest manifests/mvbench-pilot.jsonl \
  --output outputs/mvbench-query-aware-32.jsonl \
  --mode query_aware --num-candidates 128 --num-selected 32 \
  --limit 20
```

MVBench 官方 JSON 可逐 task 转换；`--video-root` 指向该 task 对应的解压视频目录：

```bash
python scripts/convert_mvbench.py \
  --annotations /data/MVBench/json/action_sequence.json \
  --video-root /data/MVBench/video/star/Charades_v1_480 \
  --task action_sequence \
  --output manifests/mvbench-pilot.jsonl \
  --limit 20
```

不同 task 的视频来源目录不同，应使用官方文件结构逐项指定，不能把不存在的视频静默纳入评测。NTU RGB+D 子集受许可约束，需要手动获取；首轮 pilot 可以先选已完整下载的 task。

依次换成：

```text
official_original
uniform
relevance_only
relevance_diversity
query_aware
```

结果汇总：

```bash
python scripts/summarize_results.py outputs/mvbench-query-aware-32.jsonl

python scripts/compare_results.py \
  outputs/mvbench-uniform-32.jsonl \
  outputs/mvbench-relevance-32.jsonl \
  outputs/mvbench-relevance-diversity-32.jsonl \
  outputs/mvbench-query-aware-32.jsonl
```

## 显存处理顺序

发生 OOM 时依次：

1. 确认 BF16 和 batch size 1；
2. 先关闭 `--measure-ttft`，它会额外跑一次一-token generation；
3. 保持 `K=32`，降低 selector image batch size；
4. 将 SigLIP2 放 CPU；
5. 再考虑量化或 CPU offload。

首轮不要直接用量化结果作为主 baseline，因为量化会引入额外变量。
