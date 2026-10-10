# 第二轮：语言先验诊断与动作顺序跨任务验证

算法冻结：V1/V2 和权重不变。先诊断，再收集新任务结果，不进行训练。
text-only 与视频模式用同一问题、全部选项、提示模板和模型，不传答案；
不解码视频、不加载 SigLIP2、不传假图像。它是无视觉条件诊断，不是公平速度竞争者。
高分只提示存在语言先验，不能直接证明数据泄漏；低分也不能证明视频模式一定利用了视频。

## 同步与测试

在 Windows 仅提交本轮代码/文档，勿提交 .pth 或 minimind_checkpoint_backups。
在 Ubuntu：

```bash
conda activate efficient-molmo2
cd ~/projects/VLLM
git pull --ff-only
python -m unittest discover -s tests -q
```

editable 安装会自动读取代码改动，无需再装依赖。

## 先跑已有验证集的无视频对照

```bash
HF_HUB_OFFLINE=1 HF_ENDPOINT=https://huggingface.co \
python scripts/run_manifest.py \
  --manifest manifests/mvbench-scene-validation20-v2.jsonl \
  --output outputs/scene-validation20-text-only-r1.jsonl \
  --mode text_only --dtype bfloat16 --max-new-tokens 64 --warmup 2

python scripts/compare_results.py \
  outputs/scene-validation20-schema2-uniform-k8-cold-r1.jsonl \
  outputs/scene-validation20-schema2-query_aware-k8-cold-r1.jsonl \
  outputs/scene-validation20-schema2-query_aware_v2-k8-cold-r1.jsonl \
  outputs/scene-validation20-text-only-r1.jsonl
```

text-only 的实际候选、所选帧、visual tokens 必须都是 0；未测 TTFT 保持 null。
CPU 测试验证了没有视觉输入的调用边界；真实 remote-code text-only 兼容性须上卡确认，
若失败会记录错误，不能偷偷替换成空白视频。

## 新任务：Action Sequence

官方任务映射为 `star/Charades_v1_480`，需要 start/end：
[MVBench 官方评测示例](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/mvbench.ipynb)。
本项目保留原始视频帧编号，选取 `[start,end)` 内的帧起始时间；与 native 的
round 边界/分段采样不是严格相同，因此本轮是统一模型下的内部配对评测。

HF 的 video 分支实际存放 `star/Charades_segment/<video>_<start>_<end>.mp4`，
不是上述 native 本地目录。下载器查询固定 revision 的真实文件列表，按源视频 ID 和
数值起止时间唯一匹配；缺失/歧义直接报错，不换题。匹配片段已经裁剪，因此 manifest
只保存 source_start/source_end 用于来源记录，不传 start_seconds/end_seconds 再裁剪。
输入时间戳/帧编号相对于片段。预检查核对片段时长（容差 max(0.25秒,2帧)）。
不宣称片段编码/裁剪与 native 原视频路径逐位等价。

[HF 视频目录](https://huggingface.co/datasets/OpenGVLab/MVBench/tree/video/star)

```bash
HF_HUB_OFFLINE=0 HF_ENDPOINT=https://huggingface.co \
python scripts/prepare_action_validation.py \
  --data-root "$MOLMO_DATA_DIR/mvbench-round2" \
  --output manifests/mvbench-action-validation20-round2.jsonl \
  --offset 0 --count 20

python scripts/check_manifest_videos.py \
  --manifest manifests/mvbench-action-validation20-round2.jsonl
```

数据写外置盘，manifest 与来源记录写代码目录。按原标注顺序选 20 个不同视频，
不按答案/结果筛题。若视频分支下载布局不可用，保留错误，不能把缺失样本换成容易题。
预检查只解码 8 帧、不加载模型；任何时间区间错误先排查，不执行下一阶段。
这些脚本的下载接口与真实数据解码仍需远程验证。

预检查全部通过，再执行（共 6 组 120 次计分推理）：

```bash
for k in 8 16; do
  for mode in uniform query_aware query_aware_v2; do
    HF_HUB_OFFLINE=1 HF_ENDPOINT=https://huggingface.co \
    python scripts/run_manifest.py \
      --manifest manifests/mvbench-action-validation20-round2.jsonl \
      --output "outputs/action-validation20-${mode}-k${k}-cold-r1.jsonl" \
      --mode "$mode" --num-candidates 128 --num-selected "$k" \
      --max-fps 2 --selector-batch-size 8 --dtype bfloat16 \
      --max-new-tokens 64 --cache-state cold --warmup 2 || break 2
  done
done
```

新任务另跑一次 text-only（不必按 K 重复），然后每个 K 单独比较三个视频模式及 text-only。
输出不覆盖，任何失败先保留记录并排查。优先检查配对逐题得失和覆盖，耗时正式主张须重复测量。
