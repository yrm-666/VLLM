# Molmo2 视频推理调用链与接入位置

核对日期：2026-10-04。以下结论来自 Ai2 官方 `allenai/molmo2` 主分支及 `allenai/Molmo2-4B` 模型卡，不凭旧版 Molmo 代码推断。

## Hugging Face 调用链

```text
messages
  → Molmo2Processor.apply_chat_template(...)
  → Molmo2Processor.__call__(text=..., videos=...)
  → Molmo2VideoProcessor.preprocess(...)
  → _decode_and_sample_videos(...)
  → sample_times(...) / sample_frames(...)
  → Decord / TorchCodec / PyAV decoder
  → _preprocess(...)
  → 378×378 frame crops + 14×14 patches + 3×3 pooling
  → video placeholder replaced by timestamped visual tokens
  → AutoModelForImageTextToText.generate(...)
```

关键官方源码：

- [`processing_molmo2.py`](https://github.com/allenai/molmo2/blob/main/olmo/hf_model/processing_molmo2.py)
- [`video_processing_molmo2.py`](https://github.com/allenai/molmo2/blob/main/olmo/hf_model/video_processing_molmo2.py)
- [`modeling_molmo2.py`](https://github.com/allenai/molmo2/blob/main/olmo/hf_model/modeling_molmo2.py)
- [`test_molmo2.py`](https://github.com/allenai/molmo2/blob/main/olmo/hf_model/test_molmo2.py)
- [`Molmo2-4B model card`](https://huggingface.co/allenai/Molmo2-4B)

截至核对时，`Molmo2VideoProcessor` 默认值是：

```text
frame_sample_mode = "uniform_last_frame"
max_fps = 2
sampling_fps = 2
do_sample_frames = True
size = 378×378
patch_size = 14
pooling_size = 3×3
```

已实现的 frame sampling 分支是 `uniform_last_frame` 和 `fps`。它们只接收视频元数据，不接收问题文本，因此不能在原函数里简单增加一个 query-aware 分支：真正的相关性选择必须先看到候选帧像素。

## 本项目的最小侵入路径

```text
video metadata
  → 按官方 uniform_last_frame 语义解码最多 M 个候选帧
  → 冻结 SigLIP2 编码 question/options 与候选帧
  → relevance + diversity + temporal coverage 贪心选 K
  → 按原始时间重新排序
  → processor(videos=selected_array,
              video_metadata=original_frame_indices,
              do_sample_frames=False)
  → Molmo2.generate(...)
```

`do_sample_frames=False` 是关键。官方处理器对数组视频会跳过 `sample_frames`；`video_metadata.frames_indices` 又会被转换为真实时间戳并写入每帧前缀，所以既不会二次均匀采样，也不会丢失时间信息。

对应本地实现：

- `video.py`：候选帧解码和官方式采样语义；
- `encoders.py`：冻结 SigLIP2 双编码器；
- `pipeline.py`：特征缓存、相关性计算和选择；
- `hf_runner.py`：关闭二次采样并调用 Molmo2；
- `experiment.py`：统一 baseline/ablation/指标记录。

这条路径不修改模型权重、视觉塔、连接器或语言模型，也不需要训练。

## 原生官方评测代码的后续接入点

正式全量 benchmark 可在服务器克隆官方仓库后，把相同 selector 接到 raw example 仍保留 `question` 的预处理阶段。原则是：必须在 formatter 消耗问题文本之前取得 query，并在视频 patch/token 化之前完成两阶段解码。不要修改 `Molmo2.generate` 或视觉 transformer。

首轮先使用本仓库的 HF manifest runner 做配对小样本，确认效果与显存；随后再接官方 `launch_scripts/eval.py` / `eval_molmo2.py`，避免在 24 GB 卡上先承担整套 native checkpoint 和数据工程成本。

