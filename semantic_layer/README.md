# 对象级视频语义层 — 原型 v0.2

一次本地分析，之后无限次本地查询，零 token。

给视频建一份"AI 能直接读懂的目录"：每个出现过的对象都有向量标注和
时间轴位置（主角有 VLM 描述、配角/龙套只留向量），场景描述和语音
转写也按时间戳进同一个统一时间轴文件。查询只在本地做向量检索，
返回的是**精确到秒的出现窗口**，而不是整段视频。

## 快速开始

```bash
# 1. 构建索引（首次需下载模型，见 download_repo.py）
python3 semantic_layer.py build <video.mp4> -o <idx_dir> --fps 1 --mode auto

# 2. 中文/英文文字查询（对象描述 + 场景 + ASR 三层融合）
python3 semantic_layer.py query <idx_dir> --text "口红" -n 5

# 3. 参考图查询（稀有对象用图搜图）
python3 semantic_layer.py query <idx_dir> --image ref.jpg -n 5

# 4. 查看统计
python3 semantic_layer.py info <idx_dir>

# 5. AI 剪辑规划：规则 -> 剪辑清单（带证据）
python3 semantic_layer.py plan <idx_dir> --text "口红" --objects lipstick --keywords 口红
```

默认 ASR 为 **large-v3**：token 消耗与 tiny 相同（token 只跟字数有关），
但转写精度大幅提升（实测"粉底气垫"类查询从全灭到命中，见
TOKEN_COMPARISON.md 精度审计）。代价是构建时间变长
（89 秒音频：tiny 6.4s → large-v3 117s）。

生产演示索引 [demo_clip03_v3](demo_clip03_v3) 即用 large-v3 构建；
[demo_clip03_v2](demo_clip03_v2) 保留 tiny 基线，供精度对比。

## v0.2.1 新增：自动分诊（auto mode）

编码前先花 ~20 秒快速探测两个信号：

- **语音密度**：tiny ASR 转一次，算语音覆盖时长占比
- **对象密度**：抽 4-8 帧做开放词汇检测，算平均每帧检测数

然后自动选择索引模式：

| 探测结果 | 模式 | 成本 |
|---|---|---:|
| 语音覆盖 ≥ 50% 且对象密度 ≤ 1.5/帧 | **asr_only**：只转写+场景边界 | ~8 秒（18× 快） |
| 语音覆盖 < 30% 且对象密度高 | **avis_full**：视觉主导，完整对象层 | ~2 分钟 |
| 其余（直播电商典型） | **avis_full**：图文平衡 | ~2 分钟 |

实测：纯语音视频（灰屏+音频）被正确判为 `asr_only`，7.5 秒建完索引
（语音覆盖 99.8%，对象密度 0）；直播切片自动走 `avis_full`。

分诊证据会写进 manifest.json 和 avis.json 的 `mode` 字段，随时可查：

```json
"mode": {
  "mode": "asr_only", "reason": "speech_dominant",
  "speech_coverage": 0.998, "transcript_chars": 473,
  "object_density": 0.0, "labels_probed": [],
  "probe_frames": 8
}
```

也可以手动指定：`--mode asr_only` 或 `--mode avis_full`。

## v0.2 新增：统一时间轴 + AVIS 集成

构建输出现在同时是 AVIS 兼容目录（`avis.json` 里声明了新的 objects
信号），核心是一个**对象标注 + 向量共用同一时间轴**的文件
`timeline.jsonl`，每行一条时间轴条目：

```json
{
  "type": "object", "track_id": 15, "label": "lipstick",
  "tier": "配角", "appearance": 0, "t0": 8.0, "t1": 15.0,
  "description": "lipstick",
  "emb_type": "visual", "emb_index": 30,
  "emb": [0.0226, -0.0448, ...]   // 512 维向量，与标注同文件
}
```

对象按"出现片段"（间隔 >3 秒算一次新出现）切分，所以查询能定位到
`24.0-26.0s` 这样的精确窗口，而不是整条轨迹。

### 大小测试（89 秒 720x1572 直播切片）

| 文件 | 大小 | 说明 |
|---|---:|---|
| timeline.jsonl | **638.5 KB** | 对象标注+向量+场景+ASR，同一时间轴 |
| embeddings.npz | 316 KB | 二进制向量，查询时实际使用 |
| objects.jsonl | 9.4 KB | 对象出现片段（无向量，人读版） |
| scenes.jsonl / asr.jsonl | ~20 KB | 场景与转写元数据 |
| avis.json + manifest.json | ~2 KB | 清单 |
| **语义层合计** | **~1 MB** | 不含缩略图 |
| 全套（含 52 张缩略图） | 3.8 MB | 可分享的完整目录 |

对比原始视频（89 秒，约 15-25 MB），语义层只占原始视频的 4-7%，
且之后每次查询不再需要解码视频。

## 演示结果（样本7 直播切片，89 秒）

构建：89 帧 → 614 检测 → 52 轨迹（13 主角 / 20 配角 / 19 龙套）
→ **59 个出现片段** → 8 场景描述 → 47 段 ASR。
耗时 **135 秒**（1.5× 实时）。

中文查询 "口红"：

```
[1] 68.6-69.2s score=0.888 layers=asr
    现在下达美好领旧会给大家送了一只口红
[4] 84.0-88.0s score=0.596 layers=objects,scenes
    lipstick (#52, 龙套) | lipstick (#28, 配角) | ...
```

参考图查询（用口红轨迹缩略图当参考）：

```
[1] 24.0-26.0s score=0.938  lipstick (#23, 配角)
[2] 30.0-30.0s score=0.935  lipstick (#28, 配角)
[4] 70.0-80.0s score=0.927  a woman with a pink lipstick and a black shirt (#30, 主角)
```

注意第 4 条：主角轨迹 #30 的返回文本是 BLIP 生成的描述
（"a woman with a pink lipstick and a black shirt"），而不是类别名
——**主角理解层**已经生效：主角花理解的钱，配角只花记录的钱。

## 目录格式（AVIS 兼容）

```
<idx_dir>/
├── avis.json          # AVIS 清单，signals.objects 为新信号
├── timeline.jsonl     # 统一时间轴：对象/场景/ASR + 向量
├── objects.jsonl      # 对象出现片段（人读版）
├── scenes.jsonl       # 场景描述
├── asr.jsonl          # 语音转写
├── embeddings.npz     # 二进制向量（obj / obj_text / scene / frame / asr / vocab）
├── tracks.png         # 对象轨迹总览
└── thumbs/            # 每个轨迹的缩略图
```

`avis.json` 中的对象信号：

```json
"objects": {
  "path": "objects.jsonl", "timeline": "timeline.jsonl",
  "embeddings": "embeddings.npz",
  "tracks": 52, "appearances": 59,
  "tiers": {"主角": 13, "配角": 20, "龙套": 19}
}
```

## 模型与依赖

全部本地推理（Apple Silicon MPS），模型在 `work/models/`：

- 检测：IDEA-Research/grounding-dino-base（开放词汇）
- 视觉向量：openai/clip-vit-base-patch32
- 场景+主角描述：Salesforce/blip-image-captioning-base
- 文本向量：sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- ASR：Systran/faster-whisper-tiny

运行环境：Python 3.13，torch + transformers + av + faster-whisper + PIL。
下载模型：`python3 download_repo.py`（走 hf-mirror，已绕开 SSL 与
重定向问题）。如遇 SSL 错误：`SSL_CERT_FILE=$(python3 -m certifi)`。

## 成本账

| 项目 | 数值 |
|---|---:|
| 建立索引 | ~1.5× 实时（89 秒视频 135 秒）|
| 语义层体积 | ~1 MB（含向量），全套 3.8 MB |
| 查询 | 本地向量运算毫秒级；冷启动加载模型 3-6 秒 |
| token | 建索引 0（全本地）；查询 0；只有深度理解命中片段才花 |

## 已知限制（下一步）

1. 主角连续出现时会是一个长窗口（如 face 0-88s），因为确实全程在；
   细分到"动作事件"需要再加一层场景/姿态聚类。
2. 检测 1.7s/帧，长视频要接 AVIS 的 MV/ASR 先缩范围再检测。
3. 同类别多对象靠 IoU 跟踪，快速运动/遮挡会断轨，下一步换
   ByteTrack + 外观 Re-ID。
4. tiny 中文 ASR 有噪声；换 base/small 可改善。
5. 增量索引：查询未命中时按新提示词补检测，不重算已有数据。

---

# MCP Server：agent 视频理解的 "ffmpeg" 原型

[avis_mcp_server.py](avis_mcp_server.py) 把语义层包装成 MCP 工具，
任何支持 MCP 的 agent（Claude、Codex、Cursor 等）都能直接调用：

```bash
python3 avis_mcp_server.py   # stdio 模式
```

暴露 4 个工具：

| 工具 | 作用 | token |
|---|---|---:|
| encode_video | 视频 → 语义层索引（一次编码） | 0 |
| search_index | 文字/参考图查询 → 时间轴窗口 | 0 |
| understand_video | 查询 → 命中片段 → 本地 VLM 理解 → 结论 | 数百 |
| plan_clips | 剪辑规则 → 带证据的剪辑清单（t0/t1/语音/原因） | 0 |
| index_info | 索引统计 | 0 |

真实调用示例（MCP 客户端，89 秒索引，查询"口红"）：

```json
{
  "ok": true,
  "query": "口红在视频里什么时候出现？主播介绍了什么？",
  "answer": "口红在视频里出现在第25.43秒，主播介绍了粉红色的口红。",
  "tokens": {"prompt": 713, "output": 28},
  "seconds": 4.8
}
```

这是"两段式"的最终形态：第一段免费检索定位，第二段只让本地 VLM
看命中片段。agent 拿到视频不再需要自己解码/抽帧/跑 ASR——像调
ffmpeg 一样调 `avis` 就行。

## AI 剪辑规划与效果测试

[CLIP_PLANNING.md](CLIP_PLANNING.md) 记录了 `plan_clips`（剪辑规划）
和完整测试方法论：

- 自动化代理评估（eval_clips.py）：P@k / GT召回 / 边界误差
- 实测：对象类规则 P=1.0，关键词过滤后平均 P 0.92 / R 0.72
- 三层测试法：代理真值 → 人工真值 → 编辑者打分（语义完整/画面完整/可用率）
