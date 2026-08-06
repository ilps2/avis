# AVIS 语义层实现 (semantic_layer)

AVIS 协议的第一个完整实现：**对象级视频语义层**。

给视频建一份 AI 能直接读懂的目录：每个出现过的对象都有向量标注和
时间轴位置（主角有 VLM 描述、配角/龙套只留向量），场景描述和语音
转写也按时间戳进入同一个统一时间轴文件。一次本地编码，之后无限次
本地查询，零 token。

## 快速开始

```bash
# 1. 构建索引（默认 large-v3 ASR + 自动分诊）
python3 semantic_layer.py build <video.mp4> -o <idx_dir> --mode auto

# 2. 中文/英文文字查询（对象描述 + 场景 + ASR 三层融合）
python3 semantic_layer.py query <idx_dir> --text "口红" -n 5

# 3. 参考图查询（稀有对象用图搜图）
python3 semantic_layer.py query <idx_dir> --image ref.jpg -n 5

# 4. 统计
python3 semantic_layer.py info <idx_dir>
```

模型缓存于本地（Apple Silicon MPS 推理），`download_repo.py` 负责从
hf-mirror 拉取。全部依赖：Python 3.13 + torch + transformers + av +
faster-whisper + PIL + mlx-lm（MCP 理解工具用）。

## 自动分诊（--mode auto）

编码前用 ~20 秒探测语音密度 + 对象密度，自动选择：

| 探测结果 | 模式 | 成本 |
|---|---|---:|
| 语音覆盖 ≥ 50% 且对象密度低 | asr_only（纯转写） | ~8 秒 |
| 语音覆盖 < 30% 且对象密度高 | avis_full（视觉主导） | ~2-4 分钟 |
| 其余（直播电商典型） | avis_full（图文平衡） | ~2-4 分钟 |

默认 ASR 为 large-v3：token 消耗与 tiny 相同（token 只跟字数有关），
精度大幅提升（实测"粉底气垫"类查询从全灭到命中）。

## MCP Server：agent 视频理解的 "ffmpeg"

`avis_mcp_server.py` 把语义层包装成 MCP 工具，任何支持 MCP 的 agent
都能直接调用：

```bash
python3 avis_mcp_server.py   # stdio 模式
```

| 工具 | 作用 | token |
|---|---|---:|
| encode_video | 视频 → 语义层索引（一次编码） | 0 |
| search_index | 文字/参考图 → 时间轴窗口 | 0 |
| understand_video | 查询 → 命中片段 → 本地 VLM 理解 → 结论 | 数百 |
| index_info | 索引统计 | 0 |

## 示例索引

[examples/demo_clip03_v3](examples/demo_clip03_v3) 是 89 秒直播切片
的完整语义层索引（large-v3 ASR + 对象层 + 统一时间轴），可直接查询：

```bash
python3 semantic_layer.py query semantic_layer/examples/demo_clip03_v3 --text "口红"
```

## 实测结论（文档索引）

- [TOKEN_COMPARISON.md](TOKEN_COMPARISON.md) — 同一视频两种做法
  token 对比：语义层 2,456 vs 全量多模态 518,673（约 200×），
  以及精度审计（ASR 档位/通道盲区/损失来源）
- [EVALUATION.md](EVALUATION.md) — 11 查询 × 3 ASR × 3 通道的
  Precision/Recall 基准：large ASR + 图文融合 P0.80 / R0.83
- [FPS_SAMPLING.md](FPS_SAMPLING.md) — 抽帧率实验：1fps 漏 68% 的
  小对象覆盖，2fps 是性价比甜点，4fps 收益递减

## 已知限制

1. 检测层漏检 = 永久失明（索引里没有就查不到），靠场景层文本兜底
   + 增量补检缓解
2. 同类别多对象靠 IoU 跟踪，快速运动/遮挡会断轨（下一步 ByteTrack
   + 外观 Re-ID）
3. 主角连续出现是长窗口，需动作事件聚类细化
4. 2B 本地 VLM 输出不稳定，生产环境建议命中片段接强模型

## 与 AVIS 协议的关系

本目录是 AVIS 协议的参考实现：`avis.json` 里声明 `signals.objects`
信号（tracks/appearances/tiers/timeline/embeddings），与
transcript/scenes/timeline 并列，符合"语义层模型无关、一次编码
多方复用"的定位。
