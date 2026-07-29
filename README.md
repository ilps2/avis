# AVIS — AI Video Semantic Layer

> **给每个视频加一份 AI 能直接读懂的"目录"，不用每次都从头看。**

传统视频格式（MP4、HEVC）是为人类眼睛设计的——像素级重建。AVIS 是为 AI 消费设计的——存储**语义信息**而非像素。结果：一次编码，所有 AI 模型复用，无需重复解码。

```
传统管线:  MP4 → 解码 → 像素 → CLIP/ViT → tokens    (每次)
AVIS 管线: 直接读取语义层 → tokens                   (一次编码, 永远复用)
```

## 定位

| | 不是 | 而是 |
|------|------|------|
| ❌ | 新的视频编码格式 | 视频的 AI 语义索引 |
| ❌ | 绑定某个模型的特征缓存 | 模型无关的语义表示层 |
| ❌ | "比 MP4 快 5-17 倍" | "一次编码，所有 AI 复用" |

AVIS 不替代你的模型。它消除**重复计算**——GPT-4V 处理一遍、Gemini 处理一遍、Claude 处理一遍，三遍都在重复解同样的像素、跑同样的编码器。AVIS 做中间层，编码一次，多方消费。

## 语义层

AVIS 不规定"里面存什么特征"，只规定**AI 如何读取语义**。就像 HTML 不规定浏览器怎么渲染，只规定文档结构。

```
video.mp4          # 给人看
video.avis/        # 给 AI 看
├── transcript/    # 时间轴对齐的字幕
├── scenes/        # 镜头切换与分类
├── timeline/      # 融合评分时间线
├── embeddings/    # 帧级语义嵌入（可选，模型可替换）
└── manifest.json  # 元数据与编码器信息
```

以后 CLIP 换 SigLIP、Whisper 换其他 ASR——协议不变，语义层不变。

## 路线图

AVIS 不应该是"写出来再推广的协议"，而应该是**从真实产品中长出来的标准**。历史反复证明：Linux、Git、SQLite、FFmpeg——都是先解决自己的问题，再沉淀成基础设施。

| 阶段 | 产品 | 对 AVIS 的意义 |
|------|------|--------------|
| 1. AI 视频笔记 | Chrome 插件，中文视频摘要 | 积累完整管线：下载→ASR→切片→摘要 |
| 2. 视频搜索 | 搜"铜锅"，跳到 4:00；搜"麻酱"，跳到 3:35 | 建立语义索引层 |
| 3. Video RAG | 开发者 API：`ask(video, question)` | 协议从真实需求中长出来 |
| 4. AVIS SDK | `from avis import Video` | 成为开发者默认依赖 |
| 5. 协议 | 开源规范 + 多编码器生态 | 不绑定模型，格式被社区采纳 |

**当前阶段：第一步，AI 视频笔记。** [live-clip](https://github.com/ilps2/live-clip) 是 AVIS 的第一个实现。

## 为什么不是"又一个视频格式"

大多数"AI 视频格式"的问题是绑定了**今天某个模型的特征**——CLIP 768 维向量、VAE latent——模型换代，格式报废。

AVIS 的赌注是：**语义层比具体特征长寿。** 字幕、场景、时间线、知识图谱——这些东西 10 年后任何 AI 都需要。768 维向量？未必。

## 与 live-clip 的关系

```
live-clip (产品)          AVIS (标准)
─────────────────        ─────────────────
Chrome 插件              语义层规范
下载→ASR→切片→搜索         manifest schema
付费用户验证需求           编码器无关接口
真实数据沉淀              长期演进方向
```

[live-clip](https://github.com/ilps2/live-clip) 赚钱 → 数据验证 AVIS 该存什么 → AVIS 从需求中长出来。

## 本仓库

这是 AVIS 的**旧版二进制格式原型**（`avis/`、`scripts/`）。新架构在当前活跃开发的 [live-clip](https://github.com/ilps2/live-clip) 仓库中（`avis.py` + `avis-online.py`）。

```bash
# live-clip 中的 AVIS CLI
cd live-clip-repo
python3 avis.py encode video.mp4 --clip   # 全信号编码
python3 avis.py search video_avis/ "铜锅"  # 视觉搜索
python3 avis.py curate "https://b23.tv/xxx"  # B站一键下载+编码
```

---

*AVIS 不追求"真正的视频理解"。地图不需要等于领土才有用。*
