# 定制学习计划：视频理解（方向）× AI 剪辑（应用）

为"非科班、英文客服组长、已有 AVIS 语义层项目"的现状定制。
原则：**论文是地图，你的项目是领土**——每篇论文读完必须回答
"这能怎么改我的 AVIS / live-clip？"

## 0. 你的现状画像（来自本次合作的实证）

**已经动手做过的（比大多数科班生简历真实）：**

- 视频底层：I/P/B 帧、GOP、运动矢量、ffmpeg、1/2/4fps 抽帧实验
- 视觉模型：GroundingDINO（开放词汇检测）、CLIP（对象向量）、
  BLIP（场景描述）、Whisper（tiny/base/large-v3 三档对比）、
  Qwen2-VL（本地多模态理解）
- 系统：对象跟踪 + 主配角分层 + 统一时间轴 + MCP server + 自动分诊
- 评估：token 对比（2456 vs 518,673）、P/R、边界误差、11 查询基准、
  48 片段人工标注包

**理论缺口（本计划要补的）：**

1. Transformer 内部机制（注意力、位置编码）——你用过，但没系统理解
2. 视频时序建模家族（3D 卷积、时空注意力、masked modeling）
3. 时序定位/检索任务的形式化（moment retrieval、TAL）
4. 视频-语言对齐理论（CLIP 为什么有效、VLM 怎么"看"视频）
5. Agent 视频理解（记忆、工具调用、查询导向采样）
6. 评测基准设计（任务定义、指标陷阱、人工标注流程）

---

## 1. 六大知识模块

### M1：视频与视觉基础理论（第 1-2 周）

**核心问题：** 视觉特征是什么？CLIP 为什么能让图和文对齐？
视频和图片的建模差别在哪？

| 论文 | 一句话人话 | 读它的收获 | 连接你的项目 |
|---|---|---|---|
| [CLIP](https://arxiv.org/abs/2103.00020) | 让"猫的照片"和"一只猫"在向量空间里靠近 | embedding 的本质、图文对齐训练 | 你的对象向量和参考图查询就是 CLIP 的应用 |
| [BLIP-2](https://arxiv.org/abs/2301.12597) | 用 Q-Former 桥接冻结的视觉编码器和 LLM | 视觉-语言桥接架构 | 你用的 BLIP 是它的小弟，理解桥接=理解 VLM 骨架 |
| [VideoMAE](https://arxiv.org/abs/2203.12602) | 把视频帧随机盖住，让模型猜 | masked modeling、视频时空管建模 | 理解"模型看视频到底看什么" |
| [Whisper](https://arxiv.org/abs/2212.04356) | 海量弱监督语音转写 | 为什么大模型 ASR 质量陡增 | 你的 large-v3 默认配置的由来 |

动手任务：把 AVIS 的对象 embedding 换成 SigLIP
（[2303.15343](https://arxiv.org/abs/2303.15343)）跑一遍检索，
对比 CLIP 的 P/R——第一次亲手体会"换编码器 = 换索引质量"。

### M2：检测与跟踪（第 3-4 周）

**核心问题：** 怎么在画面里找到对象并保持身份？开放词汇检测
为什么比固定类别强？

| 论文 | 一句话人话 | 收获 | 连接你的项目 |
|---|---|---|---|
| [DETR](https://arxiv.org/abs/2005.12872) | 用集合预测代替锚框，Transformer 直接输出物体 | 检测的现代范式 | GroundingDINO 的前身 |
| [Grounding DINO](https://arxiv.org/abs/2303.05499) | 用文字提示做开放词汇检测 | 你的检测器本身 | 你已经部署了它，现在读它的论文补原理 |
| [SAM](https://arxiv.org/abs/2304.02643) | 点一下/框一下就分割任意物体 | 分割≠检测，像素级理解 | 对象层升级到"对象分割"的路线图 |
| [ByteTrack](https://arxiv.org/abs/2110.06864) | 用低分检测框做跟踪，简单且强 | 跟踪=匹配问题，比你现在的 IoU 强在哪 | 你 README 里的"下一步换 ByteTrack" |
| [DEVA](https://arxiv.org/abs/2309.03903) | 图像分割 + 双向传播=跟踪一切 | 开放词汇视频分割 | 龙套对象不漏检的终极方案 |

动手任务：在 AVIS 上把 IoU 跟踪换成 ByteTrack，跑 eval 看
P/R 变化——这就是你简历上"我把跟踪精度提升了 X%"的来源。

### M3：视频时序建模与理解（第 5-6 周）

**核心问题：** 时间维度怎么建模？VLM 看视频和看图片差在哪？

| 论文 | 一句话人话 | 收获 | 连接你的项目 |
|---|---|---|---|
| [TimeSformer](https://arxiv.org/abs/2102.05095) | 把注意力拆成"空间"和"时间"两步 | 时空注意力的基本盘 | 理解你的关键帧序列为什么能近似视频 |
| [SlowFast](https://arxiv.org/abs/1812.03982) | 一路慢看静态、一路快看动作，双路融合 | 多速率采样的鼻祖 | 直接支撑你的 1/2/4fps 实验 |
| [Video-LLaVA](https://arxiv.org/abs/2311.10122) | 图片和视频共享同一个编码器 | 视频-语言模型的最小骨架 | 你的 understand_video 就是它的精简版 |
| [LLaVA-Video](https://arxiv.org/abs/2406.11813) | 系统研究了视频帧采样对性能的影响 | **帧采样率和理解质量的关系** | 你的 FPS_SAMPLING 实验的学界对应物 |
| [Qwen2-VL](https://arxiv.org/abs/2409.12191) | 当前最强的开源 VLM 之一 | 动态分辨率、视频 token 化 | 你的本地推理模型，现在读懂它为什么贵 |

动手任务：用 LLaVA-Video 论文的结论重做你的抽帧实验——
它证明 1fps 采样下哪些任务饱和、哪些任务崩，对照你的实测。

### M4：低成本与高效（第 7-8 周）★ 你的主战场

**核心问题：** 怎么花最少的 token/算力拿到最多的理解？
这是 AVIS 的立身之本，也是前沿最活跃的方向。

| 论文 | 一句话人话 | 收获 | 连接你的项目 |
|---|---|---|---|
| [CoViAR](https://arxiv.org/abs/1712.09692) | 直接用压缩域的运动矢量分类动作，不解码像素 | 你的第一课"白嫖运动矢量"的论文原型 | AVIS 的 MV 信号的理论源头 |
| [SlowFocus](https://arxiv.org/abs/2602.03589) | 稀疏采样全局 + 密集采样"焦点段" | **两段式采样的学术版** | 你的"全片 1fps + 命中片段 4fps" |
| [GroundVTS](https://arxiv.org/abs/2604.02093) | 按查询相关性筛视觉 token，只喂有用的进 LLM | **查询导向 token 采样的最新前沿** | 你的 B-hit 模式的进化方向 |
| [QuoTA](https://arxiv.org/abs/2503.08689) | 先用推理判断哪些帧重要，再分配 token | 训练无关的 token 预算分配 | 你的图文融合检索 + 重排 |
| [MoCrop](https://arxiv.org/abs/2509.18473) | 用 H.264 运动矢量定位运动密集区，自动裁剪 | **压缩域裁剪，剪辑应用直击** | 你的 AI 剪辑 + 运动矢量两个兴趣点的交点 |
| [QLoRA](https://arxiv.org/abs/2305.14314) | 4-bit 量化微调大模型 | 量化为什么能省钱 | 你的"模型瘦身"认知的正式版 |

动手任务：用 AVIS 的 timeline 数据复现一次"查询导向采样"：
对 10 个查询，比较均匀采样 vs 你的融合检索采样的 token 和 P/R。
这篇实验笔记就是你的招牌博客。

### M5：时序定位与检索（第 9-10 周）

**核心问题：** "找出口红出现在哪几秒"在学术上叫什么？怎么建模？

| 论文 | 一句话人话 | 收获 | 连接你的项目 |
|---|---|---|---|
| [QVHighlights / Moment-DETR](https://arxiv.org/abs/2107.09609) | 给定文字查询，端到端输出起止时间和高光分数 | moment retrieval + highlight detection 的标准范式 | 你的 plan_clips 的学术亲戚 |
| [ActionFormer](https://arxiv.org/abs/2202.07925) | 单阶段定位动作片段，无锚框 | TAL 任务形式化 | 你的"事件窗口"的严谨版本 |
| [VideoTree](https://arxiv.org/abs/2405.19209) | 按查询动态建树，只展开相关分支 | 查询自适应的层次表示 | 你的场景→对象→片段的层级索引 |
| [VideoAgent](https://arxiv.org/abs/2403.11481) | 给 LLM 配记忆和工具，让它主动看视频 | **agent 视频理解的范式** | 你的 MCP server 的论文对应物 |

动手任务：部署 Moment-DETR 到你的一段直播切片上，和你
plan_clips 的窗口对比 IoU——第一次用学界基线打自己的方案。

### M6：应用：摘要、剪辑与评测（第 11-12 周）

**核心问题：** 理解怎么变成产品？评测怎么做才可信？

| 论文 | 一句话人话 | 收获 | 连接你的项目 |
|---|---|---|---|
| [EDSNet](https://arxiv.org/abs/2409.14724) | 高效视频摘要网络 | 摘要任务和数据集（SumMe/TVSum） | 你的"AI 视频笔记"路线图 |
| [InstructVideo](https://arxiv.org/abs/2312.12490) | 用人类反馈微调视频生成模型 | 剪辑的"手感"能否学习 | 了解生成式剪辑的边界（不是你的赛道） |
| [Video-MME](https://arxiv.org/abs/2405.21075) | 多模态视频理解大基准 | 评测任务设计的严谨性 | 你的 eval 框架的行业参照 |
| [LongVideoBench](https://arxiv.org/abs/2407.15754) | 长视频 + 字幕交织理解基准 | 长视频评测的陷阱 | 你的"小时级直播回放"未来场景 |

动手任务：完成 48 片段人工标注，出人工真值版评估报告；用
Video-MME 的"任务分维度"思想把你的 eval 拆成
（检索/定位/理解/剪辑可用性）四张子表。

---

## 2. 12 周执行表

| 周 | 模块 | 论文量 | 动手任务 | 输出 |
|---|---|---:|---|---|
| 1 | M1 | 2 | 换 SigLIP 跑检索对比 | 实验笔记 |
| 2 | M1 | 2 | 写《CLIP 对象向量：为什么参考图查询比文字稳》 | **博客 1** |
| 3 | M2 | 2-3 | 读 Grounding DINO 论文对照你的部署 | 笔记 |
| 4 | M2 | 2 | 换 ByteTrack 跑 eval | P/R 对比表 |
| 5 | M3 | 2 | 读 LLaVA-Video 的采样实验 | 笔记 |
| 6 | M3 | 3 | 用论文结论重做 fps 实验 | **FPS 报告 v2** |
| 7 | M4 | 3 | 复现查询导向采样 | 实验笔记 |
| 8 | M4 | 3 | 写《200 倍 token 节省的学术坐标》 | **博客 2** |
| 9 | M5 | 2 | 部署 Moment-DETR 对比 plan_clips | IoU 对比表 |
| 10 | M5 | 2 | 读 VideoAgent 对照你的 MCP | 架构对照图 |
| 11 | M6 | 2 | 完成 48 片段人工标注 | **人工真值报告** |
| 12 | M6 | 2 | 整理作品集 + 简历叙事 | 求职材料 v1 |

每周节奏：周中 2 篇论文（每次 60-90 分钟），周末 1 个动手实验
（2-3 小时），每月 1 篇对外输出。

## 3. 非科班论文阅读法

1. **五步法（60 分钟/篇）**：
   标题摘要（5min）→ 图表（15min）→ 方法（25min）→
   相关工作扫读（10min）→ 用自己的话重述（5min，写下来）
2. **一页笔记模板**：问题 / 方法一句话 / 关键图 / 和 AVIS 的关系 /
   如果我来实现会怎么改
3. **用 AI 当陪读**：把论文 PDF 或 ar5iv 链接丢给我，
   我按你的水平讲人话、画架构、出三道自测题
4. **优先级**：P0 = M4 全部 + M5 的 VideoTree/VideoAgent +
   M3 的 LLaVA-Video（直接关系你的方向和就业）；其余 P1

## 4. 完成后的作品集（面试可直接用）

1. AVIS 语义层 + 人工真值评估报告（数字）
2. 帧采样率实验报告 v2（对照 LLaVA-Video 论文）
3. 查询导向采样复现笔记（对照 GroundVTS/QuoTA）
4. Moment-DETR vs plan_clips 对比表
5. 博客 2 篇 + MCP server 开源项目

这套东西对应的面试问题：Embedding 是什么、CLIP 怎么对齐、
视频和图片建模差在哪、抽帧率和成本怎么权衡、怎么设计评测、
agent 怎么理解视频——全部能用"论文观点 + 你的实测"双轨回答。

## 5. 重要提醒

- 不要追着公式跑：你的目标不是推导证明，而是**能用论文的语言
  解释你做过的事**，并知道前沿在往哪走
- 每篇论文必须产出一个"改 AVIS"的动作，否则这周白读
- 论文读了 20 篇不如 1 个部署：Moment-DETR、ByteTrack 各部署一次，
  你对"前沿"的理解会超过 90% 只看论文的人

---

## 6. 配套课程（按模块对应）

原则：课程只补"理论底盘"，不替代动手。每个模块最多选 1 门主线课 + 1 个
动手教程；你已经部署过的部分（GroundingDINO、Whisper、抽帧）直接跳视频、
只看概念，把时间留给 M4/M5。

### 主干课（先选一条，推荐 A）

| 路线 | 课程 | 适合你的理由 | 时间 |
|---|---|---|---|
| A（推荐） | 李沐《动手学深度学习》[zh.d2l.ai](https://zh.d2l.ai/)（B站搜"动手学深度学习"有配套视频） | 免费、中文、PyTorch，覆盖 CNN、注意力、Transformer、目标检测，与你的工具链同栈 | 每周 3-4 小时，按需选章 |
| B | Stanford CS231n 2025 [B站中英字幕全18讲](https://www.bilibili.com/video/BV19zWVziEsP/) / [官网](https://cs231n.stanford.edu/) | 视觉经典主线：分类→检测→Transformer→视频，李飞飞主讲 | 挑 8-10 讲，每讲 1-2 小时 |
| C | [3Blue1Brown 神经网络系列](https://www.youtube.com/playlist?list=PLZHQObOWTQDNu6R1_67000Dx_ZCJB-3pi) | 数学直觉动画化，非科班最友好 | 4-6 小时 |

### M1 视频与视觉基础

| 课程 | 平台 | 学什么 | 对应论文/项目 |
|---|---|---|---|
| 动手学深度学习（卷积与注意力章节） | d2l | CNN、注意力机制、Transformer | 补 CLIP/BLIP 的底层 |
| Building Multimodal Search and RAG | [DeepLearning.AI](https://learn.deeplearning.ai/courses/building-multimodal-search-and-rag)（免费） | 对比学习、CLIP 为什么能让图文对齐、多模态向量检索 | CLIP、你的语义层检索 |
| Audio Course 第 5 章（Whisper） | [Hugging Face](https://huggingface.co/learn/audio-course)（免费） | ASR 原理、Whisper 架构与微调 | Whisper、large-v3 配置 |
| Image and Video Compression | [Stanford EE398A](https://web.stanford.edu/class/ee398a/)（公开资料） | I/P/B 帧、GOP、运动补偿、H.264 | 你的视频底层知识系统化 |
| Multimodal Machine Learning | [CMU 11-777](https://cmu-multicomp-lab.github.io/mmml-course/fall2022/)（公开课，可选） | 多模态表示、对齐、融合的理论框架 | 图文融合检索的学术版 |

### M2 检测与跟踪

| 课程 | 平台 | 学什么 | 对应论文/项目 |
|---|---|---|---|
| CS231n 2025 目标检测/分割章节 | Stanford/B站 | 检测与分割的现代范式 | DETR、GroundingDINO 前置 |
| Computer Vision Course | [Hugging Face](https://huggingface.co/learn/computer-vision-course)（免费） | 图像分类、检测、分割的 transformers 实现 | GroundingDINO 部署的理论补充 |
| Roboflow Notebooks + 实战视频 | [GitHub](https://github.com/roboflow/notebooks)（免费） | YOLOv8 + ByteTrack + Supervision 跟踪实战 | 你的"换 ByteTrack"动手任务 |
| 【精读AI论文】YOLO 系列 | B站 [同济子豪兄](https://www.bilibili.com/video/BV15w411Z7LG) | 检测器原理的中文精讲 | 读 GroundingDINO 前的热身 |
| OpenMMLab AI实战营 | B站 [目标检测与MMDetection](https://www.bilibili.com/video/BV1Ak4y1p7W9/) | MMDetection/MMYOLO 中文实战 | 检测/跟踪生态全览 |

### M3 视频时序建模

| 课程 | 平台 | 学什么 | 对应论文/项目 |
|---|---|---|---|
| EECS 498 Lecture "Videos" | [密歇根大学公开课](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/schedule.html)（课件公开，录像在 Michigan Online/YouTube） | 视频分类、早/晚融合、3D CNN、双流网络 | TimeSformer/SlowFast 的地基 |
| UvA Deep Learning Course | [uvadlc.github.io](https://uvadlc.github.io/)（公开课） | Transformer 原理 + 视频处理专题 | Qwen2-VL、Video-LLaVA 原理 |
| Building Multimodal Data Pipelines | [DeepLearning.AI](https://learn.deeplearning.ai/courses/building-multimodal-data-pipelines)（免费） | VLM 处理视频帧、生成带时间戳的场景描述 | 和你的 AVIS 管线几乎同构 |

### M4 低成本与高效 ★

| 课程 | 平台 | 学什么 | 对应论文/项目 |
|---|---|---|---|
| TinyML and Efficient Deep Learning Computing | [MIT 6.5940](https://hanlab.mit.edu/courses/2024-fall-65940)（YouTube 公开录像；[中文笔记](https://github.com/erectbranch/MIT-Efficient-AI)） | 剪枝、量化、蒸馏、LLM 压缩 | QLoRA、token/算力成本理论 |
| LLM Course 量化章节 | [Hugging Face](https://huggingface.co/learn/llm-course)（免费） | bitsandbytes、QLoRA 实操 | 你的模型瘦身方案 |
| EE398A 压缩域部分 | Stanford（见 M1） | 运动矢量、帧间编码 | CoViAR、MoCrop 的底层知识 |

### M5 时序定位与检索

| 课程 | 平台 | 学什么 | 对应论文/项目 |
|---|---|---|---|
| Building Multimodal Search and RAG | DeepLearning.AI（见 M1） | query→向量→top-k 检索全流程 | Moment-DETR 之前的检索形式化 |
| Retrieval Optimization | [DeepLearning.AI + Qdrant](https://community.deeplearning.ai/t/new-course-enroll-in-retrieval-optimization-from-tokenization-to-vector-quantization/702150)（免费） | tokenization、向量量化、检索优化 | 语义层索引效率进阶 |
| AI Agents / Agents Course | [DeepLearning.AI](https://learn.deeplearning.ai/courses/ai-agents) / [Hugging Face](https://huggingface.co/learn/agents-course)（免费） | 工具调用、记忆、多步推理 | VideoAgent、你的 MCP server |

### M6 应用与评测

| 课程 | 平台 | 学什么 | 对应论文/项目 |
|---|---|---|---|
| Machine Learning Crash Course 分类指标 | [Google](https://developers.google.com/machine-learning/crash-course)（免费） | 精确率/召回率、混淆矩阵、AUC | 你的 P/R 评估的地基 |
| Evaluating and Debugging Generative AI | [DeepLearning.AI](https://learn.deeplearning.ai/courses/evaluating-debugging-generative-ai)（免费） | 实验跟踪、评测与调试流程 | eval_clips 的方法论 |
| Practical Deep Learning for Coders（可选） | [fast.ai](https://course.fast.ai/)（免费） | 从训练到部署的完整项目闭环 | 作品集项目化 |

### 没有专门课程的模块怎么学

- Moment-DETR、GroundVTS、QuoTA 等新论文没有课程：读论文 + 官方 repo README，
  必要时让我按你的水平讲人话、画架构。
- ByteTrack 有完整实战教程：搜 "Roboflow ByteTrack" 或直接跑
  [roboflow/notebooks](https://github.com/roboflow/notebooks) 里的跟踪 notebook。
- 视频摘要（EDSNet）和生成式剪辑（InstructVideo）也没有课程：用
  "论文 + AI 陪读"方式，重点落在任务定义和评测基准上。

### 时间预算

每周课程 4-6 小时，不挤占论文（2 篇/周）和周末实验（2-3 小时）。
主干课 A 的章节按模块穿插学；短课（DeepLearning.AI / Hugging Face）每门 1-2 天可完成。
学完每门课必须回答一个问题：这个知识点怎么改我的 AVIS？
