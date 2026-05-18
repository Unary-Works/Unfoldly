<p align="center">
  <img src="../assets/brand/unfoldly-logo.png" alt="Unfoldly" width="120">
</p>

# Unfoldly

**为你找回想不起来名字的本地文件。**

Unfoldly 可以按记忆、语义和视觉内容搜索本地文档、照片、视频、表格等文件，不需要上传你的文件。

你可以这样搜索：

- “那张我家狗在海边的照片”
- “我看到 manta ray 的那个视频”
- “解释我旅行预算为什么超支的表格”

Unfoldly 会找到相关本地文件，展示来源，并允许你基于该文件继续追问。

<p align="center">
  <a href="https://www.unfoldly.io/">官网</a> |
  <a href="https://www.unfoldly.io/">下载</a> |
  <a href="#演示">演示</a> |
  <a href="#隐私">隐私</a> |
  <a href="#从源码构建">构建</a> |
  <a href="#卸载">卸载</a> |
  <a href="../../README.md">English</a>
</p>

> **早期 Beta：** 当前公开版本主要面向 macOS，并提供已签名和公证的 macOS 构建。

## 为什么需要 Unfoldly

传统文件搜索要求你记得文件名、文件夹或精确关键词。

但很多时候，你记住的是别的东西：

- 照片里有什么
- 视频里发生了什么
- 文档大概讲什么
- 表格解释了什么
- 你在哪里见过某个想法、图表、引文、数字或截图

Unfoldly 解决的是“你记得内容，但想不起文件在哪”的问题。

你可以把它理解成面向个人文件的 AI-native Everything：像桌面搜索一样本地优先，但可以按语义、上下文和视觉记忆来搜索。

## 产品概览

Unfoldly 会把你选择的文件和文件夹构建成本机上的私有可搜索记忆层。

基本流程：

1. **选择来源**
   添加你希望 Unfoldly 索引的文件或文件夹。

2. **构建本地记忆**
   Unfoldly 会在本地处理文件，根据文件类型提取文本、视觉信号、元数据、转录内容和可搜索向量。

3. **搜索和追问**
   用自然语言找到相关文件，查看来源，缩小搜索范围，或在上下文中继续追问。

整个体验都围绕本地运行设计。你的文件、索引、下载模型、偏好设置、日志和聊天历史都保留在你的机器上。

## 演示

### 找到你记得的照片

按照片里的内容搜索本地图片，而不是按文件名搜索。

[![找到你记得的照片](../assets/demos/find-my-dogs.gif)](../assets/videos/find-my-dogs.mp4)

[观看完整视频](../assets/videos/find-my-dogs.mp4)

### 找到某个视频片段

描述你记得的画面或时刻，Unfoldly 会找到对应的本地视频和相关片段。

[![找到某个视频片段](../assets/demos/manta.gif)](../assets/videos/manta.mp4)

[观看完整视频](../assets/videos/manta.mp4)

### 找到藏在文件里的答案

先找到正确的本地文件，再继续追问。Unfoldly 会保留来源上下文，方便你检查答案来自哪里。

[![找到藏在文件里的答案](../assets/demos/excel-analysis.gif)](../assets/videos/excel-analysis.mp4)

[观看完整视频](../assets/videos/excel-analysis.mp4)

## 支持搜索的文件

Unfoldly 面向常见个人文件设计。

| 类型 | 格式 |
| --- | --- |
| 文档 | PDF, DOC, DOCX, TXT, Markdown, RTF, EPUB, MOBI |
| 表格 | XLSX, XLS, CSV, TSV, Numbers, ODS |
| 幻灯片 | PPTX, PPT, Keynote, ODP |
| 图片 | JPG, PNG, HEIC, HEIF, WEBP, GIF, TIFF, BMP, SVG |
| 音频 | MP3, WAV, M4A, FLAC, AAC, OGG, WMA, AIFF |
| 视频 | MP4, MOV, MKV, AVI, WEBM, M4V, WMV |
| 其他结构化文件 | JSONL, XML, SQL, YAML |

实际支持质量会受文件结构、文件大小、解析器可用性和所选模型影响。

## 核心能力

- **按记忆搜索**：按你记得的内容找文件，而不只依赖文件名或精确关键词。
- **搜索视觉内容**：让照片、截图、扫描文件和视频片段可被搜索。
- **基于来源追问**：先召回正确文件，再围绕该文件继续提问。
- **自主选择索引范围**：Unfoldly 只索引你主动选择的文件和文件夹。
- **本地运行**：核心搜索和问答流程都围绕本机运行设计。

## 隐私

Unfoldly 的核心原则很简单：

**你的本地文件应该留在本地。**

Unfoldly 会把搜索和聊天流程保留在你的设备上。

| 数据 | 存放位置 |
| --- | --- |
| 原始文件 | 你选择的本地文件夹 |
| 搜索索引 | 本地应用数据目录 |
| 下载模型 | 本地应用数据目录 |
| 偏好设置 | 本地应用数据目录 |
| 聊天历史 | 本地应用数据目录 |
| 日志 | 本地应用数据目录 |

Unfoldly 可能会联网下载模型文件或应用更新，但不会上传你的个人文件内容。

## 安装

[立即下载](https://www.unfoldly.io/)

当前公开版本主要面向 macOS，并提供已签名和公证的 macOS 应用。

## 快速开始

1. 下载并打开 Unfoldly。
2. 添加你想搜索的文件或文件夹。
3. 下载或选择支持的本地模型。
4. 等待索引完成。
5. 按记忆、语义或视觉内容搜索。
6. 打开来源文件，或继续追问。

## 卸载

Unfoldly 提供 macOS 卸载辅助脚本，方便用户查看或删除本地应用数据。

查看将被删除的应用、数据库、模型、偏好设置、日志、缓存和崩溃报告路径：

```bash
bash scripts/uninstall-mac.sh --show
```

彻底删除 Unfoldly 和本地运行数据：

```bash
bash scripts/uninstall-mac.sh
```

卸载脚本是破坏性操作，真正删除前必须输入 `yes` 确认。它会删除本地索引、下载模型、偏好设置、已选来源记录、聊天历史、日志、缓存、崩溃报告和已安装的应用包。它不会删除你选择用于索引的原始文件。

## 技术架构

Unfoldly 是一个本地桌面应用。

| 层级 | 技术 |
| --- | --- |
| 桌面壳 | Tauri |
| 前端 | React, TypeScript, Vite |
| 原生桥接 | Rust + PyO3 |
| 后端 | Python |
| 向量库 | ChromaDB |
| 检索 | Embeddings, lexical search, reranking, source-scoped filtering |
| 模型运行时 | 通过 llama.cpp / llama-cpp-python 运行本地 GGUF 模型 |
| 存储 | 本地应用数据目录 |

打包后的桌面应用会把 Python 后端嵌入 Tauri 应用中，正常使用打包版本时不需要单独启动本地 HTTP 服务。

## 从源码构建

开发和 release 打包请参考完整的 [macOS 构建指南](BUILD_MACOS.md)。
发布 GitHub Release 并让 app 内检查更新生效，请参考 [Release 发布说明](RELEASE.md)。

最小本地工具链：

- macOS
- Python 3.12
- Rust
- Node.js 和 npm
- Git 和 CMake

启动本地开发应用：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./scripts/start-dev.sh
```

创建干净的 macOS release 构建：

```bash
bash scripts/build-release.sh arm64
```

release 产物输出到：

```text
macos_bundle/release/Unfoldly.app
macos_bundle/release/Unfoldly.dmg
```

索引、下载模型、偏好设置、日志和聊天历史等运行时数据会存放在本地应用数据目录。

## 模型和检索

Unfoldly 设计为通过本地运行时运行受支持的本地 AI 模型。

当前模型注册表包含来自以下家族的 GGUF 文本模型和视觉语言模型：

- Gemma
- Qwen / Qwen-VL
- GLM
- Llama
- Ministral
- DeepSeek-R1 Distill
- gpt-oss

检索系统当前使用 **ChromaDB** 作为本地向量库，并使用 BGE 系列 embedding 和 reranking 模型。

## 贡献

欢迎贡献。提交 pull request 前请先阅读 [CONTRIBUTING.md](../../CONTRIBUTING.md)。

所有 pull request 都会由 Unary Works 维护者审核后再合并。

## 许可

Unfoldly 使用 [Apache License 2.0](../../LICENSE) 授权。

Copyright (c) 2026 Unary Works LLC.

打包的第三方依赖保留各自的许可。依赖许可和 notice 详情请见 [Third-Party License Notes](../THIRD_PARTY_DEPENDENCIES.md)、[Third-Party Notices](../THIRD_PARTY_NOTICES.md) 和 [NOTICE](../../NOTICE)。
