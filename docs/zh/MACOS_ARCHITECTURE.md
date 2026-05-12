# Unfoldly macOS 架构与数据目录

[English Version](../MACOS_ARCHITECTURE.md)

---

本目录包含将 Unfoldly 通过 **Tauri** 和 **PyO3** 打包为独立 macOS 应用所需的资源与脚本。当前 Tauri 方案会把 Python 后端直接嵌入 Rust 进程中，正常打包版本不需要单独占用本地端口。

构建说明请阅读 [BUILD_MACOS.md](BUILD_MACOS.md)。

## 1. 数据持久化

macOS 应用的资源位于 `.app/Contents/Resources` 内，签名后通常应视为只读目录。因此，Unfoldly 需要把向量索引、用户偏好、下载模型和日志写到 `.app` 外部的可写目录。

### 默认行为

如果启动时没有设置数据目录：

- **macOS 应用**：会使用应用所在目录作为数据根目录。
- **开发环境**：通常使用仓库根目录相关路径。

正式发布时建议使用专用的应用数据目录，而不是依赖应用所在目录。

### 通过 `FILEAGENT_DATA_DIR` 指定数据目录

可以在启动前设置 `FILEAGENT_DATA_DIR`，让所有运行期数据写入固定目录：

```bash
export FILEAGENT_DATA_DIR="$HOME/Library/Application Support/Unfoldly"
open /Applications/Unfoldly.app
```

正式打包版本可以在 Tauri 启动阶段注入该变量，以保证数据路径稳定。

## 2. 数据目录结构

在 `FILEAGENT_DATA_DIR` 下，应用会创建或使用以下目录：

```text
Unfoldly/
├── chroma_db/               # ChromaDB 向量索引
├── local_models/            # 下载的 GGUF 对话模型
├── models/                  # 下载的 embedding 和 reranker 模型
├── user_preferences.json    # 用户偏好与系统状态
└── logs/                    # 应用日志，例如 backend.log 和 crash.log
```

`bge-small-zh`、`bge-reranker` 等检索辅助模型会由后端自动下载到 `models/` 目录。

## 3. 启动崩溃排查

如果新打包的应用启动后立即退出：

1. 查看 `~/Library/Logs/Unfoldly/` 下的日志。
2. 如果没有 `startup.log`，通常是内部 Python framework 的动态链接路径没有正确配置，或未签名应用被 Gatekeeper 阻止。
3. 如果 `crash.log` 中包含 `UnicodeEncodeError`，通常是 Finder 启动时缺少 UTF-8 locale。请确认打包启动逻辑设置了 `PYTHONUTF8=1` 和 `LC_ALL=en_US.UTF-8`。
