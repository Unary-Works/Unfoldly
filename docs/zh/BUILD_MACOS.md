# Unfoldly macOS 构建指南

[English Version](../BUILD_MACOS.md)

---

本目录包含将 Unfoldly macOS 应用打包为独立 `.app` 和 `.dmg` 的脚本与资源。打包流程会把干净的 Python runtime 和后端依赖打入 Tauri 应用；用户模型、向量数据库和偏好设置属于运行时数据，应放在 `FILEAGENT_DATA_DIR` 下。

## 前置条件

构建前请确认 macOS 机器已安装基础工具链。通常不需要手动创建虚拟环境或手动安装 Python 运行依赖，打包脚本会自动处理 `.venv`、依赖安装，以及带 Metal 加速的 `llama-cpp-python` 源码编译。

1. **Python 3.12**：推荐版本。脚本会使用系统 `python3` 在项目根目录创建 `.venv`。
2. **Rust 工具链**：Tauri 构建需要。可通过 [rustup.rs](https://rustup.rs/) 安装。
3. **Node.js / npm**：前端构建需要。可通过 [nodejs.org](https://nodejs.org/) 或 `brew install node` 安装。
4. **Git 和 CMake**：用于在构建过程中从源码编译 `llama-cpp-python`。
5. **磁盘空间**：建议至少预留 10GB，用于 Python 运行时、缓存和模型相关构建产物。

---

## 构建命令矩阵

所有命令都在仓库根目录执行。公开 release 构建优先使用 `scripts/build-release.sh`。`macos_bundle/scripts` 下的脚本保留给打包内部流程和高级增量构建。

| 场景 | 命令 |
|------|------|
| Apple Silicon clean release 构建 | `bash scripts/build-release.sh arm64` |
| Intel clean release 构建 | `bash scripts/build-release.sh x64` |
| 自动识别当前架构的 clean release 构建 | `bash scripts/build-release.sh` |
| 底层 Apple Silicon 完整构建 | `bash macos_bundle/scripts/package_tauri.sh arm64` |
| 底层 Intel 完整构建 | `bash macos_bundle/scripts/package_tauri.sh x64` |
| 强制重建 runtime site | `rm -rf macos_bundle/python_runtime/site macos_bundle/release && UNFOLDLY_FORCE_REBUILD_SITE=1 bash macos_bundle/scripts/package_tauri.sh arm64` |
| 只改源码后的快速增量构建 | `bash macos_bundle/scripts/package_tauri_no_python.sh arm64` |
| 通过主脚本执行增量构建 | `UNFOLDLY_SKIP_PYTHON_PREP=1 bash macos_bundle/scripts/package_tauri.sh arm64` |
| 已有 runtime 后打最小体积带布局 DMG | `UNFOLDLY_SKIP_PYTHON_PREP=1 UNFOLDLY_MAX_DMG_MB=0 bash macos_bundle/scripts/package_tauri.sh --layout --udbz arm64` |
| 带 Finder 拖拽样式且保留默认体积检查 | `bash macos_bundle/scripts/package_tauri_no_python.sh --layout --udbz arm64` |
| 只测试布局且关闭体积检查 | `bash macos_bundle/scripts/package_tauri_no_python.sh --layout --udbz --no-size-guard arm64` |

## 完整构建

首次构建、依赖变更、或者准备发布 DMG 时，使用 clean release 构建。

```bash
bash scripts/build-release.sh arm64
```

clean release 构建会执行这些步骤：

- 创建或激活 `.venv`；
- 准备 `macos_bundle/python_runtime/install`；
- 默认重建 `macos_bundle/python_runtime/site`；
- 编译已固定 commit 的 JamePeng `llama-cpp-python` fork，并启用 Metal 支持；
- 构建最小 LGPL FFmpeg / FFprobe 解码工具，用于包内媒体探测、常见 MP4/MOV/MKV/WebM/AVI/音频解码、抽帧和 WAV 抽取；
- 编译已固定 commit 的 `pywhispercpp` 源码树，启用 Metal 并关闭 CoreML，用于包内 ASR；
- 校验 parser 依赖、pywhispercpp Metal 和 ChromaDB；
- 执行 Tauri release build；
- 把 `.app` 和 `.dmg` 复制到 `macos_bundle/release/`。
- 验证最终 app 签名；
- 验证 DMG checksum；
- 扫描 app 和挂载后的 DMG，确保没有本地路径、隐私标记或测试数据。

底层等价构建命令是：

```bash
bash macos_bundle/scripts/package_tauri.sh arm64
```

默认固定的 fork commit 是：

```text
ef27f333f367fdc53dc1a729ad8bb6c3c9362514
```

只有在审查并测试新 commit 后，才用环境变量覆盖：

```bash
UNFOLDLY_LLAMA_CPP_PYTHON_REF=<commit-sha> bash macos_bundle/scripts/package_tauri.sh arm64
```

包内 ASR runtime 也固定了 `pywhispercpp` commit：

```text
aaf756bd3c2e8ad38f62bbdc9a32a7549fde9c78
```

只有在审查并测试新 commit 后，才用环境变量覆盖：

```bash
UNFOLDLY_PYWHISPERCPP_REF=<commit-sha> bash macos_bundle/scripts/package_tauri.sh arm64
```

开源审查用的 fork 和 submodule commit 记录在
[../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

只强制重建可复用的 site 目录：

```bash
rm -rf macos_bundle/python_runtime/site macos_bundle/release
UNFOLDLY_FORCE_REBUILD_SITE=1 bash macos_bundle/scripts/package_tauri.sh arm64
```

完全重新解压 Python runtime 并重建 site：

```bash
rm -rf macos_bundle/python_runtime macos_bundle/release
bash macos_bundle/scripts/package_tauri.sh arm64
```

## 增量构建

已经完成过一次完整构建，并且只修改前端或 Python 源码时，使用增量路径。它会复用 `macos_bundle/python_runtime/install` 和 `macos_bundle/python_runtime/site`，并在构建前把当前后端代码同步进 runtime site。

```bash
bash macos_bundle/scripts/package_tauri_no_python.sh arm64
```

等价的主脚本写法：

```bash
UNFOLDLY_SKIP_PYTHON_PREP=1 bash macos_bundle/scripts/package_tauri.sh arm64
```

如果增量脚本提示没有可复用 runtime，先跑一次完整构建。

## 高压缩 DMG 构建

发布候选包优先使用带签名和隐私检查的 clean release 构建：

```bash
bash scripts/build-release.sh arm64
```

完成一次完整构建并准备好 `macos_bundle/python_runtime` 后，可以使用下面高级命令打最小体积的带布局本地 DMG，但它不包含 release wrapper 的额外检查：

```bash
UNFOLDLY_SKIP_PYTHON_PREP=1 UNFOLDLY_MAX_DMG_MB=0 bash macos_bundle/scripts/package_tauri.sh --layout --udbz arm64
```

这个路径会：

- 复用已准备好的 Python runtime；
- 把当前后端代码同步进打包 runtime site；
- 复用或重建最小的包内 FFmpeg 解码工具；
- 执行 Tauri release 构建；
- 创建静态 Finder 拖拽安装布局；
- 使用 `UDBZ` 高压缩格式。

`UNFOLDLY_MAX_DMG_MB=0` 表示关闭体积守卫。如果需要 release 体积检查，可以设置正数阈值，例如 `UNFOLDLY_MAX_DMG_MB=165`。

## DMG 样式与体积模式

默认情况下，`package_tauri.sh` 生成普通 DMG，里面包含 `Unfoldly.app` 和 `Applications` 软链接。默认压缩格式是 `UDBZ`。

| 模式 | 命令 | 说明 |
|------|------|------|
| 普通 DMG，默认压缩 | `bash macos_bundle/scripts/package_tauri.sh arm64` | 默认模式，适合 CI 和普通 release 构建。 |
| 尽量小的 DMG | `UNFOLDLY_SKIP_PYTHON_PREP=1 UNFOLDLY_MAX_DMG_MB=0 bash macos_bundle/scripts/package_tauri.sh --layout --udbz arm64` | 复用 Python runtime，使用 UDBZ，并关闭本地体积检查。 |
| 更快但通常更大的 DMG | `bash macos_bundle/scripts/package_tauri_no_python.sh --udzo arm64` | 使用 UDZO zlib 压缩。 |
| 带 Finder 拖拽布局 | `bash macos_bundle/scripts/package_tauri_no_python.sh --layout --udbz arm64` | 使用 `hdiutil` 和 `macos_bundle/assets/dmg_background_final.png`。 |
| 只测试拖拽布局 | `bash macos_bundle/scripts/package_tauri_no_python.sh --layout --udbz --no-size-guard arm64` | 只用于本地视觉检查，临时构建超过体积限制时使用。 |
| create-dmg 布局路径 | `UNFOLDLY_USE_CREATE_DMG=1 bash macos_bundle/scripts/package_tauri.sh arm64` | 可选路径，会调用 `create-dmg`。 |

默认体积检查是 `UNFOLDLY_MAX_DMG_MB=165`。正式 release candidate 不要关闭体积检查；`--no-size-guard` 只用于本地布局测试。

## 输出位置

最终 release 产物会复制到：

```text
macos_bundle/release/Unfoldly.app
macos_bundle/release/Unfoldly.dmg
```

Tauri 自身构建产物还会保留在：

```text
apps/desktop/src-tauri/target/aarch64-apple-darwin/release/bundle/
apps/desktop/src-tauri/target/x86_64-apple-darwin/release/bundle/
```

---

## 开发模式

开发测试时先隔离运行时数据，避免测试 ChromaDB、模型、偏好设置和聊天历史写进仓库或生产数据目录。

```bash
export FILEAGENT_DATA_DIR="$HOME/UnfoldlyData-Test"
export DB_PATH="$FILEAGENT_DATA_DIR/chroma_db"
export FILEAGENT_LOCAL_MODELS_DIR="$FILEAGENT_DATA_DIR/local_models"
export FILEAGENT_PREFERENCES_PATH="$FILEAGENT_DATA_DIR/user_preferences.json"
mkdir -p "$FILEAGENT_DATA_DIR"
./scripts/start-dev.sh
```

打包后做 smoke test 时也使用同一组数据目录：

```bash
export FILEAGENT_DATA_DIR="$HOME/UnfoldlyData-Test"
export DB_PATH="$FILEAGENT_DATA_DIR/chroma_db"
export FILEAGENT_LOCAL_MODELS_DIR="$FILEAGENT_DATA_DIR/local_models"
export FILEAGENT_PREFERENCES_PATH="$FILEAGENT_DATA_DIR/user_preferences.json"
./macos_bundle/release/Unfoldly.app/Contents/MacOS/Unfoldly
```

## 卸载本地应用数据

只查看会被删除的软件、数据库、模型、日志、缓存和偏好设置路径：

```bash
bash scripts/uninstall-mac.sh --show
```

彻底删除 Unfoldly 及其本地运行数据：

```bash
bash scripts/uninstall-mac.sh
```

这个脚本是破坏性操作，必须输入完整的 `yes` 才会删除。它会删除本地索引、已下载模型、偏好设置、聊天历史、日志和缓存；不会删除用户选择过用于索引的原始文件。

---

## 常见问题

### 1. 应用启动后立即崩溃

查看 `~/Library/Logs/Unfoldly/`：

- **缺少 `startup.log`**：可能是 macOS `dyld` 无法定位内部 Python framework，或 Gatekeeper 阻止未签名应用。
- **终端中出现 `dyld: Library not loaded`**：删除损坏的 runtime 缓存后重新完整构建。

```bash
rm -rf macos_bundle/python_runtime
bash macos_bundle/scripts/package_tauri.sh
```

### 2. Gatekeeper 阻止打开应用

确认系统版本不低于 macOS 10.15。如果是本地未公证构建，可移除 quarantine 属性：

```bash
xattr -cr /Applications/Unfoldly.app
```

### 3. 构建时报 `OSError: [Errno 28] No space left on device`

独立打包需要较大的缓存空间。可以把 Python 构建缓存放到更大的磁盘：

```bash
export UNFOLDLY_PYTHON_CACHE="/Volumes/ExternalDrive/unfoldly-build-cache"
bash macos_bundle/scripts/package_tauri.sh arm64
```
