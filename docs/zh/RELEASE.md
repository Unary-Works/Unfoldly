# Release 发布说明

[English version](../RELEASE.md)

这份文档说明如何构建 macOS release 包、上传到 GitHub Releases，并让
Tauri app 内更新器发现最新版本。

## 1. 更新应用版本号

发布前需要统一更新版本号：

```bash
cd apps/desktop
npm run version:sync -- 1.0.0
npm install --package-lock-only
cd ../..
```

版本号使用 SemVer，例如 `1.0.0`；Git tag 使用 `v1.0.0`。不要在
`package.json`、`Cargo.toml` 或 `tauri.conf.json` 里加入 `v` 前缀。

## 2. 构建 release 包

在仓库根目录执行：

```bash
export TAURI_SIGNING_PRIVATE_KEY="$(cat path/to/updater-private.key)"
export TAURI_SIGNING_PRIVATE_KEY_PASSWORD="$(cat path/to/updater-private.password)"
bash scripts/build-release.sh arm64
```

这会走当前项目推荐的 clean release wrapper，并使用最高压缩 DMG 路径：

```text
macos_bundle/release/Unfoldly.app
macos_bundle/release/Unfoldly.app.tar.gz
macos_bundle/release/Unfoldly.app.tar.gz.sig
macos_bundle/release/Unfoldly.dmg
```

脚本会验证 app 签名、验证 DMG checksum，并扫描 app 和挂载后的 DMG，避免本地私有路径和测试数据进入发布包。

如果要连 Python runtime 和依赖也完全重新构建：

```bash
rm -rf macos_bundle/python_runtime macos_bundle/release
bash scripts/build-release.sh arm64
```

## 3. 生成 updater manifest

在 `apps/desktop` 目录执行：

```bash
npm run updater:manifest -- --repo Unary-Works/Unfoldly --tag v1.0.0 --notes "Release v1.0.0"
```

这会写入：

```text
macos_bundle/release/latest.json
```

## 4. 创建 GitHub Release

推送 release commit 和 tag：

```bash
git tag v1.0.0
git push origin v1.0.0
```

然后在 GitHub 上基于该 tag 创建 release。

app 内更新必需上传的 assets：

```text
macos_bundle/release/Unfoldly.app.tar.gz
macos_bundle/release/Unfoldly.app.tar.gz.sig
macos_bundle/release/latest.json
```

可选的手动安装 asset：

```text
macos_bundle/release/Unfoldly.dmg
```

如果使用 GitHub 网页：

1. 打开 GitHub 仓库。
2. 进入 **Releases**。
3. 点击 **Draft a new release**。
4. 选择或创建 tag `v1.0.0`。
5. 上传 `Unfoldly.app.tar.gz`、`Unfoldly.app.tar.gz.sig` 和 `latest.json`。
6. 可选上传 `Unfoldly.dmg` 供手动安装。
7. 发布 release。

如果使用 GitHub CLI：

```bash
gh release create v1.0.0 \
  macos_bundle/release/Unfoldly.app.tar.gz \
  macos_bundle/release/Unfoldly.app.tar.gz.sig \
  macos_bundle/release/latest.json \
  macos_bundle/release/Unfoldly.dmg \
  --repo Unary-Works/Unfoldly \
  --title "Unfoldly v1.0.0" \
  --notes "Release notes here"
```

## 5. App 如何检查更新

设置页的 Updates / 检查更新会请求：

```text
https://github.com/Unary-Works/Unfoldly/releases/latest/download/latest.json
```

它会比较：

- 当前打包 Tauri app 的版本；
- `latest.json.version`，例如 `1.0.0`。

如果 `latest.json.version` 更新，app 会下载签名后的
`Unfoldly.app.tar.gz`，用已提交的 updater public key 校验签名，安装更新，
并提示用户重启。

注意：

- GitHub 仓库和 release asset 必须对最终用户可公开访问。
- Draft release 对 app 不可见。
- `latest.json` 必须指向同一 tag 下的 release asset URL。
- 不要把本地测试数据、源码目录或事后修改过且未重新签名的 app bundle 当成 release asset 上传。
