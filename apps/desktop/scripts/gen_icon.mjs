import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execSync } from 'node:child_process';

// Always resolve paths relative to this package,
// not the caller's cwd (build scripts may run from repo root).
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');
const BUILD_DIR = path.join(ROOT, 'build');
const LOGO_PNG = path.join(ROOT, 'assets', 'logo.png');
const OUT_PNG = path.join(BUILD_DIR, 'icon.png');

function crc32(buf) {
  // standard CRC32 (IEEE)
  let crc = 0xffffffff;
  for (let i = 0; i < buf.length; i++) {
    crc ^= buf[i];
    for (let j = 0; j < 8; j++) {
      const mask = -(crc & 1);
      crc = (crc >>> 1) ^ (0xedb88320 & mask);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const typeBuf = Buffer.from(type, 'ascii');
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const crcBuf = Buffer.alloc(4);
  const crc = crc32(Buffer.concat([typeBuf, data]));
  crcBuf.writeUInt32BE(crc, 0);
  return Buffer.concat([len, typeBuf, data, crcBuf]);
}

function makePngRGBA(width, height, rgbaBuffer) {
  const signature = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr.writeUInt8(8, 8); // bit depth
  ihdr.writeUInt8(6, 9); // color type RGBA
  ihdr.writeUInt8(0, 10); // compression
  ihdr.writeUInt8(0, 11); // filter
  ihdr.writeUInt8(0, 12); // interlace

  // scanlines: each row starts with filter byte 0
  const stride = width * 4;
  const scan = Buffer.alloc((stride + 1) * height);
  for (let y = 0; y < height; y++) {
    scan[(stride + 1) * y] = 0;
    rgbaBuffer.copy(scan, (stride + 1) * y + 1, stride * y, stride * (y + 1));
  }

  const idatData = zlib.deflateSync(scan, { level: 9 });
  const png = Buffer.concat([
    signature,
    chunk('IHDR', ihdr),
    chunk('IDAT', idatData),
    chunk('IEND', Buffer.alloc(0)),
  ]);
  return png;
}

function fillRect(img, w, h, x0, y0, x1, y1, r, g, b, a) {
  const ix0 = Math.max(0, Math.min(w, Math.floor(x0)));
  const iy0 = Math.max(0, Math.min(h, Math.floor(y0)));
  const ix1 = Math.max(0, Math.min(w, Math.floor(x1)));
  const iy1 = Math.max(0, Math.min(h, Math.floor(y1)));
  for (let y = iy0; y < iy1; y++) {
    for (let x = ix0; x < ix1; x++) {
      const i = (y * w + x) * 4;
      img[i] = r;
      img[i + 1] = g;
      img[i + 2] = b;
      img[i + 3] = a;
    }
  }
}

async function main() {
  fs.mkdirSync(BUILD_DIR, { recursive: true });

  // 检查 logo.png 是否存在
  if (!fs.existsSync(LOGO_PNG)) {
    console.error(`[gen_icon] logo.png not found at ${LOGO_PNG}`);
    process.exit(1);
  }

  console.log(`[gen_icon] Using logo from ${LOGO_PNG}`);

  // 方案1：尝试使用 sharp（如果已安装）
  try {
    const sharp = await import('sharp');
    await sharp.default(LOGO_PNG)
      .resize(1024, 1024, { fit: 'contain', background: { r: 255, g: 255, b: 255, alpha: 0 } })
      .png()
      .toFile(OUT_PNG);
    console.log(`[gen_icon] ✓ Generated ${OUT_PNG} using sharp`);
    return;
  } catch (e) {
    console.log(`[gen_icon] sharp not available, trying macOS sips...`);
  }

  // 方案2：使用 macOS sips 命令行工具
  try {
    // macOS sips 直接缩放并输出 PNG
    execSync(`sips -z 1024 1024 "${LOGO_PNG}" --out "${OUT_PNG}" 2>/dev/null`, { stdio: 'inherit' });
    console.log(`[gen_icon] ✓ Generated ${OUT_PNG} using sips`);
    return;
  } catch (e) {
    console.log(`[gen_icon] sips failed, falling back to direct copy...`);
  }

  // 方案3：直接复制 PNG（最后兜底）
  fs.copyFileSync(LOGO_PNG, OUT_PNG);
  console.log(`[gen_icon] ⚠️  Copied ${LOGO_PNG} to ${OUT_PNG}`);
}

main().catch((err) => {
  console.error('[gen_icon] Error:', err);
  process.exit(1);
});
