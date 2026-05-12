#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

function readArg(name, fallback) {
  const index = process.argv.indexOf(name);
  if (index === -1) return fallback;
  return process.argv[index + 1] || fallback;
}

const repo = readArg('--repo', 'Unary-Works/Unfoldly');
const tag = readArg('--tag', '');
const notes = readArg('--notes', '');
const archive = readArg(
  '--archive',
  path.resolve(process.cwd(), '..', '..', 'macos_bundle', 'release', 'Unfoldly.app.tar.gz'),
);
const signaturePath = readArg('--signature', `${archive}.sig`);
const output = readArg(
  '--output',
  path.resolve(process.cwd(), '..', '..', 'macos_bundle', 'release', 'latest.json'),
);
const defaultPlatform =
  process.platform === 'darwin' && process.arch === 'arm64'
    ? 'darwin-aarch64'
    : process.platform === 'darwin' && process.arch === 'x64'
      ? 'darwin-x86_64'
      : 'darwin-aarch64';
const platformsArg = readArg('--platforms', defaultPlatform);

if (!tag) {
  console.error('Missing --tag v1.0.0');
  process.exit(1);
}

if (!fs.existsSync(signaturePath)) {
  console.error(`Missing signature file: ${signaturePath}`);
  process.exit(1);
}

const version = tag.replace(/^v/i, '');
if (!/^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$/.test(version)) {
  console.error(`Invalid SemVer tag: ${tag}`);
  process.exit(1);
}

const signature = fs.readFileSync(signaturePath, 'utf8').trim();
const assetName = path.basename(archive);
const url = `https://github.com/${repo}/releases/download/${tag}/${encodeURIComponent(assetName)}`;
const platforms = {};

for (const platform of platformsArg.split(',').map((item) => item.trim()).filter(Boolean)) {
  platforms[platform] = {
    signature,
    url,
  };
}

const manifest = {
  version,
  notes,
  pub_date: new Date().toISOString(),
  platforms,
};

fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`Wrote updater manifest: ${output}`);
