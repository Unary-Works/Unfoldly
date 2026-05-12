#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const version = process.argv[2]?.trim().replace(/^v/i, '');

if (!version || !/^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$/.test(version)) {
  console.error('Usage: node scripts/set-version.mjs 1.2.3');
  process.exit(1);
}

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const desktopRoot = path.resolve(scriptDir, '..');
const repoRoot = path.resolve(desktopRoot, '..', '..');
const packagePath = path.join(desktopRoot, 'package.json');
const tauriConfigPath = path.join(desktopRoot, 'src-tauri', 'tauri.conf.json');
const cargoPath = path.join(desktopRoot, 'src-tauri', 'Cargo.toml');
const backendPyprojectPath = path.join(repoRoot, 'pyproject.toml');

const packageJson = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
packageJson.version = version;
fs.writeFileSync(packagePath, `${JSON.stringify(packageJson, null, 2)}\n`);

const tauriConfig = JSON.parse(fs.readFileSync(tauriConfigPath, 'utf8'));
tauriConfig.version = version;
fs.writeFileSync(tauriConfigPath, `${JSON.stringify(tauriConfig, null, 2)}\n`);

const cargoToml = fs.readFileSync(cargoPath, 'utf8');
fs.writeFileSync(
  cargoPath,
  cargoToml.replace(/^version = ".*"$/m, `version = "${version}"`),
);

if (fs.existsSync(backendPyprojectPath)) {
  const pyprojectToml = fs.readFileSync(backendPyprojectPath, 'utf8');
  fs.writeFileSync(
    backendPyprojectPath,
    pyprojectToml.replace(/^version = ".*"$/m, `version = "${version}"`),
  );
}

console.log(`Synced desktop and backend versions to ${version}`);
