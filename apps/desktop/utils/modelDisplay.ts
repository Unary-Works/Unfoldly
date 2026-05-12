import type { Model } from '../types';

export function formatModelName(name?: string): string {
  const raw = String(name || '').trim();
  if (!raw) return '';
  return raw
    .replace(/gguf/gi, '')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Remove redundant “(Q4_K_M)” / “(Q4 K M)” from display name when quant is shown as a separate badge. */
function stripParentheticalQuant(formattedName: string, level: string): string {
  const spaced = level.replace(/_/g, ' ').trim();
  let s = formattedName;
  for (const token of [level, spaced]) {
    if (!token) continue;
    const esc = token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    s = s.replace(new RegExp(`\\s*\\(${esc}\\)\\s*`, 'gi'), ' ');
  }
  return s.replace(/\s+/g, ' ').trim();
}

/**
 * Manage models table: base name (primary style) + optional quant badge (secondary style).
 * Single-quantization models used to put quant only in `name`; multi-quant uses per-row `quantizationFile`.
 */
export function getManageModelsRowDisplay(
  model: Model,
  rowQuantFile?: string,
): { baseName: string; quantBadge: string | null } {
  const quants = Array.isArray(model.quantizations) ? model.quantizations : [];
  const hasQuantRows = quants.length > 1;
  const qf = String(rowQuantFile || '').trim()
    || (!hasQuantRows && quants.length === 1 ? String(quants[0]?.file || '').trim() : '');
  if (!qf) {
    return { baseName: formatModelName(model.name), quantBadge: null };
  }
  const qMeta = quants.find((q) => String(q?.file || '') === qf);
  const level = String(qMeta?.level || '').trim();
  const quantBadge = level || qf;
  let baseName = formatModelName(model.name);
  if (!String(rowQuantFile || '').trim() && level) {
    baseName = stripParentheticalQuant(baseName, level);
  }
  return { baseName, quantBadge };
}

/**
 * Chat model dropdown: extract a short quantization badge for a Model.
 * Returns null if there is only one (or no) quantization so we don't clutter simple cases.
 */
export function getModelQuantBadge(model: Model): string | null {
  const quants = Array.isArray(model.quantizations) ? model.quantizations : [];
  // Only show quant badge when multiple quantizations exist for this model ID
  if (quants.length <= 1) return null;

  // Prefer the currently selected / active quantization file
  const selectedQF = String(model.selected_quantization || model.default_quantization || '').trim();
  if (selectedQF) {
    const qMeta = quants.find((q) => String(q?.file || '') === selectedQF);
    const level = String(qMeta?.level || '').trim();
    if (level) return level;
    // Fallback: strip .gguf and take last 2 segments (e.g. Q5_K_S)
    return selectedQF.replace(/\.gguf$/i, '').split(/[-_]/).slice(-2).join('_') || selectedQF;
  }

  // Fallback: first installed quantization
  const installed = Array.isArray(model.installed_quantizations) ? model.installed_quantizations : [];
  if (installed.length > 0) {
    const qMeta = quants.find((q) => String(q?.file || '') === installed[0]);
    const level = String(qMeta?.level || '').trim();
    if (level) return level;
    return String(installed[0]).replace(/\.gguf$/i, '').split(/[-_]/).slice(-2).join('_') || String(installed[0]);
  }

  return null;
}
