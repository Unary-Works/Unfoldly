import type { FileSource } from '../types';

export function isIndexedKbFile(node: FileSource): boolean {
  return node.type === 'file' && node.status === 'indexed';
}

function buildIdToPathMap(nodes: FileSource[]): Map<string, string> {
  const m = new Map<string, string>();
  const walk = (n: FileSource) => {
    if (n.path && n.path.length > 0) {
      m.set(n.id, n.path);
    }
    n.children?.forEach(walk);
  };
  nodes.forEach(walk);
  return m;
}

/**
 * Count selected files that are already indexed, deduplicated by path.
 */
export function countDistinctSelectedFiles(nodes: FileSource[], selectedIds: string[]): number {
  const selected = new Set(selectedIds);
  const seen = new Set<string>();
  const walk = (list: FileSource[]) => {
    for (const node of list) {
      if (isIndexedKbFile(node) && selected.has(node.id)) {
        const key = node.path && node.path.length > 0 ? node.path : node.id;
        seen.add(key);
      }
      if (node.children?.length) {
        walk(node.children);
      }
    }
  };
  walk(nodes);
  return seen.size;
}

/**
 * Build active_source_ids for the backend with only one id per file path.
 */
export function dedupeEffectiveSourceIdsByPath(nodes: FileSource[], ids: string[]): string[] {
  const idToPath = buildIdToPathMap(nodes);
  const seenPaths = new Set<string>();
  const out: string[] = [];
  for (const id of ids) {
    const p = idToPath.get(id);
    if (p !== undefined) {
      if (seenPaths.has(p)) continue;
      seenPaths.add(p);
    }
    out.push(id);
  }
  return out;
}
