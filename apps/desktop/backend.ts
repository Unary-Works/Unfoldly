import type { FileSource } from './types';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { open } from '@tauri-apps/plugin-dialog';

// ─── File / Folder Selection (Tauri dialog) ───

export async function selectFolder(): Promise<string | null> {
  const result = await open({ directory: true, multiple: false });
  if (typeof result === 'string') return result;
  return null;
}

export async function selectFiles(): Promise<string[] | null> {
  const result = await open({ directory: false, multiple: true });
  if (Array.isArray(result)) return result as string[];
  if (typeof result === 'string') return [result];
  return null;
}

export async function checkMediaFiles(path: string): Promise<boolean> {
  try {
    return await invoke('check_media_files', { path });
  } catch (e) {
    console.warn('Failed to check media files:', e);
    return false;
  }
}

export async function notifyUIReady(): Promise<any> {
  return await invoke('notify_ui_ready');
}

// ─── Sources ───

let _listSourcesChain: Promise<unknown> = Promise.resolve();

export async function fetchSources(): Promise<FileSource[]> {
  const run = async () => {
    const data = await invoke<{ sources: FileSource[] }>('list_sources');
    return data.sources || [];
  };
  const next = _listSourcesChain.then(run, run);
  _listSourcesChain = next.then(
    () => undefined,
    () => undefined
  );
  return next as Promise<FileSource[]>;
}

export async function getActiveIndexJob(): Promise<any> {
  return await invoke('get_active_job');
}

export async function startIndex(folder: string): Promise<{ ok?: boolean; job_id?: string; folder?: string; error?: string; message?: string }> {
  return await invoke('start_index', { folder });
}

export async function indexFiles(files: string[]): Promise<{ ok: boolean; job_id: string; files: string[]; error?: string }> {
  return await invoke('index_files', { files });
}

export async function removeSource(folder: string): Promise<{
  ok: boolean;
  folder: string;
  deleted_count?: number;
  message?: string;
  error?: string;
}> {
  return await invoke('remove_source', { folder });
}

export async function removeSourcesBatch(folders: string[]): Promise<{
  ok: boolean;
  folders: string[];
  deleted_count?: number;
  error?: string;
}> {
  return await invoke('remove_sources_batch', { folders });
}

export async function refreshSource(folder: string): Promise<{
  job_id: string;
  folder: string;
  already_running?: boolean;
}> {
  return await invoke('refresh_source', { folder });
}

let _indexStatusChain: Promise<unknown> = Promise.resolve();

export async function getIndexStatus(jobId: string): Promise<any> {
  const run = () => invoke('get_index_status', { jobId });
  const next = _indexStatusChain.then(run, run);
  _indexStatusChain = next.then(
    () => undefined,
    () => undefined
  );
  return next;
}

export async function cancelIndex(jobId?: string | null): Promise<{ ok: boolean; job_id?: string; error?: string; active_job_id?: string }> {
  return await invoke('cancel_index', jobId ? { jobId } : {});
}

export async function skipFiles(filePaths: string[]): Promise<{ ok: boolean; skipped?: string[]; job_id?: string }> {
  return await invoke('skip_files', { filePaths });
}

// ─── Query ───

export async function queryBackend(payload: {
  message: string;
  active_source_ids?: string[];
  model_id?: string;
  session_id?: string;
  language?: string;
  opened_file_path?: string;
}): Promise<{
  answer: string;
  sources: any[];
  query_type: string;
  need_clarify: boolean;
  relevantFiles: { id: string; name: string; type: any; path?: string; doc_summary?: string }[];
}> {
  return await invoke('query', {
    message: payload.message,
    activeSourceIds: payload.active_source_ids,
    modelId: payload.model_id,
    sessionId: payload.session_id,
    language: payload.language,
    openedFilePath: payload.opened_file_path,
  });
}

// ─── History ───

export async function fetchHistory(): Promise<any[]> {
  const data = await invoke<{ sessions: any[] }>('get_history');
  return data.sessions || [];
}

export async function syncHistory(session: any): Promise<void> {
  await invoke('sync_history', { session });
}

export async function deleteHistory(id: string): Promise<void> {
  await invoke('delete_history', { id });
}

// ─── Models ───

export async function fetchModels(): Promise<any[]> {
  const data = await invoke<{ models: any[] }>('list_models');
  return data.models || [];
}

export type DownloadModelResult = {
  ok?: boolean;
  error?: string;
  job_id?: string;
  [key: string]: any;
};

export async function downloadModel(modelId: string, source: string, quantizationFile?: string): Promise<DownloadModelResult> {
  return await invoke('download_model', {
    // Backward/forward compatibility for different Tauri argument naming variants.
    model_id: modelId,
    modelId,
    modelIdd: modelId,
    source,
    quantization_file: quantizationFile,
    quantizationFile,
  });
}

export async function deleteModel(modelId: string, quantizationFile?: string): Promise<any> {
  return await invoke('delete_model', { modelId, quantizationFile: quantizationFile || null });
}

export async function cancelDownloadModel(modelId: string): Promise<any> {
  return await invoke('cancel_download', { modelId });
}

export async function selectModel(modelId: string): Promise<any> {
  return await invoke('select_model', { modelId });
}

export async function selectModelQuantization(modelId: string, quantizationFile: string): Promise<any> {
  return await invoke('select_quantization', {
    modelId,
    quantizationFile,
  });
}

// ─── Core Models (Embedding / Reranker) ───

export type CoreModelItemStatus = {
  installed: boolean;
  status: 'idle' | 'downloading' | 'installed' | 'error';
  error?: string;
  percent?: number;
  speed?: number;
  eta?: number;
  downloaded_bytes?: number;
  total_bytes?: number;
};

export type CoreModelsStatus = {
  embedding: CoreModelItemStatus;
  reranker: CoreModelItemStatus;
  progress: number;
  is_downloading: boolean;
};

export async function fetchCoreModelsStatus(): Promise<CoreModelsStatus> {
  return await invoke('core_models_status');
}

export async function startCoreModelsDownload(): Promise<any> {
  return await invoke('download_core_models');
}

export async function cancelCoreModelsDownload(): Promise<any> {
  return await invoke('cancel_core_models_download');
}

// ─── ASR Model (Whisper) ───

export async function fetchAsrModelStatus(): Promise<any> {
  return await invoke('asr_model_status');
}

export async function startAsrModelDownload(): Promise<any> {
  return await invoke('download_asr_model');
}

export async function cancelAsrModelDownload(): Promise<any> {
  return await invoke('cancel_asr_model_download');
}

// ─── Settings ───

export async function getSettings(): Promise<any> {
  return await invoke('get_settings');
}

export async function updateSettings(key: string, value: any): Promise<any> {
  return await invoke('update_settings', { key, value });
}

// ─── File open / reveal ───

export async function openPath(path: string): Promise<void> {
  await invoke('open_file_native', { path });
}

export async function revealPath(path: string): Promise<void> {
  await invoke('reveal_file_native', { path });
}

export async function openExternalUrl(url: string): Promise<void> {
  await invoke('open_external_url', { url });
}

// ─── Streaming Query ───

export async function abortQuery(sessionId: string): Promise<void> {
  try {
    await invoke('abort_query', { sessionId });
  } catch (err) {
    console.error("Failed to abort query:", err);
  }
}

export async function queryBackendStream(
  payload: { message: string; active_source_ids?: string[]; model_id?: string; session_id?: string; language?: string; opened_file_path?: string },
  onEvent: (event: string, data: any) => void,
  signal?: AbortSignal,
): Promise<void> {
  const requestId = `qs_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
  let unlisten: (() => void) | null = null;

  // Expose resolve/reject so the listener callback can settle the promise.
  let resolve!: () => void;
  let reject!: (reason?: any) => void;
  const done = new Promise<void>((res, rej) => { resolve = res; reject = rej; });

  const cleanup = () => { if (unlisten) { unlisten(); unlisten = null; } };

  if (signal?.aborted) {
    abortQuery(payload.session_id || "");
    throw new DOMException('Aborted', 'AbortError');
  }
  signal?.addEventListener('abort', () => {
    cleanup();
    abortQuery(payload.session_id || "");
    reject(new DOMException('Aborted', 'AbortError'));
  });

  // Await listener registration BEFORE invoking so no early events are lost.
  // Previously listen() was not awaited, causing the first status/thinking
  // events to be emitted before the listener was registered — appearing as a
  // cold-start delay on the first query.
  unlisten = await listen('chat-token', (ev: { payload: any }) => {
    const p = ev.payload;
    if (!p || p.request_id !== requestId) return;
    const eventName = p?.event || 'message';
    const eventData = p?.data || p;
    onEvent(eventName, eventData);
    if (eventName === 'done') {
      cleanup();
      resolve();
    }
  });

  await invoke('query_stream', {
    message: payload.message,
    activeSourceIds: payload.active_source_ids,
    modelId: payload.model_id,
    sessionId: payload.session_id,
    language: payload.language,
    requestId,
    openedFilePath: payload.opened_file_path,
  });
  await done;
}
