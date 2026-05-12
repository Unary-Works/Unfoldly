use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::Serialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tauri::{ipc::Channel, AppHandle, Emitter, Manager, ResourceId, Runtime, State, Webview};
use tokio::sync::{oneshot, Mutex};

use crate::AppState;

#[derive(Default)]
pub struct UpdateDownloadState {
    next_id: AtomicU64,
    active_cancel: Mutex<Option<ActiveUpdateDownload>>,
}

struct ActiveUpdateDownload {
    id: u64,
    sender: oneshot::Sender<()>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event", content = "data")]
pub enum UpdateDownloadEvent {
    #[serde(rename_all = "camelCase")]
    Started { content_length: Option<u64> },
    #[serde(rename_all = "camelCase")]
    Progress { chunk_length: usize },
    Finished,
    Cancelled,
}

const UPDATE_DOWNLOAD_CANCELLED: &str = "update_download_cancelled";

fn expand_user_path(raw: &str) -> PathBuf {
    let trimmed = raw.trim();
    if trimmed == "~" || trimmed.starts_with("~/") {
        if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
            let suffix = trimmed.strip_prefix("~/").unwrap_or("");
            return PathBuf::from(home).join(suffix);
        }
    }
    PathBuf::from(trimmed)
}

fn canonical_existing_path(raw: &str) -> Result<PathBuf, String> {
    if raw.trim().is_empty() {
        return Err("empty_path".to_string());
    }
    std::fs::canonicalize(expand_user_path(raw))
        .map_err(|_| "path_not_found_or_inaccessible".to_string())
}

fn collect_json_paths(value: &Value) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(obj) = value.as_object() {
        for key in ["folders", "files"] {
            if let Some(items) = obj.get(key).and_then(|v| v.as_array()) {
                out.extend(items.iter().filter_map(|v| v.as_str()).map(str::to_string));
            }
        }
    } else if let Some(items) = value.as_array() {
        out.extend(items.iter().filter_map(|v| v.as_str()).map(str::to_string));
    }
    out
}

fn read_source_paths(path: &Path) -> Vec<String> {
    let Ok(raw) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<Value>(&raw) else {
        return Vec::new();
    };
    collect_json_paths(&value)
}

fn configured_index_source_paths() -> Vec<PathBuf> {
    let Ok(data_dir) = std::env::var("FILEAGENT_DATA_DIR") else {
        return Vec::new();
    };
    let data_dir = expand_user_path(&data_dir);
    let mut raw_paths = Vec::new();
    raw_paths.extend(read_source_paths(&data_dir.join("indexed_sources.json")));
    raw_paths.extend(read_source_paths(&data_dir.join("indexed_folders.json")));

    raw_paths
        .into_iter()
        .filter_map(|p| std::fs::canonicalize(expand_user_path(&p)).ok())
        .collect()
}

fn is_indexed_source_path(target: &Path) -> bool {
    configured_index_source_paths().into_iter().any(|source| {
        if source.is_dir() {
            target == source || target.starts_with(&source)
        } else {
            target == source
        }
    })
}

fn resolve_openable_indexed_path(raw: &str) -> Result<PathBuf, String> {
    let target = canonical_existing_path(raw)?;
    if is_indexed_source_path(&target) {
        Ok(target)
    } else {
        Err("path_is_not_an_indexed_source".to_string())
    }
}

fn validate_external_url(raw: &str) -> Result<String, String> {
    let url = raw.trim();
    if url.is_empty() {
        return Err("empty_url".to_string());
    }
    if !(url.starts_with("https://") || url.starts_with("http://")) {
        return Err("unsupported_url_scheme".to_string());
    }
    if url.contains('\0')
        || url.contains('\n')
        || url.contains('\r')
        || url.contains('\t')
    {
        return Err("invalid_url".to_string());
    }
    Ok(url.to_string())
}

#[tauri::command]
pub fn open_file_native(path: String) -> Result<(), String> {
    let path = resolve_openable_indexed_path(&path)?;
    #[cfg(target_os = "macos")]
    {
        Command::new("open").arg(&path).spawn().map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "windows")]
    {
        let path_s = path.to_string_lossy().to_string();
        Command::new("cmd").args(&["/C", "start", "", &path_s]).spawn().map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "linux")]
    {
        Command::new("xdg-open").arg(&path).spawn().map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
pub fn open_external_url(url: String) -> Result<(), String> {
    let url = validate_external_url(&url)?;
    #[cfg(target_os = "macos")]
    {
        Command::new("open").arg(&url).spawn().map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "windows")]
    {
        Command::new("cmd")
            .args(&["/C", "start", "", &url])
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "linux")]
    {
        Command::new("xdg-open").arg(&url).spawn().map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
pub async fn download_and_install_update_cancelable<R: Runtime>(
    webview: Webview<R>,
    state: State<'_, UpdateDownloadState>,
    rid: ResourceId,
    on_event: Channel<UpdateDownloadEvent>,
) -> Result<(), String> {
    let session_id = state.next_id.fetch_add(1, Ordering::Relaxed) + 1;
    let (cancel_tx, cancel_rx) = oneshot::channel::<()>();
    {
        let mut active_cancel = state.active_cancel.lock().await;
        if let Some(previous) = active_cancel.take() {
            let _ = previous.sender.send(());
        }
        *active_cancel = Some(ActiveUpdateDownload {
            id: session_id,
            sender: cancel_tx,
        });
    }

    let update = webview
        .resources_table()
        .get::<tauri_plugin_updater::Update>(rid)
        .map_err(|err| err.to_string())?;
    let update = (*update).clone();

    let progress_channel = on_event.clone();
    let finish_channel = on_event.clone();
    let mut first_chunk = true;
    let download = update.download(
        move |chunk_length, content_length| {
            if first_chunk {
                first_chunk = false;
                let _ = progress_channel.send(UpdateDownloadEvent::Started { content_length });
            }
            let _ = progress_channel.send(UpdateDownloadEvent::Progress { chunk_length });
        },
        move || {
            let _ = finish_channel.send(UpdateDownloadEvent::Finished);
        },
    );

    let result = tokio::select! {
        _ = cancel_rx => {
            let _ = on_event.send(UpdateDownloadEvent::Cancelled);
            Err(UPDATE_DOWNLOAD_CANCELLED.to_string())
        }
        bytes = download => {
            let bytes = bytes.map_err(|err| err.to_string())?;
            update.install(bytes).map_err(|err| err.to_string())
        }
    };

    let mut active_cancel = state.active_cancel.lock().await;
    if active_cancel
        .as_ref()
        .map(|active| active.id == session_id)
        .unwrap_or(false)
    {
        *active_cancel = None;
    }

    result
}

#[tauri::command]
pub async fn cancel_update_download(state: State<'_, UpdateDownloadState>) -> Result<(), String> {
    if let Some(active) = state.active_cancel.lock().await.take() {
        let _ = active.sender.send(());
    }
    Ok(())
}

#[tauri::command]
pub fn reveal_file_native(path: String) -> Result<(), String> {
    let path = resolve_openable_indexed_path(&path)?;
    #[cfg(target_os = "macos")]
    {
        Command::new("open").arg("-R").arg(&path).spawn().map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "windows")]
    {
        let path_s = path.to_string_lossy().to_string();
        Command::new("explorer").args(&["/select,", &path_s]).spawn().map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "linux")]
    {
        if let Some(parent) = path.parent() {
            Command::new("xdg-open").arg(parent).spawn().map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

async fn py_call_spawn(
    backend: Arc<Py<PyAny>>,
    method: String,
    kwargs: Option<Vec<(String, Value)>>,
) -> Result<Value, String> {
    let label_join = method.clone();
    tokio::task::spawn_blocking(move || {
        let label_inner = method.clone();
        Python::with_gil(|py| {
            let py_obj = if let Some(ref pairs) = kwargs {
                let dict = PyDict::new(py);
                for (k, v) in pairs {
                    let py_val = pythonize::pythonize(py, v).map_err(|e| e.to_string())?;
                    dict.set_item(k.as_str(), py_val).map_err(|e| e.to_string())?;
                }
                backend
                    .call_method(py, method.as_str(), (), Some(&dict))
                    .map_err(|e| format!("Python error in {method}: {e}"))?
            } else {
                backend
                    .call_method0(py, method.as_str())
                    .map_err(|e| format!("Python error in {method}: {e}"))?
            };
            pythonize::depythonize::<Value>(py_obj.bind(py))
                .map_err(|e| format!("Deserialization error ({label_inner}): {e}"))
        })
    })
    .await
    .map_err(|e| format!("spawn_blocking({label_join}): {e}"))?
}

fn own_kw(kw: &[(&str, Value)]) -> Vec<(String, Value)> {
    kw.iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

// ─── Health ───

#[tauri::command]
pub async fn health_check(state: State<'_, AppState>) -> Result<Value, String> {
    py_call_spawn(Arc::clone(&state.backend), "health_check".into(), None).await
}

#[tauri::command]
pub async fn get_runtime_paths(state: State<'_, AppState>) -> Result<Value, String> {
    py_call_spawn(
        Arc::clone(&state.backend),
        "get_runtime_paths".into(),
        None,
    )
    .await
}

#[tauri::command]
pub async fn notify_ui_ready(state: State<'_, AppState>) -> Result<Value, String> {
    py_call_spawn(Arc::clone(&state.backend), "notify_ui_ready".into(), None).await
}

// ─── Sources ───

#[tauri::command]
pub async fn list_sources(state: State<'_, AppState>) -> Result<Value, String> {
    py_call_spawn(Arc::clone(&state.backend), "list_sources".into(), None).await
}

#[tauri::command]
pub async fn add_source(state: State<'_, AppState>, folder: String) -> Result<Value, String> {
    py_call_spawn(
        Arc::clone(&state.backend),
        "add_source".into(),
        Some(own_kw(&[("folder", Value::String(folder))])),
    )
    .await
}

#[tauri::command]
pub async fn remove_source(state: State<'_, AppState>, folder: String) -> Result<Value, String> {
    py_call_spawn(
        Arc::clone(&state.backend),
        "remove_source".into(),
        Some(own_kw(&[("folder", Value::String(folder))])),
    )
    .await
}

#[tauri::command]
pub async fn remove_sources_batch(state: State<'_, AppState>, folders: Vec<String>) -> Result<Value, String> {
    let folders_val = Value::Array(folders.into_iter().map(Value::String).collect());
    py_call_spawn(
        Arc::clone(&state.backend),
        "remove_sources_batch".into(),
        Some(own_kw(&[("folders", folders_val)])),
    )
    .await
}

#[tauri::command]
pub async fn refresh_source(state: State<'_, AppState>, folder: String) -> Result<Value, String> {
    py_call_spawn(
        Arc::clone(&state.backend),
        "refresh_source".into(),
        Some(own_kw(&[("folder", Value::String(folder))])),
    )
    .await
}

// ─── History ───

#[tauri::command]
pub async fn get_history(state: State<'_, AppState>) -> Result<Value, String> {
    py_call_spawn(Arc::clone(&state.backend), "get_history".into(), None).await
}

#[tauri::command]
pub async fn sync_history(state: State<'_, AppState>, session: Value) -> Result<Value, String> {
    py_call_spawn(
        Arc::clone(&state.backend),
        "sync_history".into(),
        Some(own_kw(&[("session", session)])),
    )
    .await
}

#[tauri::command]
pub async fn delete_history(state: State<'_, AppState>, id: String) -> Result<Value, String> {
    py_call_spawn(
        Arc::clone(&state.backend),
        "delete_history".into(),
        Some(own_kw(&[("id", Value::String(id))])),
    )
    .await
}

// ─── Models ───

#[tauri::command]
pub async fn list_models(state: State<'_, AppState>) -> Result<Value, String> {
    py_call_spawn(Arc::clone(&state.backend), "list_models".into(), None).await
}

#[tauri::command]
#[allow(non_snake_case)]
pub async fn select_model(
    state: State<'_, AppState>,
    model_id: Option<String>,
    modelId: Option<String>,
) -> Result<Value, String> {
    let mid = model_id
        .or(modelId)
        .ok_or_else(|| "missing model_id".to_string())?;
    py_call_spawn(
        Arc::clone(&state.backend),
        "select_model".into(),
        Some(own_kw(&[("model_id", Value::String(mid))])),
    )
    .await
}

#[tauri::command]
#[allow(non_snake_case)]
pub async fn select_quantization(
    state: State<'_, AppState>,
    model_id: Option<String>,
    modelId: Option<String>,
    quantization_file: Option<String>,
    quantizationFile: Option<String>,
) -> Result<Value, String> {
    let mid = model_id
        .or(modelId)
        .ok_or_else(|| "missing model_id".to_string())?;
    let qf = quantization_file.or(quantizationFile).unwrap_or_default();
    py_call_spawn(
        Arc::clone(&state.backend),
        "select_quantization".into(),
        Some(own_kw(&[
            ("model_id", Value::String(mid)),
            ("quantization_file", Value::String(qf)),
        ])),
    )
    .await
}

#[tauri::command]
#[allow(non_snake_case)]
pub async fn download_model(
    state: State<'_, AppState>,
    model_id: Option<String>,
    modelId: Option<String>,
    modelIdd: Option<String>,
    source: String,
    quantization_file: Option<String>,
    quantizationFile: Option<String>,
) -> Result<Value, String> {
    let resolved_model_id = model_id
        .or(modelId)
        .or(modelIdd)
        .ok_or_else(|| "missing required model_id/modelId/modelIdd".to_string())?;
    let resolved_qf = quantization_file.or(quantizationFile);

    let mut pairs = vec![
        ("model_id".to_string(), Value::String(resolved_model_id)),
        ("source".to_string(), Value::String(source)),
    ];
    if let Some(qf) = resolved_qf {
        pairs.push(("quantization_file".to_string(), Value::String(qf)));
    }
    py_call_spawn(
        Arc::clone(&state.backend),
        "download_model".into(),
        Some(pairs),
    )
    .await
}

#[tauri::command]
pub async fn cancel_download(state: State<'_, AppState>, model_id: String) -> Result<Value, String> {
    py_call_spawn(
        Arc::clone(&state.backend),
        "cancel_download".into(),
        Some(own_kw(&[("model_id", Value::String(model_id))])),
    )
    .await
}

#[tauri::command]
#[allow(non_snake_case)]
pub async fn delete_model(
    state: State<'_, AppState>,
    model_id: Option<String>,
    modelId: Option<String>,
    quantization_file: Option<String>,
    quantizationFile: Option<String>,
) -> Result<Value, String> {
    let mid = model_id
        .or(modelId)
        .ok_or_else(|| "missing model_id".to_string())?;
    let resolved_qf = quantization_file.or(quantizationFile);

    let mut pairs = vec![("model_id".to_string(), Value::String(mid))];
    if let Some(qf) = resolved_qf {
        if !qf.is_empty() {
            pairs.push(("quantization_file".to_string(), Value::String(qf)));
        }
    }
    py_call_spawn(
        Arc::clone(&state.backend),
        "delete_model".into(),
        Some(pairs),
    )
    .await
}

// ─── Core Models ───

#[tauri::command]
pub async fn core_models_status(state: State<'_, AppState>) -> Result<Value, String> {
    py_call_spawn(
        Arc::clone(&state.backend),
        "core_models_status".into(),
        None,
    )
    .await
}

#[tauri::command]
pub async fn download_core_models(state: State<'_, AppState>) -> Result<Value, String> {
    py_call_spawn(
        Arc::clone(&state.backend),
        "download_core_models".into(),
        None,
    )
    .await
}

#[tauri::command]
pub async fn cancel_core_models_download(state: State<'_, AppState>) -> Result<Value, String> {
    py_call_spawn(
        Arc::clone(&state.backend),
        "cancel_core_models_download".into(),
        None,
    )
    .await
}

// ─── ASR Model ───

#[tauri::command]
pub async fn asr_model_status(state: State<'_, AppState>) -> Result<Value, String> {
    py_call_spawn(
        Arc::clone(&state.backend),
        "asr_model_status".into(),
        None,
    )
    .await
}

#[tauri::command]
pub async fn download_asr_model(state: State<'_, AppState>) -> Result<Value, String> {
    py_call_spawn(
        Arc::clone(&state.backend),
        "download_asr_model".into(),
        None,
    )
    .await
}

#[tauri::command]
pub async fn cancel_asr_model_download(state: State<'_, AppState>) -> Result<Value, String> {
    py_call_spawn(
        Arc::clone(&state.backend),
        "cancel_asr_model_download".into(),
        None,
    )
    .await
}

// ─── Indexing ───

#[tauri::command]
pub async fn get_active_job(state: State<'_, AppState>) -> Result<Value, String> {
    py_call_spawn(Arc::clone(&state.backend), "get_active_job".into(), None).await
}

#[tauri::command]
pub async fn start_index(state: State<'_, AppState>, folder: String) -> Result<Value, String> {
    py_call_spawn(
        Arc::clone(&state.backend),
        "start_index".into(),
        Some(own_kw(&[("folder", Value::String(folder))])),
    )
    .await
}

#[tauri::command]
pub async fn index_files(state: State<'_, AppState>, files: Vec<String>) -> Result<Value, String> {
    let files_val: Vec<Value> = files.into_iter().map(Value::String).collect();
    py_call_spawn(
        Arc::clone(&state.backend),
        "index_files".into(),
        Some(own_kw(&[("files", Value::Array(files_val))])),
    )
    .await
}

#[tauri::command]
#[allow(non_snake_case)]
pub async fn get_index_status(state: State<'_, AppState>, jobId: String) -> Result<Value, String> {
    py_call_spawn(
        Arc::clone(&state.backend),
        "get_index_status".into(),
        Some(own_kw(&[("job_id", Value::String(jobId))])),
    )
    .await
}

#[tauri::command]
#[allow(non_snake_case)]
pub async fn cancel_index(
    state: State<'_, AppState>,
    job_id: Option<String>,
    jobId: Option<String>,
) -> Result<Value, String> {
    let jid = job_id.or(jobId).unwrap_or_default();
    py_call_spawn(
        Arc::clone(&state.backend),
        "cancel_index".into(),
        Some(own_kw(&[("job_id", Value::String(jid))])),
    )
    .await
}

#[tauri::command]
#[allow(non_snake_case)]
pub async fn skip_files(state: State<'_, AppState>, filePaths: Vec<String>) -> Result<Value, String> {
    let files_val: Vec<Value> = filePaths.into_iter().map(Value::String).collect();
    py_call_spawn(
        Arc::clone(&state.backend),
        "skip_files".into(),
        Some(own_kw(&[("file_paths", Value::Array(files_val))])),
    )
    .await
}

// ─── Query ───

#[tauri::command]
#[allow(non_snake_case)]
pub async fn abort_query(
    state: State<'_, AppState>,
    session_id: Option<String>,
    sessionId: Option<String>,
) -> Result<Value, String> {
    let sid = session_id.or(sessionId).unwrap_or_default();
    py_call_spawn(
        Arc::clone(&state.backend),
        "abort_query".into(),
        Some(own_kw(&[("session_id", Value::String(sid))])),
    )
    .await
}

#[tauri::command]
#[allow(non_snake_case)]
pub async fn query(
    state: State<'_, AppState>,
    message: String,
    activeSourceIds: Option<Vec<String>>,
    modelId: Option<String>,
    sessionId: Option<String>,
    language: Option<String>,
    openedFilePath: Option<String>,
) -> Result<Value, String> {
    let mut pairs: Vec<(String, Value)> =
        vec![("message".to_string(), Value::String(message))];
    if let Some(ids) = activeSourceIds {
        pairs.push((
            "active_source_ids".to_string(),
            Value::Array(ids.into_iter().map(Value::String).collect()),
        ));
    }
    if let Some(mid) = modelId {
        pairs.push(("model_id".to_string(), Value::String(mid)));
    }
    if let Some(sid) = sessionId {
        pairs.push(("session_id".to_string(), Value::String(sid)));
    }
    if let Some(lang) = language {
        pairs.push(("language".to_string(), Value::String(lang)));
    }
    if let Some(opened) = openedFilePath {
        pairs.push(("opened_file_path".to_string(), Value::String(opened)));
    }
    py_call_spawn(
        Arc::clone(&state.backend),
        "query".into(),
        Some(pairs),
    )
    .await
}

// ─── Query (streaming via Tauri events) ───

#[tauri::command]
#[allow(non_snake_case)]
pub fn query_stream(
    app: AppHandle,
    state: State<AppState>,
    message: String,
    activeSourceIds: Option<Vec<String>>,
    modelId: Option<String>,
    sessionId: Option<String>,
    language: Option<String>,
    requestId: Option<String>,
    openedFilePath: Option<String>,
) -> Result<(), String> {
    fn attach_request_id(mut payload: Value, request_id: &Option<String>) -> Value {
        if let Some(rid) = request_id {
            if let Value::Object(ref mut map) = payload {
                map.insert("request_id".to_string(), Value::String(rid.clone()));
            }
        }
        payload
    }

    let backend = state.backend.clone();
    let active_ids_clone = activeSourceIds.clone();
    let model_id_clone = modelId.clone();
    let session_id_clone = sessionId.clone();
    let language_clone = language.clone();
    let request_id_clone = requestId.clone();
    let opened_file_path_clone = openedFilePath.clone();

    std::thread::spawn(move || {
        Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            let query_stream_debug = std::env::var("FILEAGENT_QUERY_STREAM_DEBUG")
                .ok()
                .map(|v| {
                    let s = v.trim().to_ascii_lowercase();
                    matches!(s.as_str(), "1" | "true" | "yes" | "on")
                })
                .unwrap_or(false);
            kwargs.set_item("message", &message).ok();
            if let Some(ref ids) = active_ids_clone {
                kwargs.set_item("active_source_ids", ids).ok();
                if query_stream_debug {
                    eprintln!("[query_stream] passing active_source_ids count: {}", ids.len());
                }
            } else if query_stream_debug {
                eprintln!("[query_stream] active_source_ids is None!");
            }
            if let Some(ref mid) = model_id_clone {
                kwargs.set_item("model_id", mid).ok();
            }
            if let Some(ref sid) = session_id_clone {
                kwargs.set_item("session_id", sid).ok();
            }
            if let Some(ref lang) = language_clone {
                kwargs.set_item("language", lang).ok();
            }
            if let Some(ref opened) = opened_file_path_clone {
                kwargs.set_item("opened_file_path", opened).ok();
            }

            let gen_result = backend.call_method(py, "query_stream", (), Some(&kwargs));
            let generator = match gen_result {
                Ok(g) => g,
                Err(e) => {
                    let err_payload = attach_request_id(serde_json::json!({
                        "event": "done",
                        "data": {"type": "done", "ok": false, "error": e.to_string()}
                    }), &request_id_clone);
                    let _ = app.emit("chat-token", &err_payload);
                    return;
                }
            };

            loop {
                match generator.call_method0(py, "__next__") {
                    Ok(item) => {
                        match pythonize::depythonize::<Value>(item.bind(py)) {
                            Ok(payload) => {
                                let payload = attach_request_id(payload, &request_id_clone);
                                let _ = app.emit("chat-token", &payload);
                            }
                            Err(e) => {
                                eprintln!("[query_stream] depythonize error: {e}");
                            }
                        }
                    }
                    Err(e) => {
                        if e.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) {
                            break;
                        }
                        let err_payload = attach_request_id(serde_json::json!({
                            "event": "done",
                            "data": {"type": "done", "ok": false, "error": e.to_string()}
                        }), &request_id_clone);
                        let _ = app.emit("chat-token", &err_payload);
                        break;
                    }
                }
            }
        });
    });

    Ok(())
}

// ─── Settings ───

#[tauri::command]
pub async fn check_media_files(path: String) -> Result<bool, String> {
    tokio::task::spawn_blocking(move || {
        let media_extensions = ["mp3", "mp4", "wav", "m4a", "webm", "ogg", "mov", "avi", "mkv", "flac", "aac"];
        fn check_dir(dir: &std::path::Path, exts: &[&str]) -> bool {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.filter_map(Result::ok) {
                    let path = entry.path();
                    if path.is_file() {
                        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                            if exts.contains(&ext.to_lowercase().as_str()) {
                                return true;
                            }
                        }
                    } else if path.is_dir() {
                        if check_dir(&path, exts) {
                            return true;
                        }
                    }
                }
            }
            false
        }
        
        let path_obj = std::path::Path::new(&path);
        if path_obj.is_file() {
            if let Some(ext) = path_obj.extension().and_then(|e| e.to_str()) {
                if media_extensions.contains(&ext.to_lowercase().as_str()) {
                    return Ok(true);
                }
            }
            Ok(false)
        } else {
            Ok(check_dir(path_obj, &media_extensions))
        }
    })
    .await
    .map_err(|e| format!("spawn_blocking error: {}", e))?
}

#[tauri::command]
pub async fn get_settings(state: State<'_, AppState>) -> Result<Value, String> {
    py_call_spawn(Arc::clone(&state.backend), "get_settings".into(), None).await
}

#[tauri::command]
pub async fn update_settings(
    state: State<'_, AppState>,
    key: String,
    value: Value,
) -> Result<Value, String> {
    py_call_spawn(
        Arc::clone(&state.backend),
        "update_settings".into(),
        Some(own_kw(&[
            ("key", Value::String(key)),
            ("value", value),
        ])),
    )
    .await
}
