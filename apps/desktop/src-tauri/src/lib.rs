mod commands;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::prelude::*;
use pyo3::types::PyModule;

/// macOS: background drag-drop enhancement is temporarily disabled.
///
/// We previously called `setAcceptsMouseMovedEvents:` via a raw `NSWindow*`
/// bridge here so the window could receive drag events while unfocused. That
/// native bridge started causing occasional startup-time EXC_BAD_ACCESS on
/// recent macOS builds, and it is outside the indexing/search pipeline.
///
/// Keep this as a no-op so startup stays stable. The only tradeoff is that
/// dragging files onto an unfocused window may be less responsive on macOS.
#[cfg(target_os = "macos")]
fn configure_window_for_background_drag(_window: &tauri::WebviewWindow) {}

fn write_python_runtime_log() {
    #[cfg(target_os = "macos")]
    if let Some(home) = std::env::var_os("HOME") {
        let log_dir = PathBuf::from(home).join("Library/Logs/Unfoldly");
        let _ = std::fs::create_dir_all(&log_dir);
        let content = Python::with_gil(|py| -> String {
            let mut out = String::new();
            if let Ok(sys) = py.import("sys") {
                if let Ok(exe) = sys.getattr("executable").and_then(|o| o.repr()).and_then(|r| r.str()).and_then(|s| s.to_str().map(String::from)) {
                    out.push_str(&format!("sys.executable: {}\n", exe));
                }
                if let Ok(base) = sys.getattr("base_prefix").and_then(|o| o.repr()).and_then(|r| r.str()).and_then(|s| s.to_str().map(String::from)) {
                    out.push_str(&format!("sys.base_prefix: {}\n", base));
                }
                if let Ok(path) = sys.getattr("path").and_then(|o| o.repr()).and_then(|r| r.str()).and_then(|s| s.to_str().map(String::from)) {
                    out.push_str(&format!("sys.path: {}\n", path));
                }
            } else {
                out.push_str("(could not import sys)\n");
            }
            out
        });
        let _ = std::fs::write(log_dir.join("python_runtime.log"), content);
    }
}

fn write_stderr_safe(msg: &str) {
    let _ = std::panic::catch_unwind(|| {
        let _ = writeln!(std::io::stderr(), "{}", msg);
    });
}

fn install_crash_logger() {
    let default_panic = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg = format!("{}\n{:?}", info, std::backtrace::Backtrace::capture());
        let _ = std::panic::catch_unwind(|| {
            eprintln!("{}", msg);
        });
        #[cfg(target_os = "macos")]
        if let Some(home) = std::env::var_os("HOME") {
            let log_dir = PathBuf::from(home).join("Library/Logs/Unfoldly");
            let _ = std::fs::create_dir_all(&log_dir);
            let _ = std::fs::write(log_dir.join("crash.log"), &msg);
        }
        default_panic(info);
    }));
}

pub struct AppState {
    pub backend: Arc<Py<PyAny>>,
    pub project_root: PathBuf,
    pub shutdown_called: Arc<AtomicBool>,
}

fn shutdown_python_backend_once(
    backend: &Arc<Py<PyAny>>,
    shutdown_called: &Arc<AtomicBool>,
    reason: &str,
) {
    if shutdown_called.swap(true, Ordering::AcqRel) {
        return;
    }
    write_stderr_safe(&format!("[Tauri] Python backend shutdown start: reason={reason}"));
    Python::with_gil(|py| {
        if let Err(e) = backend.call_method0(py, "shutdown") {
            write_stderr_safe(&format!("[Tauri] Python backend shutdown error: {e}"));
        }
    });
    write_stderr_safe(&format!("[Tauri] Python backend shutdown done: reason={reason}"));
}

fn ensure_app_data_dir() {
    if std::env::var("FILEAGENT_DATA_DIR").is_ok() {
        return;
    }
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        _ => return,
    };
    let exe_parent = match exe.parent() {
        Some(p) => p.to_path_buf(),
        _ => return,
    };

    #[cfg(target_os = "macos")]
    let app_data_dir: Option<PathBuf> = {
        let mut cur = exe_parent.clone();
        let mut result = None;
        for _ in 0..5 {
            if let Some(name) = cur.file_name() {
                let s = name.to_string_lossy();
                if s.ends_with(".app") {
                    if let Some(home) = std::env::var_os("HOME") {
                        result = Some(std::path::PathBuf::from(home).join("Library/Application Support/Unfoldly"));
                    } else {
                        if let Some(parent) = cur.parent() {
                            result = Some(parent.to_path_buf());
                        }
                    }
                    break;
                }
            }
            if !cur.pop() {
                break;
            }
        }
        result
    };

    #[cfg(not(target_os = "macos"))]
    let app_data_dir = Some(exe_parent);

    if let Some(app_data_dir) = app_data_dir {
        let s = app_data_dir.to_string_lossy().to_string();
        if !s.is_empty() {
            let _ = std::fs::create_dir_all(&app_data_dir);
            std::env::set_var("FILEAGENT_DATA_DIR", &s);
        }
    }
}

impl AppState {
    pub fn new() -> anyhow::Result<Self> {
        ensure_app_data_dir();

        let project_root = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

        let backend_dir = find_backend_dir(&project_root);

        if !backend_dir.join("backend_core.py").exists() && !backend_dir.join("unfoldly-backend").exists() && !backend_dir.to_string_lossy().contains("site") {
            let exe_path = std::env::current_exe().unwrap_or_default();
            let resources_dir = project_root.join("..").join("Resources");
            let mut extra = String::new();
            if let Ok(entries) = std::fs::read_dir(&resources_dir) {
                let names: Vec<String> = entries
                    .flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect();
                extra = format!("  Resources 下目录/文件: {:?}", names);
            }
            return Err(anyhow::anyhow!(
                "Python backend 模块未找到。\n  查找目录: {}\n  exe 所在: {}\n{}\n  崩溃日志: ~/Library/Logs/Unfoldly/crash.log",
                backend_dir.display(),
                exe_path.display(),
                extra
            ));
        }

        let is_bundled_site = backend_dir
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n == "site")
            .unwrap_or(false)
            && backend_dir
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .map(|n| n == "python_runtime")
                .unwrap_or(false);

        let mut forced_bundled_python = false;
        if is_bundled_site {
            let python_runtime = backend_dir.join("..");
            let bundled_install_python = python_runtime.join("install").join("bin").join("python3");
            let bundled_install = python_runtime.join("install");
            if bundled_install_python.exists() {
                if let Some(p) = bundled_install_python
                    .canonicalize()
                    .ok()
                    .and_then(|p| p.to_str().map(String::from))
                {
                    std::env::set_var("PYO3_PYTHON", &p);
                    forced_bundled_python = true;
                    std::env::remove_var("PYTHONPATH");
                    if let Some(home) = bundled_install
                        .canonicalize()
                        .ok()
                        .and_then(|p| p.to_str().map(String::from))
                    {
                        std::env::set_var("PYTHONHOME", &home);
                    }
                }
            }
        }

        if !forced_bundled_python && std::env::var("PYO3_PYTHON").is_err() {
            let python_runtime = backend_dir.join("..").join("..");
            let bundled_install_python = python_runtime.join("install").join("bin").join("python3");
            let bundled_install = python_runtime.join("install");

            if bundled_install_python.exists() {
                if let Some(p) = bundled_install_python
                    .canonicalize()
                    .ok()
                    .and_then(|p| p.to_str().map(String::from))
                {
                    std::env::set_var("PYO3_PYTHON", &p);
                    if let Some(home) = bundled_install
                        .canonicalize()
                        .ok()
                        .and_then(|p| p.to_str().map(String::from))
                    {
                        std::env::set_var("PYTHONHOME", &home);
                    }
                }
            }

            if std::env::var("PYO3_PYTHON").is_err() {
                let venv_path_file = backend_dir.join("venv_path.txt");
                if let Ok(s) = std::fs::read_to_string(&venv_path_file) {
                    let path = s.trim();
                    if !path.is_empty() && Path::new(path).exists() {
                        std::env::set_var("PYO3_PYTHON", path);
                    }
                }
            }

            #[cfg(target_os = "macos")]
            if std::env::var("PYO3_PYTHON").is_err() {
                for candidate in [
                    "/usr/bin/python3",
                    "/opt/homebrew/bin/python3",
                    "/usr/local/bin/python3",
                ] {
                    if Path::new(candidate).exists() {
                        std::env::set_var("PYO3_PYTHON", candidate);
                        break;
                    }
                }
            }
        }

        let extra_path: Option<PathBuf> = std::env::var("PYO3_PYTHON").ok().and_then(|p| {
            let exe = PathBuf::from(&p);
            let venv_root = exe.parent().and_then(|b| b.parent())?;
            let lib = venv_root.join("lib");
            if let Ok(entries) = std::fs::read_dir(&lib) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if name.starts_with("python") {
                        let site = lib.join(&name).join("site-packages");
                        if site.is_dir() {
                            return Some(site);
                        }
                    }
                }
            }
            None
        });

        if std::env::var("PYTHONUTF8").is_err() {
            std::env::set_var("PYTHONUTF8", "1");
        }
        if std::env::var("LANG").is_err() {
            std::env::set_var("LANG", "en_US.UTF-8");
        }
        if std::env::var("LC_ALL").is_err() {
            std::env::set_var("LC_ALL", "en_US.UTF-8");
        }

        if std::env::var("FILEAGENT_LOG_LEVEL").is_err() {
            #[cfg(debug_assertions)]
            std::env::set_var("FILEAGENT_LOG_LEVEL", "INFO");
            
            #[cfg(not(debug_assertions))]
            std::env::set_var("FILEAGENT_LOG_LEVEL", "INFO");
        }
        
        if std::env::var("FILEAGENT_UVICORN_LOG_LEVEL").is_err() {
            #[cfg(debug_assertions)]
            std::env::set_var("FILEAGENT_UVICORN_LOG_LEVEL", "info");
            
            #[cfg(not(debug_assertions))]
            std::env::set_var("FILEAGENT_UVICORN_LOG_LEVEL", "warning");
        }

        let backend = Python::with_gil(|py| -> PyResult<Py<PyAny>> {
            let sys = py.import("sys")?;
            let path = sys.getattr("path")?;

            let dir_str = backend_dir.to_string_lossy().to_string();
            path.call_method1("insert", (0i32, &dir_str))?;

            if let Some(ref extra) = extra_path {
                let s = extra.to_string_lossy().to_string();
                let _ = path.call_method1("insert", (1i32, s));
            }

            // PyO3 sets sys.executable to the host binary (unfoldly).
            // Python's multiprocessing spawns workers via sys.executable,
            // which would launch duplicate Tauri windows. Fix by pointing
            // both sys.executable and multiprocessing to the real Python.
            let real_python = std::env::var("PYO3_PYTHON").unwrap_or_else(|_| {
                let py_executable: String = py.import("sys")
                    .and_then(|s| s.getattr("prefix"))
                    .and_then(|p| p.extract())
                    .map(|p: String| format!("{}/bin/python3", p))
                    .unwrap_or_else(|_| "python3".into());
                py_executable
            });
            sys.setattr("executable", &real_python)?;

            if let Ok(mp_util) = py.import("multiprocessing.util") {
                let _ = mp_util;
            }
            if let Ok(mp) = py.import("multiprocessing") {
                let _ = mp.call_method1("set_executable", (&real_python,));
            }

            let module = PyModule::import(py, "backend_core")?;
            let backend_class = module.getattr("Backend")?;

            let data_dir = std::env::var("FILEAGENT_DATA_DIR").unwrap_or_default();
            let instance = backend_class.call1((data_dir,))?;

            Ok(instance.unbind())
        })
        .map_err(|e| anyhow::anyhow!("Failed to initialize Python backend: {e}"))?;

        write_python_runtime_log();

        Ok(Self {
            backend: Arc::new(backend),
            project_root: backend_dir,
            shutdown_called: Arc::new(AtomicBool::new(false)),
        })
    }
}

impl Drop for AppState {
    fn drop(&mut self) {
        shutdown_python_backend_once(&self.backend, &self.shutdown_called, "app_state_drop");
    }
}

fn find_backend_dir(start: &PathBuf) -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        let resources_dir = start.join("..").join("Resources");
        let site = resources_dir.join("python_runtime").join("site");
        if site.exists() {
            return std::fs::canonicalize(&site).unwrap_or(site);
        }
    }

    let candidates = [
        start.clone(),
        start.join(".."),
        start.join("../.."),
        start.join("../../.."),
        start.join("../../../.."),
        std::env::current_dir().unwrap_or_default(),
    ];
    for c in &candidates {
        if c.join("backend_core.py").exists() || c.join("pyproject.toml").exists() {
            return std::fs::canonicalize(c).unwrap_or_else(|_| c.clone());
        }
    }
    
    if let Ok(dir) = std::env::var("FILEAGENT_BACKEND_DIR") {
        let p = PathBuf::from(&dir);
        if p.exists() {
            return p;
        }
    }
    std::env::current_dir().unwrap_or_default()
}

#[cfg(target_os = "macos")]
fn write_startup_log() {
    if let Some(home) = std::env::var_os("HOME") {
        let log_dir = PathBuf::from(home).join("Library/Logs/Unfoldly");
        let _ = std::fs::create_dir_all(&log_dir);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let arch = std::env::consts::ARCH;
        let msg = format!(
            "run() entered at {} (arch={})\n若出现「请与开发者联系」且本文件不存在，说明系统在应用启动前就拦截了。\n",
            now, arch
        );
        let _ = std::fs::write(log_dir.join("startup.log"), msg);
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    #[cfg(target_os = "macos")]
    write_startup_log();

    if std::env::var("PYTHONUTF8").is_err() {
        std::env::set_var("PYTHONUTF8", "1");
    }
    // The bundled Python runtime lives inside the signed .app bundle. Never let
    // it create __pycache__ files there, or codesign verification will fail.
    std::env::set_var("PYTHONDONTWRITEBYTECODE", "1");
    std::env::set_var("PYTHONNOUSERSITE", "1");
    if std::env::var("LANG").is_err() {
        std::env::set_var("LANG", "en_US.UTF-8");
    }
    if std::env::var("LC_ALL").is_err() {
        std::env::set_var("LC_ALL", "en_US.UTF-8");
    }

    install_crash_logger();

        let app_state = match AppState::new() {
        Ok(s) => s,
        Err(e) => {
            write_python_runtime_log();
            let msg = format!("[Unfoldly] 启动失败: {}", e);
            eprintln!("{}", msg);
            #[cfg(target_os = "macos")]
            if let Some(home) = std::env::var_os("HOME") {
                let log_dir = PathBuf::from(home).join("Library/Logs/Unfoldly");
                let _ = std::fs::create_dir_all(&log_dir);
                let extra = format!(
                    "\n\nPYO3_PYTHON={}\nPYTHONHOME={}",
                    std::env::var("PYO3_PYTHON").unwrap_or_else(|_| "(未设置)".into()),
                    std::env::var("PYTHONHOME").unwrap_or_else(|_| "(未设置)".into())
                );
                let full = format!("{}{}\n\n排查: 见 python_runtime.log（含 sys.path）。若缺 openai，请删 macos_bundle/python_runtime/venv 后重新执行打包脚本，并确保安装的是本次打出的 .app/DMG。", msg, extra);
                let _ = std::fs::write(log_dir.join("crash.log"), full);
            }
            panic!("{}", msg);
        }
    };

    let shutdown_backend = Arc::clone(&app_state.backend);
    let shutdown_called = Arc::clone(&app_state.shutdown_called);

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_process::init())
        .setup(|app| {
            #[cfg(any(target_os = "macos", windows, target_os = "linux"))]
            app.handle()
                .plugin(tauri_plugin_updater::Builder::new().build())?;

            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            // macOS: allow drag-drop into window even when it's not in focus
            #[cfg(target_os = "macos")]
            {
                use tauri::Manager;
                if let Some(window) = app.get_webview_window("main") {
                    configure_window_for_background_drag(&window);
                    // Register for drag-enter to auto-focus window
                    let win_clone = window.clone();
                    window.on_window_event(move |event| {
                        if let tauri::WindowEvent::DragDrop(tauri::DragDropEvent::Enter { .. }) = event {
                            let _ = win_clone.set_focus();
                        }
                    });
                }
            }
            Ok(())
        })
        .manage(app_state)
        .manage(commands::UpdateDownloadState::default())
        .invoke_handler(tauri::generate_handler![
            commands::health_check,
            commands::get_runtime_paths,
            commands::notify_ui_ready,
            commands::list_models,
            commands::select_model,
            commands::select_quantization,
            commands::download_model,
            commands::cancel_download,
            commands::delete_model,
            commands::core_models_status,
            commands::download_core_models,
            commands::cancel_core_models_download,
            commands::asr_model_status,
            commands::download_asr_model,
            commands::cancel_asr_model_download,
            commands::list_sources,
            commands::add_source,
            commands::remove_source,
            commands::remove_sources_batch,
            commands::refresh_source,
            commands::get_active_job,
            commands::start_index,
            commands::index_files,
            commands::get_index_status,
            commands::cancel_index,
            commands::skip_files,
            commands::get_history,
            commands::sync_history,
            commands::delete_history,
            commands::query,
            commands::abort_query,
            commands::query_stream,
            commands::get_settings,
            commands::update_settings,
            commands::open_file_native,
            commands::open_external_url,
            commands::download_and_install_update_cancelable,
            commands::cancel_update_download,
            commands::reveal_file_native,
            commands::check_media_files,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(move |_app_handle, event| match event {
        tauri::RunEvent::ExitRequested { .. } => {
            shutdown_python_backend_once(&shutdown_backend, &shutdown_called, "exit_requested");
        }
        tauri::RunEvent::Exit => {
            shutdown_python_backend_once(&shutdown_backend, &shutdown_called, "exit");
        }
        _ => {}
    });

    //
    // → C exit() → __cxa_finalize_ranges() → ggml_metal_device_free()
    //
    //
    std::thread::sleep(std::time::Duration::from_millis(200));
    unsafe { libc::_exit(0); }
}
