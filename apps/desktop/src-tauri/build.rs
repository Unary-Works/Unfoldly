fn main() {
    tauri_build::build();

    let python = std::env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".into());

    let libdir = if std::env::var("UNFOLDLY_BUNDLE_PYTHON_RPATH").ok().as_deref() == Some("1") {
        let exe = std::path::Path::new(&python);
        exe.parent()
            .and_then(|b| b.parent())
            .map(|p| p.join("lib"))
            .filter(|p| p.exists())
            .and_then(|p| p.canonicalize().ok())
            .and_then(|p| p.to_str().map(String::from))
            .unwrap_or_default()
    } else {
        let out = std::process::Command::new(&python)
            .args(["-c", "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"])
            .output()
            .expect("failed to query python LIBDIR")
            .stdout;
        String::from_utf8_lossy(&out).trim().to_string()
    };

    let fwprefix = if std::env::var("UNFOLDLY_BUNDLE_PYTHON_RPATH").ok().as_deref() != Some("1") {
        let out = std::process::Command::new(&python)
            .args(["-c", "import sysconfig; print(sysconfig.get_config_var('PYTHONFRAMEWORKPREFIX') or '')"])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        if out != "None" { out } else { String::new() }
    } else {
        String::new()
    };

    if !libdir.is_empty() {
        println!("cargo:rustc-link-search=native={libdir}");
        #[cfg(target_os = "macos")]
        {
            if std::env::var("UNFOLDLY_BUNDLE_PYTHON_RPATH").ok().as_deref() == Some("1") {
                println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../Resources/python_runtime/install/lib");
            } else {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
                if !fwprefix.is_empty() && fwprefix != "None" {
                    println!("cargo:rustc-link-arg=-Wl,-rpath,{fwprefix}");
                }
            }
        }
        #[cfg(target_os = "linux")]
        println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
    }
}
