/**
 * Close the Tauri window; fall back to window.close in dev or non-Tauri builds.
 */
export async function closeAppWindow(): Promise<void> {
  try {
    const { getCurrentWindow } = await import('@tauri-apps/api/window');
    await getCurrentWindow().close();
  } catch {
    try {
      window.close();
    } catch {
      /* ignore */
    }
  }
}
