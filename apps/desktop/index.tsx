import React, { StrictMode } from 'react';
import ReactDOM from 'react-dom/client';

window.addEventListener("error", function(e) { window.alert("JS ERROR: " + (e.error?.stack || e.message)); });
window.addEventListener("unhandledrejection", function(e) { window.alert("PROMISE ERROR: " + (e.reason?.stack || e.reason)); });

/**
 * First paint: render a lightweight blank shell before importing App and i18n
 * so large dependencies do not block the initial frame.
 */
function FirstPaintShell() {
  return (
    <div className="h-full w-full bg-white" />
  );
}

function AppLoadError({ message }: { message: string }) {
  return (
    <div className="flex h-full w-full items-center justify-center bg-white p-6 text-red-600 text-sm">
      无法加载应用界面：{message}
    </div>
  );
}

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error('Could not find root element to mount to');
}

const root = ReactDOM.createRoot(rootElement);
root.render(<FirstPaintShell />);

void import('./i18n')
  .then(async (i18nModule) => {
    if (typeof i18nModule.initializeI18n === 'function') {
      await i18nModule.initializeI18n();
    }
  })
  .then(() => import('./App'))
  .then((module) => {
    const App = module.default;
    root.render(
      <StrictMode>
        <App />
      </StrictMode>
    );
  })
  .catch((e) => {
    console.error(e);
    root.render(<AppLoadError message={String(e?.message || e)} />);
  });
