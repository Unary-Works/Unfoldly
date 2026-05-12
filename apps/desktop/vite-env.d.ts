/// <reference types="vite/client" />

// Build-time constant injected by vite.config.ts → define
declare const __BUILD_DATE__: string;

declare module '*.svg' {
  const content: string;
  export default content;
}

declare module '*.png' {
  const content: string;
  export default content;
}

declare module '*.jpg' {
  const content: string;
  export default content;
}
