import path from 'path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig(({ mode }) => {
    return {
      base: './',
      clearScreen: false,
      envPrefix: ['VITE_', 'TAURI_ENV_*'],
      plugins: [react(), tailwindcss()],
      define: {
        '__BUILD_DATE__': JSON.stringify(new Date().toISOString().slice(0, 16).replace('T', ' ')),
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      },
      build: {
        outDir: 'dist',
        emptyOutDir: true,
        target: ['es2021', 'chrome100', 'safari14'],
        minify: !process.env.TAURI_ENV_DEBUG ? 'esbuild' : false,
        sourcemap: !!process.env.TAURI_ENV_DEBUG,
        assetsDir: 'assets',
        rollupOptions: {
          input: {
            main: path.resolve(__dirname, 'index.html'),
          },
        },
      },
    };
});
