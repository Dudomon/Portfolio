/// <reference types="vitest/config" />
import { fileURLToPath, URL } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const apiTarget = env.VITE_DEV_API_PROXY_TARGET ?? 'http://localhost:8000';

  return {
    plugins: [react()],
    resolve: {
      alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
    },
    server: {
      port: 5173,
      proxy: {
        '/api': { target: apiTarget, changeOrigin: true },
      },
    },
    build: {
      sourcemap: true,
    },
    test: {
      // happy-dom rather than jsdom: jsdom's AbortSignal is rejected by Node's
      // fetch, which MSW intercepts, so cancellable queries would fail in tests.
      environment: 'happy-dom',
      environmentOptions: {
        happyDOM: { settings: { disableIframePageLoading: true } },
      },
      globals: true,
      setupFiles: ['./src/test/setup.ts'],
      restoreMocks: true,
    },
  };
});
