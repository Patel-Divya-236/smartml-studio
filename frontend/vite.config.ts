import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * The API is proxied rather than called cross-origin in development, so the browser
 * treats it as same-origin and the WebSocket upgrade for training progress works
 * without extra CORS handling.
 *
 * The target is read from the environment so moving the API port (run_dev.py's
 * --api-port) cannot leave the proxy pointing at a port nothing is listening on.
 */
const API_TARGET = process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
        ws: true,
      },
    },
  },
});
