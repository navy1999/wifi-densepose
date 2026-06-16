import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, proxy API + WebSocket to the FastAPI server so the frontend can call
// relative paths (/api/...) exactly as it will in production (served by FastAPI).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true, ws: true },
      "/metrics": "http://localhost:8000",
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
