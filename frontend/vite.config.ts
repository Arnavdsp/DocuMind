import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  // Relative asset URLs. The built app is served by FastAPI itself
  // (StaticFiles mounted at "/"), and reached through whatever public
  // tunnel Colab is using, so it must not assume it lives at an absolute
  // origin root.
  base: "./",
  build: {
    target: "es2020",
    rollupOptions: {
      output: {
        // The scene is already a lazy chunk; keeping React separate means a
        // shader edit doesn't invalidate the vendor cache entry.
        manualChunks: (id: string) =>
          id.includes("node_modules/react") ? "react" : undefined,
      },
    },
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/readiness": "http://127.0.0.1:8000",
    },
  },
});
