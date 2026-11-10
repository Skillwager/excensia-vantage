/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

// One self-contained dist/index.html: every script, style and image is inlined.
// The only outside request the page makes is the Google Fonts stylesheet.
export default defineConfig({
  base: "./",
  plugins: [react(), viteSingleFile({ removeViteModuleLoader: true })],
  build: {
    target: "es2020",
    cssCodeSplit: false,
    assetsInlineLimit: 100_000_000,
    chunkSizeWarningLimit: 1500,
    reportCompressedSize: false,
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
