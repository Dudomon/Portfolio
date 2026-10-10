import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm', 'cjs'],
  dts: true,
  clean: true,
  sourcemap: true,
  target: 'es2022',
  // Shared is a runtime dependency; bundling it would duplicate zod schemas
  // in apps that also import @aigw/shared directly.
  external: ['@aigw/shared'],
});
