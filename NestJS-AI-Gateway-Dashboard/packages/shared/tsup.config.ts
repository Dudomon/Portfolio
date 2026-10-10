import { defineConfig } from 'tsup';

// Dual ESM/CJS output: the NestJS gateway compiles to CommonJS while the
// dashboard and browser SDK consumers prefer ESM.
export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm', 'cjs'],
  dts: true,
  clean: true,
  sourcemap: true,
  target: 'es2022',
});
