import path from 'node:path';
import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // Self-contained server bundle for the Docker image.
  output: 'standalone',
  // Trace dependencies from the monorepo root so workspace packages and
  // hoisted node_modules are included in the standalone output.
  outputFileTracingRoot: path.join(__dirname, '../..'),
  poweredByHeader: false,
  eslint: {
    // Linting runs once for the whole monorepo (`npm run lint` at the root)
    // with the shared flat config; running it again here would duplicate it.
    ignoreDuringBuilds: true,
  },
};

export default nextConfig;
