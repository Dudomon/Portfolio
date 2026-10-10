import { existsSync } from 'node:fs';

// Local development convenience: load apps/gateway/.env when present. In
// containers configuration comes from the real environment, which always
// takes precedence because loadEnvFile never overrides existing variables.
if (existsSync('.env')) process.loadEnvFile('.env');
