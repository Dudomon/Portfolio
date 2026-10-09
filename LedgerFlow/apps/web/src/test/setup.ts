import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';

import { db } from '@/mocks/db';
import { server } from '@/mocks/server';

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => {
  cleanup();
  server.resetHandlers();
  db.reset();
});
afterAll(() => server.close());
