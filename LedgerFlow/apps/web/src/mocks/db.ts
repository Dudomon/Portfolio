import type { AuditEvent, DocumentDetail } from '@/api';

import { createFixtures, type MockDocument } from './fixtures';

/** Mutable in-memory store behind the mock API. `reset()` restores the seed data. */
function createDb() {
  let documents = new Map<string, MockDocument>();

  const db = {
    reset() {
      documents = new Map(createFixtures().map((doc) => [doc.detail.id, doc]));
    },
    all(): DocumentDetail[] {
      return Array.from(documents.values(), (doc) => doc.detail);
    },
    find(id: string): MockDocument | undefined {
      return documents.get(id);
    },
    findBySha(sha256: string): MockDocument | undefined {
      return Array.from(documents.values()).find((doc) => doc.detail.sha256 === sha256);
    },
    insert(doc: MockDocument) {
      documents.set(doc.detail.id, doc);
    },
    /** Applies an update, bumping `version` and `updated_at` like the real API. */
    update(id: string, change: (detail: DocumentDetail) => DocumentDetail, event?: AuditEvent) {
      const doc = documents.get(id);
      if (!doc) throw new Error(`Unknown mock document ${id}`);
      const next = change(doc.detail);
      doc.detail = {
        ...next,
        version: doc.detail.version + 1,
        updated_at: new Date().toISOString(),
      };
      if (event) doc.audit.push(event);
      return doc.detail;
    },
    count: () => documents.size,
  };
  db.reset();
  return db;
}

export const db = createDb();
