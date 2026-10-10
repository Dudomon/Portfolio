'use client';

import { useState } from 'react';
import styles from './CreatedKeyNotice.module.css';
import ui from './ui.module.css';

/**
 * Displays a newly created key. The gateway stores only a hash, so this is
 * the single moment the plaintext exists outside the caller's hands; the
 * copy affordance and the warning make that explicit.
 */
export function CreatedKeyNotice({ secret, keyName }: { secret: string; keyName: string }) {
  const [copied, setCopied] = useState(false);

  async function copy(): Promise<void> {
    await navigator.clipboard.writeText(secret);
    setCopied(true);
  }

  return (
    <section className={styles.notice} role="status" aria-labelledby="new-key-title">
      <h2 id="new-key-title" className={styles.title}>
        Key “{keyName}” created
      </h2>
      <p className={styles.warning}>
        Copy it now. It will not be shown again; only a hash is stored.
      </p>
      <div className={styles.secretRow}>
        <code className={styles.secret} aria-label="New API key">
          {secret}
        </code>
        <button type="button" className={ui.buttonSecondary} onClick={() => void copy()}>
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
    </section>
  );
}
