'use client';

import { useTransition } from 'react';
import ui from './ui.module.css';

export function RevokeKeyButton({
  keyName,
  onRevoke,
}: {
  keyName: string;
  onRevoke: () => Promise<void>;
}) {
  const [pending, startTransition] = useTransition();

  return (
    <button
      type="button"
      className={ui.buttonDanger}
      disabled={pending}
      aria-label={`Revoke key ${keyName}`}
      onClick={() => {
        // Revocation is irreversible and breaks live clients; confirm first.
        if (!window.confirm(`Revoke "${keyName}"? Clients using it will start receiving 401s.`))
          return;
        startTransition(() => onRevoke());
      }}
    >
      {pending ? 'Revoking…' : 'Revoke'}
    </button>
  );
}
