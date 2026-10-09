import { useRef } from 'react';

import { Button, Dialog } from '@/ui';

/** Shown when the server answers 412: someone changed the document after it was loaded here. */
export function ConflictDialog({
  open,
  reloading,
  onReload,
  onDismiss,
}: {
  open: boolean;
  reloading: boolean;
  onReload: () => void;
  onDismiss: () => void;
}) {
  const reloadRef = useRef<HTMLButtonElement>(null);
  return (
    <Dialog
      open={open}
      onClose={onDismiss}
      role="alertdialog"
      size="sm"
      title="This document was changed by someone else"
      description="Your action was not applied. Reload to see the latest version; unsaved edits on this page will be discarded."
      initialFocusRef={reloadRef}
      dismissible={!reloading}
      footer={
        <>
          <Button onClick={onDismiss} disabled={reloading}>
            Keep my edits
          </Button>
          <Button ref={reloadRef} variant="primary" loading={reloading} onClick={onReload}>
            Reload latest
          </Button>
        </>
      }
    />
  );
}
