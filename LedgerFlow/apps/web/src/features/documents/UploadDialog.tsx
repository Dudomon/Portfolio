import { useRef, useState } from 'react';

import { describeError, useUploadDocument } from '@/api';
import { Badge, Button, Dialog, FileDropzone, useToast, type FileRejection } from '@/ui';

import styles from './UploadDialog.module.css';

/** Contract: multipart `file`, ≤ 15 MB; pdf, png, jpeg, xml. */
export const UPLOAD_ACCEPT = [
  'application/pdf',
  'image/png',
  'image/jpeg',
  'application/xml',
  'text/xml',
  '.xml',
] as const;
export const UPLOAD_MAX_BYTES = 15 * 1024 * 1024;

type UploadState =
  | { status: 'queued' }
  | { status: 'uploading' }
  | { status: 'done'; duplicate: boolean }
  | { status: 'failed'; message: string };

interface UploadItem {
  key: string;
  file: File;
  state: UploadState;
}

const REJECTION_TEXT: Record<FileRejection['reason'], string> = {
  type: 'unsupported file type',
  size: 'larger than 15 MB',
};

export function UploadDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const upload = useUploadDocument();
  const toast = useToast();
  const [items, setItems] = useState<UploadItem[]>([]);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  function update(key: string, state: UploadState) {
    setItems((current) => current.map((item) => (item.key === key ? { ...item, state } : item)));
  }

  async function uploadAll(files: File[]) {
    const batch = files.map((file) => ({
      key: `${file.name}-${file.size}-${file.lastModified}-${crypto.randomUUID()}`,
      file,
      state: { status: 'queued' } as const,
    }));
    setItems((current) => [...current, ...batch]);
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;

    // Sequential on purpose: keeps progress meaningful and avoids saturating
    // the uplink when someone drops a folder of invoices.
    for (const item of batch) {
      if (controller.signal.aborted) {
        update(item.key, { status: 'failed', message: 'Cancelled' });
        continue;
      }
      update(item.key, { status: 'uploading' });
      try {
        const result = await upload.mutateAsync({ file: item.file, signal: controller.signal });
        update(item.key, { status: 'done', duplicate: result.duplicate });
      } catch (error) {
        const cancelled = error instanceof DOMException && error.name === 'AbortError';
        update(item.key, {
          status: 'failed',
          message: cancelled ? 'Cancelled' : describeError(error),
        });
      }
    }
    abortRef.current = null;
    setBusy(false);
  }

  function onRejected(rejections: FileRejection[]) {
    toast.show({
      tone: 'warning',
      title: `${rejections.length} file${rejections.length === 1 ? '' : 's'} skipped`,
      description: rejections
        .map(({ file, reason }) => `${file.name}: ${REJECTION_TEXT[reason]}`)
        .join('; '),
    });
  }

  function close() {
    setItems([]);
    onClose();
  }

  return (
    <Dialog
      open={open}
      onClose={close}
      title="Upload documents"
      description="Supplier invoices as PDF or image, or Brazilian NF-e XML."
      size="lg"
      dismissible={!busy}
      footer={
        busy ? (
          <Button onClick={() => abortRef.current?.abort()}>Cancel upload</Button>
        ) : (
          <Button variant="primary" onClick={close}>
            Done
          </Button>
        )
      }
    >
      <FileDropzone
        accept={UPLOAD_ACCEPT}
        maxSizeBytes={UPLOAD_MAX_BYTES}
        multiple
        disabled={busy}
        label="Drop invoices here"
        hint="PDF, PNG, JPEG or XML"
        onFilesAccepted={(files) => void uploadAll(files)}
        onFilesRejected={onRejected}
      />
      {items.length > 0 && (
        <ul className={styles.list} aria-label="Uploads">
          {items.map((item) => (
            <li key={item.key} className={styles.item}>
              <span className={styles.name}>{item.file.name}</span>
              <UploadStatus state={item.state} progress={upload.progress} />
            </li>
          ))}
        </ul>
      )}
    </Dialog>
  );
}

function UploadStatus({ state, progress }: { state: UploadState; progress: number }) {
  switch (state.status) {
    case 'queued':
      return <Badge>Waiting</Badge>;
    case 'uploading':
      return (
        <progress
          className={styles.progress}
          value={progress}
          max={1}
          aria-label="Upload progress"
        />
      );
    case 'done':
      return state.duplicate ? (
        <Badge tone="info">Already uploaded</Badge>
      ) : (
        <Badge tone="success">Uploaded</Badge>
      );
    case 'failed':
      return <Badge tone="danger">{state.message}</Badge>;
  }
}
