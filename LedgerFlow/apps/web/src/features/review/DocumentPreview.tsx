import { useEffect, useMemo, useState } from 'react';

import { describeError, useDocumentFile, type DocumentDetail } from '@/api';
import { Button, Callout, Skeleton } from '@/ui';

import styles from './DocumentPreview.module.css';

type PreviewKind = 'pdf' | 'image' | 'xml' | 'unsupported';

/**
 * Only well-known types are rendered, and blobs are re-typed from this
 * allowlist: a blob: URL shares the app's origin, so a file served as
 * text/html must never be loaded into a frame.
 */
const PREVIEW_TYPES: Record<string, PreviewKind> = {
  'application/pdf': 'pdf',
  'image/png': 'image',
  'image/jpeg': 'image',
  'application/xml': 'xml',
  'text/xml': 'xml',
};

function previewKind(contentType: string): PreviewKind {
  const mime = contentType.split(';')[0]?.trim().toLowerCase() ?? '';
  return PREVIEW_TYPES[mime] ?? 'unsupported';
}

function useObjectUrl(blob: Blob | undefined): string | undefined {
  const [url, setUrl] = useState<string>();
  useEffect(() => {
    if (!blob) {
      setUrl(undefined);
      return;
    }
    const objectUrl = URL.createObjectURL(blob);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [blob]);
  return url;
}

function useBlobText(blob: Blob | undefined): string | undefined {
  const [text, setText] = useState<string>();
  useEffect(() => {
    if (!blob) return;
    let cancelled = false;
    void blob.text().then((value) => {
      if (!cancelled) setText(value);
    });
    return () => {
      cancelled = true;
    };
  }, [blob]);
  return text;
}

export function DocumentPreview({ document }: { document: DocumentDetail }) {
  const file = useDocumentFile(document.id);
  const kind = previewKind(document.content_type);
  const renderable = useMemo(() => {
    if (!file.data || kind === 'unsupported' || kind === 'xml') return undefined;
    return new Blob([file.data], {
      type: kind === 'pdf' ? 'application/pdf' : document.content_type,
    });
  }, [file.data, kind, document.content_type]);
  const url = useObjectUrl(renderable);
  const xml = useBlobText(kind === 'xml' ? file.data : undefined);

  if (file.isPending) {
    return (
      <div className={styles.frame} aria-busy="true">
        <Skeleton shape="rect" height="100%" />
        <span className="visually-hidden" role="status">
          Loading document preview
        </span>
      </div>
    );
  }

  if (file.isError) {
    return (
      <Callout
        tone="danger"
        title="The original file could not be loaded"
        actions={
          <Button size="sm" onClick={() => void file.refetch()}>
            Retry
          </Button>
        }
      >
        {describeError(file.error)}
      </Callout>
    );
  }

  switch (kind) {
    case 'pdf':
      return url ? (
        <iframe className={styles.frame} src={url} title={`Preview of ${document.filename}`} />
      ) : null;
    case 'image':
      return url ? (
        <div className={styles.frame}>
          <img className={styles.image} src={url} alt={`Scan of ${document.filename}`} />
        </div>
      ) : null;
    case 'xml':
      return (
        <pre className={styles.xml} aria-label={`XML source of ${document.filename}`}>
          {xml}
        </pre>
      );
    case 'unsupported':
      return (
        <Callout title="No preview available">
          Files of type {document.content_type} cannot be previewed in the browser.
        </Callout>
      );
  }
}
