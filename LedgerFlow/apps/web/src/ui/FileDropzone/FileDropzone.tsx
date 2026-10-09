import { useId, useState, type ChangeEvent, type DragEvent } from 'react';

import { cx } from '@/lib/cx';
import { formatFileSize } from '@/lib/format';

import { UploadIcon } from '../icons';
import styles from './FileDropzone.module.css';

export interface FileRejection {
  file: File;
  reason: 'type' | 'size';
}

export interface FileDropzoneProps {
  /** MIME types (`application/pdf`) and/or extensions (`.xml`). */
  accept: readonly string[];
  maxSizeBytes?: number;
  multiple?: boolean;
  disabled?: boolean;
  label?: string;
  hint?: string;
  onFilesAccepted: (files: File[]) => void;
  onFilesRejected?: (rejections: FileRejection[]) => void;
}

export function matchesAccept(file: File, accept: readonly string[]): boolean {
  const name = file.name.toLowerCase();
  const type = file.type.toLowerCase();
  return accept.some((rule) => {
    const normalized = rule.toLowerCase();
    if (normalized.startsWith('.')) return name.endsWith(normalized);
    if (normalized.endsWith('/*')) return type.startsWith(normalized.slice(0, -1));
    return type === normalized;
  });
}

/**
 * Drag-and-drop target backed by a real file input. The input is visually
 * hidden but stays focusable, so keyboard and screen-reader users get the
 * native picker; the visible "browse" text is its label.
 */
export function FileDropzone({
  accept,
  maxSizeBytes,
  multiple = false,
  disabled = false,
  label = 'Drop files here',
  hint,
  onFilesAccepted,
  onFilesRejected,
}: FileDropzoneProps) {
  const inputId = useId();
  const hintId = useId();
  const [dragging, setDragging] = useState(false);

  function handleFiles(fileList: FileList | null) {
    if (!fileList || disabled) return;
    const files = multiple ? Array.from(fileList) : Array.from(fileList).slice(0, 1);
    const accepted: File[] = [];
    const rejected: FileRejection[] = [];
    for (const file of files) {
      if (!matchesAccept(file, accept)) rejected.push({ file, reason: 'type' });
      else if (maxSizeBytes !== undefined && file.size > maxSizeBytes)
        rejected.push({ file, reason: 'size' });
      else accepted.push(file);
    }
    if (accepted.length > 0) onFilesAccepted(accepted);
    if (rejected.length > 0) onFilesRejected?.(rejected);
  }

  function onDragOver(event: DragEvent<HTMLDivElement>) {
    if (disabled) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
    setDragging(true);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    handleFiles(event.dataTransfer.files);
  }

  function onChange(event: ChangeEvent<HTMLInputElement>) {
    handleFiles(event.target.files);
    // Allow picking the same file again after a rejection.
    event.target.value = '';
  }

  const hintText = [
    hint,
    maxSizeBytes !== undefined ? `Up to ${formatFileSize(maxSizeBytes)}` : undefined,
  ]
    .filter(Boolean)
    .join(' · ');

  return (
    <div
      className={cx(styles.zone, dragging && styles.dragging, disabled && styles.disabled)}
      onDragOver={onDragOver}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <UploadIcon className={styles.icon} />
      <p className={styles.label}>{label}</p>
      <p>
        or{' '}
        <input
          id={inputId}
          type="file"
          className={cx('visually-hidden', styles.input)}
          accept={accept.join(',')}
          multiple={multiple}
          disabled={disabled}
          aria-describedby={hintText ? hintId : undefined}
          onChange={onChange}
        />
        <label htmlFor={inputId} className={styles.browse}>
          browse files
        </label>
      </p>
      {hintText && (
        <p id={hintId} className={styles.hint}>
          {hintText}
        </p>
      )}
    </div>
  );
}
