import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { createPortal } from 'react-dom';

import { cx } from '@/lib/cx';

import { CloseIcon } from '../icons';
import { IconButton } from '../IconButton/IconButton';
import styles from './Toast.module.css';

export type ToastTone = 'info' | 'success' | 'warning' | 'danger';

export interface ToastOptions {
  title: string;
  description?: string;
  tone?: ToastTone;
  /** Milliseconds before auto-dismiss; `null` keeps the toast until dismissed. */
  durationMs?: number | null;
}

interface ToastRecord extends Required<Omit<ToastOptions, 'description'>> {
  id: number;
  description?: string;
}

interface ToastApi {
  show: (options: ToastOptions) => number;
  dismiss: (id: number) => void;
}

const ToastContext = createContext<ToastApi | null>(null);

const DEFAULT_DURATION_MS = 5_000;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastRecord[]>([]);
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const show = useCallback((options: ToastOptions) => {
    const id = nextId.current++;
    const tone = options.tone ?? 'info';
    setToasts((current) => [
      ...current,
      {
        id,
        title: options.title,
        description: options.description,
        tone,
        // Errors stay until acknowledged so they cannot be missed.
        durationMs:
          options.durationMs !== undefined
            ? options.durationMs
            : tone === 'danger'
              ? null
              : DEFAULT_DURATION_MS,
      },
    ]);
    return id;
  }, []);

  const api = useMemo(() => ({ show, dismiss }), [show, dismiss]);

  const polite = toasts.filter((toast) => toast.tone !== 'danger');
  const assertive = toasts.filter((toast) => toast.tone === 'danger');

  return (
    <ToastContext.Provider value={api}>
      {children}
      {createPortal(
        <section className={styles.viewport} aria-label="Notifications">
          {/* Live regions must exist before content is added to be announced reliably. */}
          <ol className={styles.list} aria-live="assertive" aria-relevant="additions">
            {assertive.map((toast) => (
              <ToastItem key={toast.id} toast={toast} onDismiss={dismiss} />
            ))}
          </ol>
          <ol className={styles.list} aria-live="polite" aria-relevant="additions">
            {polite.map((toast) => (
              <ToastItem key={toast.id} toast={toast} onDismiss={dismiss} />
            ))}
          </ol>
        </section>,
        document.body,
      )}
    </ToastContext.Provider>
  );
}

function ToastItem({ toast, onDismiss }: { toast: ToastRecord; onDismiss: (id: number) => void }) {
  const [paused, setPaused] = useState(false);
  const remaining = useRef(toast.durationMs);

  useEffect(() => {
    if (paused || remaining.current === null) return;
    const startedAt = Date.now();
    const timer = window.setTimeout(() => onDismiss(toast.id), remaining.current);
    return () => {
      window.clearTimeout(timer);
      if (remaining.current !== null) remaining.current -= Date.now() - startedAt;
    };
  }, [paused, toast.id, onDismiss]);

  return (
    <li
      className={cx(styles.toast, styles[toast.tone])}
      // Pausing on hover/focus gives people time to read (WCAG 2.2.1).
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      onFocus={() => setPaused(true)}
      onBlur={() => setPaused(false)}
    >
      <div className={styles.content}>
        <p className={styles.title}>{toast.title}</p>
        {toast.description && <p className={styles.description}>{toast.description}</p>}
      </div>
      <IconButton
        label="Dismiss notification"
        icon={<CloseIcon />}
        size="sm"
        onClick={() => onDismiss(toast.id)}
      />
    </li>
  );
}

export function useToast(): ToastApi {
  const api = useContext(ToastContext);
  if (!api) throw new Error('useToast must be used inside <ToastProvider>');
  return api;
}
