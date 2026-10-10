'use client';

import ui from '@/components/ui.module.css';

/** Shown when a server component's admin API call fails (gateway down, 5xx). */
export default function DashboardError({ error, reset }: { error: Error; reset: () => void }) {
  return (
    <section className={ui.panel} role="alert">
      <h1 className={ui.panelTitle}>The gateway could not be reached</h1>
      <p className={ui.panelHint}>{error.message || 'Unexpected error while loading this page.'}</p>
      <button type="button" className={ui.button} onClick={reset}>
        Try again
      </button>
    </section>
  );
}
